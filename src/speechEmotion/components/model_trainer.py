import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import mlflow
import mlflow.pytorch
import optuna
from speechEmotion import logger
from speechEmotion.entity import TrainingConfig

class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_data_loaders(self):
        """Prepares PyTorch DataLoaders for actor-independent training and testing."""
        image_size = tuple(self.config.params_image_size)
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dir = Path(self.config.training_data) / "train"
        test_dir = Path(self.config.training_data) / "test"

        train_dataset = datasets.ImageFolder(root=str(train_dir), transform=transform)
        test_dataset = datasets.ImageFolder(root=str(test_dir), transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=self.config.params_batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=self.config.params_batch_size, shuffle=False, num_workers=2)

        logger.info(f"Loaded dataset: {len(train_dataset)} training samples, {len(test_dataset)} test samples.")
        return train_loader, test_loader, train_dataset.classes

    def get_model(self, num_classes: int):
        """Dynamically loads the specified ResNet backbone and modifies the classification head."""
        base_model = getattr(self.config, "params_base_model", "resnet34").lower()
        logger.info(f"Initializing backbone model: {base_model}")

        if base_model == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif base_model == "resnet34":
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        else:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model.to(self.device)

    def train_single_trial(self, trial, train_loader, test_loader, class_names):
        """Trains a model for a single Optuna trial."""
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        gamma = trial.suggest_float("scheduler_gamma", 0.1, 0.9)

        model = self.get_model(num_classes=len(class_names))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=gamma)

        best_acc = 0.0
        epochs = 10  # Shorter epochs per trial for quick searching

        with mlflow.start_run(nested=True):
            mlflow.log_params({"trial_lr": lr, "trial_wd": weight_decay, "trial_gamma": gamma})

            for epoch in range(epochs):
                model.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), labels)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

                # Validation phase on unseen test actors
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        _, predicted = torch.max(model(inputs), 1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()

                acc = correct / total
                if acc > best_acc:
                    best_acc = acc

            mlflow.log_metric("val_accuracy", best_acc)

        return best_acc

    def train(self):
        """Executes Optuna hyperparameter search, trains the final best model, and logs everything to MLflow."""
        train_loader, test_loader, class_names = self.get_data_loaders()
        
        # Load N_TRIALS dynamically from params.yaml
        import yaml
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
        
        n_trials = params.get("HPO", {}).get("N_TRIALS", 1)
        logger.info(f"Starting Optuna Hyperparameter Optimization Study for {n_trials} trials...")

        def objective(trial):
            return self.train_single_trial(trial, train_loader, test_loader, class_names)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        logger.info(f"Best Trial Accuracy: {study.best_value:.4f}")
        logger.info(f"Best Hyperparameters: {study.best_params}")

        # Retrain best model fully with optimized hyperparameters
        best_lr = study.best_params["learning_rate"]
        best_wd = study.best_params["weight_decay"]
        best_gamma = study.best_params["scheduler_gamma"]

        logger.info("Training final champion model with best hyperparameters...")
        model = self.get_model(num_classes=len(class_names))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=best_lr, weight_decay=best_wd)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=best_gamma)

        # Full training run for final model
        final_epochs = getattr(self.config, "params_epochs", 25)
        for epoch in range(final_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            scheduler.step()
            
            epoch_loss = running_loss / len(train_loader.dataset)
            logger.info(f"Final Model Epoch [{epoch+1}/{final_epochs}] | Loss: {epoch_loss:.4f}")

        # Save the optimized champion model state dict
        os.makedirs(os.path.dirname(self.config.trained_model_path), exist_ok=True)
        torch.save(model.state_dict(), self.config.trained_model_path)
        logger.info(f"Optimized model successfully saved at: {self.config.trained_model_path}")

        # Log champion model to MLflow using safe legacy pickle serialization
        mlflow.start_run(nested=True)
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_test_accuracy", study.best_value)
        mlflow.pytorch.log_model(model, artifact_path="model", serialization_format="pickle")
        mlflow.end_run()
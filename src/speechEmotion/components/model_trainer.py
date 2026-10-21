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

        return train_loader, test_loader, train_dataset.classes

    def train_single_trial(self, trial, train_loader, test_loader, class_names):
        """Trains a model for a single Optuna trial."""
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        gamma = trial.suggest_float("scheduler_gamma", 0.1, 0.9)

        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=gamma)

        best_acc = 0.0
        epochs = 12  # Quick epochs per trial

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

                # Validation phase on test actors
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
        """Executes Optuna hyperparameter search and saves the best model."""
        train_loader, test_loader, class_names = self.get_data_loaders()
        
        logger.info("Starting Optuna Hyperparameter Optimization Study...")

        def objective(trial):
            return self.train_single_trial(trial, train_loader, test_loader, class_names)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)

        logger.info(f"Best Trial Accuracy: {study.best_value:.4f}")
        logger.info(f"Best Hyperparameters: {study.best_params}")

        # Retrain best model fully with best parameters
        best_lr = study.best_params["learning_rate"]
        best_wd = study.best_params["weight_decay"]
        best_gamma = study.best_params["scheduler_gamma"]

        logger.info("Training final model with best hyperparameters...")
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=best_lr, weight_decay=best_wd)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=best_gamma)

        # Final longer training run
        for epoch in range(25):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Save the optimized best model
        os.makedirs(os.path.dirname(self.config.trained_model_path), exist_ok=True)
        torch.save(model.state_dict(), self.config.trained_model_path)
        logger.info(f"Optimized model saved at: {self.config.trained_model_path}")
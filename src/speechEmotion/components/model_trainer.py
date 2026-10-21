import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from pathlib import Path
import mlflow
import mlflow.pytorch
from speechEmotion import logger
from speechEmotion.entity import TrainingConfig

class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_data_loaders(self):
        """Prepares PyTorch DataLoaders for training and testing from actor-independent split folders."""
        image_size = tuple(self.config.params_image_size)
        
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Define paths for train and test splits based on the feature extraction output root
        train_dir = Path(self.config.training_data) / "train"
        test_dir = Path(self.config.training_data) / "test"

        # Load datasets using ImageFolder
        train_dataset = datasets.ImageFolder(root=str(train_dir), transform=transform)
        test_dataset = datasets.ImageFolder(root=str(test_dir), transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=self.config.params_batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=self.config.params_batch_size, shuffle=False, num_workers=2)

        logger.info(f"Actor-independent dataset loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples.")
        return train_loader, test_loader, train_dataset.classes

    def get_model(self, num_classes: int):
        """Initializes the pretrained backbone model and modifies the classification head."""
        if self.config.params_base_model == "resnet34":
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        else:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modify final fully connected layer for RAVDESS 8 emotions
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model.to(self.device)

    def train(self):
        """Executes the full training loop with MLflow tracking and saves the model."""
        train_loader, val_loader, class_names = self.get_data_loaders()
        model = self.get_model(num_classes=len(class_names))

        criterion = nn.CrossEntropyLoss()
        
        # Optimizer selection
        if self.config.params_optimizer.lower() == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=self.config.params_learning_rate, momentum=0.9, weight_decay=self.config.params_weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=self.config.params_learning_rate, weight_decay=self.config.params_weight_decay)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=self.config.params_scheduler_gamma)

        # Start MLflow Run
        mlflow.start_run()
        mlflow.log_params({
            "epochs": self.config.params_epochs,
            "batch_size": self.config.params_batch_size,
            "learning_rate": self.config.params_learning_rate,
            "optimizer": self.config.params_optimizer,
            "weight_decay": self.config.params_weight_decay,
            "base_model": self.config.params_base_model
        })

        logger.info(f"Starting training on device: {self.device} for {self.config.params_epochs} epochs...")

        for epoch in range(self.config.params_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            scheduler.step()
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = correct / total

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_epoch_loss = val_loss / len(val_loader.dataset)
            val_epoch_acc = val_correct / val_total

            logger.info(f"Epoch [{epoch+1}/{self.config.params_epochs}] | "
                        f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
                        f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}")

            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss": epoch_loss,
                "train_accuracy": epoch_acc,
                "val_loss": val_epoch_loss,
                "val_accuracy": val_epoch_acc
            }, step=epoch)

        # Save trained model state dictionary
        os.makedirs(os.path.dirname(self.config.trained_model_path), exist_ok=True)
        torch.save(model.state_dict(), self.config.trained_model_path)
        logger.info(f"Trained model successfully saved at: {self.config.trained_model_path}")

        
        mlflow.pytorch.log_model(model, name="model", serialization_format="pickle")
        mlflow.end_run()
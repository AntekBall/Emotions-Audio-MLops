import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import json
from speechEmotion import logger
from speechEmotion.entity import EvaluationConfig
from speechEmotion.utils.common import save_json

class ModelEvaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_model(self, num_classes: int):
        """Dynamically loads the correct ResNet backbone matching training configuration."""
        base_model = self.config.all_params.get("BASE_MODEL", "resnet34").lower()
        logger.info(f"Evaluation loading backbone model: {base_model}")

        if base_model == "resnet50":
            model = models.resnet50(weights=None)
        elif base_model == "resnet34":
            model = models.resnet34(weights=None)
        else:
            model = models.resnet18(weights=None)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model.to(self.device)

    def evaluate(self):
        """Loads the trained model and evaluates it against the test dataset."""
        image_size = tuple(self.config.image_size)
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_dir = Path(self.config.training_data) / "test"
        test_dataset = datasets.ImageFolder(root=str(test_dir), transform=transform)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=0
        )

        num_classes = len(test_dataset.classes)
        model = self._get_model(num_classes=num_classes)

        # Load saved weights
        logger.info(f"Loading model weights from: {self.config.model_path}")
        state_dict = torch.load(self.config.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()

        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_loss = running_loss / total
        test_accuracy = correct / total

        logger.info(f"Evaluation Results -> Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

        # Save metrics JSON for DVC tracking
        scores = {"loss": test_loss, "accuracy": test_accuracy}
        save_json(path=Path(self.config.metric_file_name), data=scores)
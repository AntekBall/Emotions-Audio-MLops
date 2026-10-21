import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from pathlib import Path
import mlflow
import mlflow.pytorch
from sklearn.metrics import classification_report, accuracy_score
from speechEmotion import logger
from speechEmotion.entity import EvaluationConfig
from speechEmotion.utils.common import save_json

class ModelEvaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def eval_metrics(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

    def evaluate(self):
        """Loads trained model, runs inference on actor-independent test split, computes metrics, and logs to MLflow."""
        image_size = tuple(self.config.image_size)
        
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Point directly to the actor-independent test split folder
        test_dir = Path(self.config.training_data) / "test"
        
        test_dataset = datasets.ImageFolder(root=str(test_dir), transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=2)

        # Load ResNet model architecture and weights (explicitly 8 classes for RAVDESS)
        num_classes = 8
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
        model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        criterion = nn.CrossEntropyLoss()
        
        all_preds = []
        all_labels = []
        total_loss = 0.0

        logger.info("Running evaluation on actor-independent test dataset...")
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss = total_loss / len(test_loader.dataset)
        accuracy = self.eval_metrics(all_labels, all_preds)

        scores = {
            "val_loss": float(test_loss),
            "val_accuracy": float(accuracy)
        }
        
        # Save metrics to JSON for DVC tracking
        save_json(path=Path(self.config.metric_file_name), data=scores)
        logger.info(f"Evaluation metrics saved to {self.config.metric_file_name}")

        # Log metrics to MLflow
        mlflow.start_run(nested=True)
        mlflow.log_metrics(scores)
        mlflow.end_run()

        logger.info(f"Test Loss: {test_loss:.4f} | Test Accuracy: {accuracy:.4f}")
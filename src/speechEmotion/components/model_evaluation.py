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
        """Loads trained model, runs inference on validation split, computes metrics, and logs to MLflow."""
        image_size = tuple(self.config.image_size)
        
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = datasets.ImageFolder(root=str(self.config.training_data), transform=transform)
        class_names = dataset.classes

        # Same 80-20 Split used during training to access validation subset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        _, val_dataset = random_split(dataset, [train_size, val_size])

        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=2)

        # Load ResNet model architecture and weights
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))
        
        model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        criterion = nn.CrossEntropyLoss()
        
        all_preds = []
        all_labels = []
        total_loss = 0.0

        logger.info("Running evaluation on validation dataset...")
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = total_loss / len(val_loader.dataset)
        accuracy = self.eval_metrics(all_labels, all_preds)

        scores = {
            "val_loss": float(val_loss),
            "val_accuracy": float(accuracy)
        }
        
        # Save metrics to JSON for DVC tracking
        save_json(path=Path(self.config.metric_file_name), data=scores)
        logger.info(f"Evaluation metrics saved to {self.config.metric_file_name}")

        # Log metrics to MLflow
        mlflow.start_run(nested=True)
        mlflow.log_metrics(scores)
        mlflow.end_run()

        logger.info(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {accuracy:.4f}")
from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class FeatureExtractionConfig:
    root_dir: Path
    spectrogram_dir: Path
    image_size: list
    sample_rate: int
    n_mels: int

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_image_size: list
    params_learning_rate: float
    params_optimizer: str
    params_weight_decay: float
    params_base_model: str
    params_scheduler_gamma: float

@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    model_path: Path
    metric_file_name: Path
    training_data: Path
    all_params: dict
    image_size: list
    batch_size: int
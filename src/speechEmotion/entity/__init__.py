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
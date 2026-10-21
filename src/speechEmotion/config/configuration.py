from pathlib import Path
from speechEmotion.constants import *
from speechEmotion.utils.common import read_yaml, create_directories
from speechEmotion.entity import (FeatureExtractionConfig, TrainingConfig, EvaluationConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = Path("config/config.yaml"),
        params_filepath = Path("params.yaml")
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([Path(self.config.artifacts_root)])

    def get_feature_extraction_config(self) -> FeatureExtractionConfig:
        config = self.config.feature_extraction
        create_directories([Path(config.root_dir)])

        feature_extraction_config = FeatureExtractionConfig(
            root_dir=Path(config.root_dir),
            spectrogram_dir=Path(config.spectrogram_dir),
            image_size=self.params.IMAGE_SIZE,
            sample_rate=config.sample_rate,
            n_mels=config.n_mels
        )
        return feature_extraction_config

    def get_training_config(self) -> TrainingConfig:
        training = self.config.model_trainer
        params = self.params
        feature_extraction = self.config.feature_extraction
        
        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(training.trained_model_path), # placeholder
            training_data=Path(feature_extraction.spectrogram_dir),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE,
            params_optimizer=params.OPTIMIZER,
            params_weight_decay=params.WEIGHT_DECAY,
            params_base_model=params.BASE_MODEL,
            params_scheduler_gamma=params.SCHEDULER_GAMMA
        )
        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = self.config.evaluation
        feature_extraction = self.config.feature_extraction
        
        create_directories([Path(eval_config.root_dir)])

        evaluation_config = EvaluationConfig(
            root_dir=Path(eval_config.root_dir),
            model_path=Path(eval_config.model_path),
            metric_file_name=Path(eval_config.metric_file_name),
            training_data=Path(feature_extraction.spectrogram_dir),
            all_params=self.params,
            image_size=self.params.IMAGE_SIZE,
            batch_size=self.params.BATCH_SIZE
        )
        return evaluation_config
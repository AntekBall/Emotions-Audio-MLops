import logging as logger
#from src.speechEmotion.utils import logger    

STAGE_NAMES = [
    "Data Preprocessing & Feature Extraction",
    "Model Training",
    "Model Evaluation"
]

if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info("🚀 Starting MLOps Pipeline Orchestration...")
        logger.info("=" * 60)

        # -------------------------------------------------------------
        # STAGE 1: Data Ingestion / Feature Extraction
        # -------------------------------------------------------------
        try:
            logger.info(f">>>>>> Stage 1: {STAGE_NAMES[0]} started <<<<<<")
            
            
            from src.speechEmotion.pipeline.stage_01_data_transformation import DataTransformationPipeline
            obj = DataTransformationPipeline()
            obj.main()
            logger.info(f">>>>>> Stage 1: {STAGE_NAMES[0]} completed <<<<<<\n" + "=" * 60)
        except Exception as e:
            logger.error(f"❌ Stage 1 failed: {e}")
            raise e

        # -------------------------------------------------------------
        # STAGE 2: Model Training
        # -------------------------------------------------------------
        try:
            logger.info(f">>>>>> Stage 2: {STAGE_NAMES[1]} started <<<<<<")
            
            
            from src.speechEmotion.pipeline.stage_02_model_trainer import ModelTrainingPipeline
            obj = ModelTrainingPipeline()
            obj.main()
            logger.info(f">>>>>> Stage 2: {STAGE_NAMES[1]} completed <<<<<<\n" + "=" * 60)
        except Exception as e:
            logger.error(f"❌ Stage 2 failed: {e}")
            raise e

        # -------------------------------------------------------------
        # STAGE 3: Model Evaluation
        # -------------------------------------------------------------
        try:
            logger.info(f">>>>>> Stage 3: {STAGE_NAMES[2]} started <<<<<<")
            from src.speechEmotion.pipeline.stage_03_model_evaluation import ModelEvaluationPipeline
            obj = ModelEvaluationPipeline()
            obj.main()
            logger.info(f">>>>>> Stage 3: {STAGE_NAMES[2]} completed <<<<<<\n" + "=" * 60)
        except Exception as e:
            logger.error(f"❌ Stage 3 failed: {e}")
            raise e

        logger.info("🎉 All pipeline stages completed successfully!")

    except Exception as e:
        logger.exception(f"💥 Pipeline crashed due to critical error: {e}")
        raise e
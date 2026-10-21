from speechEmotion.config.configuration import ConfigurationManager
from speechEmotion.components.feature_extraction import FeatureExtraction 
from speechEmotion import logger

STAGE_NAME = "Data Transformation & Feature Extraction Stage"

class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        feature_extraction_config = config.get_feature_extraction_config()
        feature_extraction = FeatureExtraction(config=feature_extraction_config)
        feature_extraction.extract_features() 

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
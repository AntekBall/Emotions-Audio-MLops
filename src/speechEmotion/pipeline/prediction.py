from speechEmotion.components.prediction import PredictionComponent
from speechEmotion import logger

class PredictionPipeline:
    def __init__(self, filename: str):
        self.filename = filename

    def predict(self) -> dict:
        predictor = PredictionComponent()
        result = predictor.predict_audio(self.filename)
        return result
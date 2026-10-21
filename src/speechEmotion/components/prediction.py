import os
import torch
import torch.nn as nn
from torchvision import transforms, models
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from speechEmotion import logger

EMOTION_MAP = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised"
}

class PredictionComponent:
    def __init__(self, model_path: str = "artifacts/training/model.pth", image_size: tuple = (224, 224)):
        self.model_path = model_path
        self.image_size = image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self) -> nn.Module:
        """Loads ResNet18 architecture and populates trained weights."""
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(EMOTION_MAP))
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")

        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def wav_to_spectrogram_image(self, audio_path: str, temp_img_path: str = "temp_spec.png") -> str:
        """Converts an input .wav file to a temporary Mel-spectrogram PNG."""
        y, sr = librosa.load(audio_path, sr=None)
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)

        plt.figure(figsize=(3, 3))
        plt.axis('off')
        librosa.display.specshow(mel_spect_db, sr=sr, fmax=8000)
        plt.savefig(temp_img_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return temp_img_path

    def predict_audio(self, audio_path: str) -> dict:
        """Runs end-to-end inference on a single raw audio file."""
        temp_img = self.wav_to_spectrogram_image(audio_path)
        
        image = Image.open(temp_img).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        # Cleanup temporary image file
        if os.path.exists(temp_img):
            os.remove(temp_img)

        predicted_emotion = EMOTION_MAP[predicted_idx.item()]
        confidence_score = float(confidence.item())

        logger.info(f"Audio Path: {audio_path} | Predicted Emotion: {predicted_emotion} ({confidence_score * 100:.2f}%)")
        return {"emotion": predicted_emotion, "confidence": confidence_score}
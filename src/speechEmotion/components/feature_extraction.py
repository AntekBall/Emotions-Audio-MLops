import os
import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from speechEmotion import logger
from speechEmotion.entity import FeatureExtractionConfig

class FeatureExtraction:
    def __init__(self, config: FeatureExtractionConfig):
        self.config = config

    def extract_features(self):
        """
        Loops through RAVDESS dataset audio files, converts them to Mel-spectrograms,
        and saves them into emotion-labeled directories for PyTorch 2D CNN training.
        """
        emotion_map = {
            "01": "neutral",
            "02": "calm",
            "03": "happy",
            "04": "sad",
            "05": "angry",
            "06": "fearful",
            "07": "disgust",
            "08": "surprised"
        }

        actor_directories = glob.glob("data/Actor_*")
        if not actor_directories:
            logger.warning("No Actor directories found in data/ folder!")
            return

        logger.info(f"Starting feature extraction for {len(actor_directories)} actors...")

        total_processed = 0
        for actor_dir in actor_directories:
            audio_files = glob.glob(os.path.join(actor_dir, "*.wav"))
            for audio_file in audio_files:
                filename = os.path.basename(audio_file)
                parts = filename.split("-")
                if len(parts) < 3:
                    continue
                emotion_code = parts[2]
                emotion_label = emotion_map.get(emotion_code, "unknown")
                if emotion_label == "unknown":
                    continue

                # Define output path: artifacts/feature_extraction/spectrograms/{emotion_label}/
                out_dir = self.config.spectrogram_dir / emotion_label
                os.makedirs(out_dir, exist_ok=True)
                
                file_stem = Path(filename).stem
                output_image_path = out_dir / f"{file_stem}.png"

                # Skip if already processed to save time on re-runs
                if output_image_path.exists():
                    continue

                try:
                    # Load audio with librosa
                    y, sr = librosa.load(audio_file, sr=self.config.sample_rate)
                    
                    # Compute Mel-spectrogram
                    mel_spectrogram = librosa.feature.melspectrogram(
                        y=y, sr=sr, n_mels=self.config.n_mels
                    )
                    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

                    # Plot and save as image without borders
                    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
                    ax.axis('off')
                    librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis=None, y_axis=None, ax=ax)
                    
                    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)
                    total_processed += 1
                except Exception as e:
                    logger.error(f"Error processing {audio_file}: {e}")

        logger.info(f"Feature extraction complete! Successfully processed and saved {total_processed} spectrogram images.")
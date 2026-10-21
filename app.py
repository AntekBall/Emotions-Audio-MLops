import streamlit as st
import os
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms, models
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Speech Emotion Recognition MLOps",
    page_icon="🎙️",
    layout="centered"
)

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

@st.cache_resource
def load_model():
    """Loads the trained ResNet18 model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "artifacts/training/model.pth"
    
    if not os.path.exists(model_path):
        st.error(f"Model checkpoint not found at {model_path}. Please run training first!")
        return None, device

    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 8)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# App UI Header
st.title("🎙️ Speech Emotion Recognition (SER)")
st.markdown("""
This application uses a fine-tuned **ResNet18** deep learning model trained on Mel-spectrograms to detect emotions from speech audio files. 
Built as part of an end-to-end production-grade **MLOps pipeline** (PyTorch, MLflow, DVC).
""")

st.markdown("---")

# File Uploader
uploaded_file = st.file_uploader("Upload a speech audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_audio_path = "temp_uploaded.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Audio Playback")
        st.audio(temp_audio_path, format="audio/wav")

    with col2:
        st.subheader("Mel-Spectrogram")
        # Generate Spectrogram Visual
        y, sr = librosa.load(temp_audio_path, sr=None)
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.axis('off')
        librosa.display.specshow(mel_spect_db, sr=sr, fmax=8000, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    if model is not None:
        with st.spinner("Analyzing emotion..."):
            # Process Spectrogram Image for Model
            temp_img_path = "temp_spec.png"
            plt.figure(figsize=(3, 3))
            plt.axis('off')
            librosa.display.specshow(mel_spect_db, sr=sr, fmax=8000)
            plt.savefig(temp_img_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            image = Image.open(temp_img_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(tensor)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

        # Display Results
        predicted_idx = np.argmax(probabilities)
        predicted_emotion = EMOTION_MAP[predicted_idx]
        confidence = probabilities[predicted_idx] * 100

        st.markdown("---")
        st.subheader("Prediction Results")
        
        # Highlight Main Prediction
        st.success(f"### Predicted Emotion: **{predicted_emotion.upper()}** ({confidence:.2f}% Confidence)")

        # Probability Breakdown Chart
        prob_df = pd.DataFrame({
            "Emotion": [EMOTION_MAP[i] for i in range(8)],
            "Confidence (%)": probabilities * 100
        }).set_index("Emotion")

        st.bar_chart(prob_df)
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
import yaml

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
    """Dynamically loads the trained ResNet backbone model and weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "artifacts/training/model.pth"
    
    if not os.path.exists(model_path):
        st.error(f"Model checkpoint not found at {model_path}. Please run training first!")
        return None, device

    # Read configuration from params.yaml to match backbone and classes
    try:
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
        base_model = params.get("BASE_MODEL", "resnet34").lower()
        num_classes = params.get("CLASSES", 8)
    except Exception:
        base_model = "resnet34"
        num_classes = 8

    # Instantiate matching backbone
    if base_model == "resnet50":
        model = models.resnet50(weights=None)
    elif base_model == "resnet34":
        model = models.resnet34(weights=None)
    else:
        model = models.resnet18(weights=None)

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
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
This application uses a fine-tuned **ResNet34** deep learning model trained on Mel-spectrograms to detect emotions from speech audio files. 
Built as part of an end-to-end production-grade **MLOps pipeline** (PyTorch, MLflow, DVC).
""")

st.markdown("---")

# Sidebar for Quick Samples
st.sidebar.header("⚡ Quick Test Samples")
st.sidebar.markdown("If you don't have an audio sample, try one of these pre-loaded samples:")

sample_files = {
    "Select a sample...": None,
    "Calm Speech Sample": "samples/calm.wav",
    "Suprised Speech Sample": "samples/suprised.wav",
    "Happy Speech Sample": "samples/happy.wav"
}

selected_sample_name = st.sidebar.selectbox("Choose a sample audio:", list(sample_files.keys()))

# Main Input Method: File Uploader or Selected Sample
active_audio_path = None
temp_audio_path = "temp_active.wav"

if selected_sample_name != "Select a sample...":
    sample_path = sample_files[selected_sample_name]
    if os.path.exists(sample_path):
        active_audio_path = sample_path
    else:
        st.sidebar.error(f"Sample file not found at {sample_path}. Please place .wav files in the 'samples/' folder.")

# Fallback to file uploader if no sample is chosen
uploaded_file = st.file_uploader("Or upload your own speech audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    active_audio_path = temp_audio_path

# Process and Display if audio source is available
if active_audio_path is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Audio Playback")
        st.audio(active_audio_path, format="audio/wav")

    with col2:
        st.subheader("Mel-Spectrogram")
        y, sr = librosa.load(active_audio_path, sr=None)
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.axis('off')
        librosa.display.specshow(mel_spect_db, sr=sr, fmax=8000, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    if model is not None:
        with st.spinner("Analyzing emotion..."):
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

        predicted_idx = np.argmax(probabilities)
        predicted_emotion = EMOTION_MAP[predicted_idx]
        confidence = probabilities[predicted_idx] * 100

        st.markdown("---")
        st.subheader("Prediction Results")
        
        st.success(f"### Predicted Emotion: **{predicted_emotion.upper()}** ({confidence:.2f}% Confidence)")

        prob_df = pd.DataFrame({
            "Emotion": [EMOTION_MAP[i] for i in range(8)],
            "Confidence (%)": probabilities * 100
        }).set_index("Emotion")

        st.bar_chart(prob_df)
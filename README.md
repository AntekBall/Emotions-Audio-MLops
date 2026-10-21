# 🎙️ Speech Emotion Recognition (SER) MLOps Pipeline

An end-to-end MLOps pipeline for classifying human emotions from speech audio using deep learning, orchestrated with **DVC**, tracked via **MLflow**, and deployed live as a cloud web application.

---

## 🚀 Live Demo

Test the live application here: **[🔗 Speech Emotion Recognizer App](https://emotions-audio-mlops-kkwgrpeqjpzzfvpmsrjqqx.streamlit.app/)**

---

## 🧠 Project Architecture & Tech Stack

- **Deep Learning Model:** PyTorch with a fine-tuned **ResNet34** backbone trained on the RAVDESS dataset.
- **Pipeline Orchestration:** **DVC (Data Version Control)** to manage data stages, feature extraction, and model training reproducibly.
- **Experiment Tracking:** **MLflow** and **Optuna** for logging model parameters, metrics, and tracking training runs.
- **Audio Processing:** `librosa`, `soundfile`, and system-level `libsndfile1` bindings for robust Mel-spectrogram extraction.
- **Containerization & Deployment:** Fully containerized via **Docker** and hosted live on **Streamlit Community Cloud** with automated GitHub Release checkpoint retrieval.

---

## 🐳 Local Deployment via Docker

To run the fully containerized application locally on your machine, ensure you have [Docker](https://www.docker.com/) installed, then follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AntekBall/Emotions-Audio-MLops
   cd Emotions-Audio-MLops
   ```
2. **Build the Docker image:**

  ```bash
  docker build -t speech-emotion-recognizer .
  ```
3. **Run the container:**
  ```bash
  docker run -p 7860:7860 speech-emotion-recognizer
  ```
4. **Access the app** <br>
Open your browser and navigate to: http://localhost:7860

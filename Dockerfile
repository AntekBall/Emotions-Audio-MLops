FROM python:3.10-slim

# Install system dependencies including git and libsndfile1
RUN apt-get update && apt-get install -y \
    git \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency list and setup files
COPY requirements.txt .
COPY setup.py .
COPY src/ ./src/

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts explicitly
COPY app.py .
COPY samples/ ./samples/
#COPY artifacts/training/model.pth ./artifacts/training/model.pth

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
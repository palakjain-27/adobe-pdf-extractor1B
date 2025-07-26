FROM --platform=linux/amd64 python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for PyMuPDF, spaCy, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better layer caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Download spaCy model (EN)
RUN python -m spacy download en_core_web_sm

# Warm up Hugging Face pipeline (optional, but slows build)
# You can comment this out and do lazy loading in code if build takes too long
# RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')"

# Copy application code
COPY app/ /app/app/

# Make sure these exist before runtime
RUN mkdir -p /app/input /app/output

# Set entrypoint
CMD ["python", "/app/app/main.py"]

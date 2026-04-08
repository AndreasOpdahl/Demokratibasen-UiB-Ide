FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (GPU-enabled PyTorch from official CUDA wheel index)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" torch torchvision torchaudio && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make scripts executable
RUN chmod +x *.sh *.py

# Expose port
EXPOSE 8000

# Default command (can be overridden)
CMD ["python", "app.py", "--help"]

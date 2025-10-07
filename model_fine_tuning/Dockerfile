FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv $VIRTUAL_ENV

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Create working directory
WORKDIR /app

# Copy scripts
COPY scripts/ ./scripts/

# Create data directories
RUN mkdir -p /app/data/input /app/data/output /app/models

# Set entrypoint
ENTRYPOINT ["/bin/bash"]

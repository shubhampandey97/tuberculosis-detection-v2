FROM nvidia/cuda:12.6.3-base-ubuntu22.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install TensorFlow GPU (stable version, NOT nightly)
RUN pip3 install "numpy<2"
RUN pip3 install tensorflow==2.16.1

# Copy requirements
COPY requirements.txt .

# Install remaining dependencies
RUN pip3 install -r requirements.txt

# Copy project
COPY . .

# Disable problematic optimizations
ENV TF_ENABLE_ONEDNN_OPTS=0

CMD ["python3", "-m", "src.pipeline.train_pipeline"]
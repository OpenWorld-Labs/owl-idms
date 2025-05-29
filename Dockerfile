# Use CUDA 12.8 runtime as base image for lightweight deployment
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    python3-pip \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3.10-venv \
    git \
    && rm -rf /var/lib/apt/lists/*


# Set Python 3.10 as default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/root/.cargo/bin sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Create working directory
WORKDIR /app

# Install PyTorch with CUDA 12.8 support
RUN uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install other requirements
RUN uv pip install --system \
    einops \
    ema-pytorch \
    numpy \
    opencv-python \
    imageio \
    wandb \
    tqdm \
    omegaconf


COPY . /app

# Set the default command
CMD ["python3"]
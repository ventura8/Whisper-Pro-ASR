# Start with OpenVINO runtime which has verified Intel NPU/GPU drivers
FROM openvino/ubuntu24_runtime:2025.4.1

# Switch to root for installations
USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Install system tools
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update && \
  apt-get install -y --no-install-recommends \
  wget \
  gnupg \
  git \
  xz-utils \
  build-essential \
  patchelf \
  python3-dev \
  python3-pip \
  python3-venv \
  software-properties-common \
  && rm -rf /var/lib/apt/lists/* \
  && wget https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.0-latest-linux64-gpl-8.0.tar.xz \
  && tar -xvf ffmpeg-n8.0-latest-linux64-gpl-8.0.tar.xz \
  && mv ffmpeg-n8.0-latest-linux64-gpl-8.0/bin/ffmpeg /usr/local/bin/ \
  && mv ffmpeg-n8.0-latest-linux64-gpl-8.0/bin/ffprobe /usr/local/bin/ \
  && rm -rf ffmpeg-n8.0-latest-linux64-gpl-8.0* \
  && ffmpeg -version

# Install NVIDIA CUDA 12.8 explicitly (Ubuntu 24.04)
# This adds CUDA support to the Intel-optimized base image
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
  dpkg -i cuda-keyring_1.1-1_all.deb && \
  apt-get update && \
  apt-get install -y --no-install-recommends \
  cuda-libraries-12-8 \
  cuda-cudart-12-8 \
  libcudnn9-cuda-12 \
  cuda-nvcc-12-8 \
  && rm -rf /var/lib/apt/lists/*


# Copy pip configuration
COPY pip.conf /etc/pip.conf

# Upgrade pip (safe in this environment)
RUN python3 -m pip install --upgrade pip

# Install Python requirements
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
  python3 -m pip install -r requirements.txt "audio-separator~=0.41.1" --retries 3 && \
  # Segregated Install: NVIDIA CUDA Support
  mkdir -p /app/libs/nvidia && \
  python3 -m pip install "onnxruntime-gpu~=1.21.0" --target /app/libs/nvidia --no-dependencies && \
  # Segregated Install: Intel OpenVINO Support
  mkdir -p /app/libs/intel && \
  python3 -m pip install "onnxruntime-openvino~=1.21.0" --target /app/libs/intel --no-dependencies

# Fix CTranslate2 executable stack issues
RUN find /usr/local/lib/python3.*/ -name "*.so*" -exec patchelf --clear-execstack {} \;

# Set Hugging Face cache location to a persistent volume for build-time caching
ENV HF_HOME=/root/.cache/huggingface

# Preload AI Models into image (Bake them into the layer)
# We use cache mounts for both pip and the model directories to speed up rebuilds
COPY scripts/preload_model.py ./scripts/
RUN --mount=type=cache,target=/root/.cache \
  python3 scripts/preload_model.py --skip-intel-whisper



# Copy Application Code (Frequent changes here won't trigger model redownload)
WORKDIR /app
COPY modules/ ./modules/
COPY scripts/ ./scripts/
COPY whisper_server.py .

CMD ["python3", "whisper_server.py"]

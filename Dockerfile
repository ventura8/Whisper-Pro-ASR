# Stage 0: Swagger UI Assets
FROM swaggerapi/swagger-ui:v5.32.6 AS swagger-ui-source

# Start with OpenVINO runtime which has verified Intel NPU/GPU drivers
FROM openvino/ubuntu24_runtime:2026.2.1

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
  intel-opencl-icd \
  intel-level-zero-gpu \
  && wget https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz \
  && tar -xvf ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz \
  && mv ffmpeg-n8.1-latest-linux64-gpl-8.1/bin/ffmpeg /usr/local/bin/ \
  && mv ffmpeg-n8.1-latest-linux64-gpl-8.1/bin/ffprobe /usr/local/bin/ \
  && rm -rf ffmpeg-n8.1-latest-linux64-gpl-8.1* \
  && ffmpeg -version

# Install NVIDIA CUDA 12.8 explicitly (Ubuntu 24.04)
# This adds CUDA support to the Intel-optimized base image
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
  dpkg -i cuda-keyring_1.1-1_all.deb && \
  apt-get update && \
  apt-get install -y --no-install-recommends \
  cuda-libraries-12-8 \
  cuda-cudart-12-8 \
  libcudnn9-cuda-12 \
  cuda-nvcc-12-8 && \
  rm -f cuda-keyring_1.1-1_all.deb


# Copy pip configuration
WORKDIR /app
COPY pip.conf /etc/pip.conf
COPY pyproject.toml poetry.lock* ./

# Upgrade pip (safe in this environment)
RUN --mount=type=cache,target=/root/.cache \
  python3 -m pip install --upgrade pip

# Install Python dependencies via Poetry (no requirements.txt)
ENV POETRY_VIRTUALENVS_CREATE=false
RUN --mount=type=cache,target=/root/.cache \
  pip install poetry && \
  poetry install --without dev && \
  # Segregated Install: NVIDIA CUDA Support\
  mkdir -p /app/libs/nvidia && \
  python3 -m pip install "onnxruntime-gpu~=1.25.0" --target /app/libs/nvidia --no-dependencies && \
  # Segregated Install: Intel OpenVINO Support\
  mkdir -p /app/libs/intel && \
  python3 -m pip install "onnxruntime-openvino~=1.24.0" --target /app/libs/intel --no-dependencies

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
COPY static/ ./static/
# Copy offline Swagger assets from the swagger-ui image
COPY --from=swagger-ui-source /usr/share/nginx/html/swagger-ui.css ./static/swagger-ui.css
COPY --from=swagger-ui-source /usr/share/nginx/html/swagger-ui-bundle.js ./static/swagger-ui-bundle.js
COPY --from=swagger-ui-source /usr/share/nginx/html/favicon-32x32.png ./static/favicon.png
COPY whisper_pro_asr.py .

# Create persistent storage directory
RUN mkdir -p /app/data && chmod 777 /app/data

# Create default temp processing directory
# Mount as tmpfs in docker-compose for zero SSD writes
RUN mkdir -p /tmp/whisper && chmod 777 /tmp/whisper
ENV WHISPER_TEMP_DIR=/tmp/whisper
ENV WHISPER_PERSISTENT_DIR=/app/data

EXPOSE 9000
CMD ["python3", "whisper_pro_asr.py"]

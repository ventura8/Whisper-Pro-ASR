# Stage 0: Swagger UI Assets
FROM swaggerapi/swagger-ui:v5.32.6 AS swagger-ui-source

# Start with OpenVINO runtime which has verified Intel NPU/GPU drivers
FROM openvino/ubuntu24_runtime:2026.2.1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Switch to root for installations
USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

ARG POETRY_VERSION=2.4.1
ARG PIP_VERSION=26.1.2
ARG FFMPEG_VERSION=8.1.2
ARG FFMPEG_TARBALL=ffmpeg-${FFMPEG_VERSION}.tar.xz
ARG FFMPEG_URL=https://ffmpeg.org/releases/${FFMPEG_TARBALL}
ARG FFMPEG_SIG_URL=${FFMPEG_URL}.asc
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV INTEL_OPENVINO_DIR=/opt/intel/openvino
ENV LD_LIBRARY_PATH=/usr/lib/wsl/lib:/opt/rocm/lib:/opt/intel/openvino/runtime/lib/intel64:/opt/intel/openvino/runtime/3rdparty/tbb/lib:/opt/intel/openvino/runtime/3rdparty/omp/lib:${LD_LIBRARY_PATH}
ENV HSA_ENABLE_DXG_DETECTION=1

# Install system tools
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update && \
  apt-get install -y --no-install-recommends \
  wget=* \
  gnupg=* \
  git=* \
  xz-utils=* \
  build-essential=* \
  nasm=* \
  pkg-config=* \
  patchelf=* \
  python3-dev=* \
  python3-pip=* \
  python3-venv=* \
  software-properties-common=* \
  intel-opencl-icd=* \
  intel-level-zero-gpu=* \
  libhipblas0=* \
  librocblas0=* \
  && wget --progress=dot:giga -O /tmp/ffmpeg.tar.xz "${FFMPEG_URL}" \
  && wget --progress=dot:giga -O /tmp/ffmpeg.tar.xz.asc "${FFMPEG_SIG_URL}" \
  && wget --progress=dot:giga -O /tmp/ffmpeg-devel.asc https://ffmpeg.org/ffmpeg-devel.asc \
  && gpg --batch --import /tmp/ffmpeg-devel.asc \
  && gpg --batch --verify /tmp/ffmpeg.tar.xz.asc /tmp/ffmpeg.tar.xz \
  && tar -xf /tmp/ffmpeg.tar.xz -C /tmp

WORKDIR /tmp/ffmpeg-${FFMPEG_VERSION}

RUN ./configure --prefix=/usr/local --disable-debug --disable-doc --disable-static --enable-shared --enable-pic \
  && make -j"$(nproc)" \
  && make install \
  && ldconfig \
  && ffmpeg -version \
  && ffprobe -version \
  && rm -rf /tmp/ffmpeg* /root/.gnupg

# Install NVIDIA CUDA 13.2 explicitly (Ubuntu 24.04)
# This adds CUDA support to the Intel-optimized base image
WORKDIR /

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  wget --progress=dot:giga https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
  dpkg -i cuda-keyring_1.1-1_all.deb && \
  apt-get update && \
  apt-get install -y --no-install-recommends \
  cuda-libraries-13-2=* \
  cuda-cudart-13-2=* \
  libcudnn9-cuda-13=* \
  cuda-nvcc-13-2=* && \
  rm -f cuda-keyring_1.1-1_all.deb

# Install AMD ROCm 6.2 libraries explicitly (Ubuntu 24.04)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/trusted.gpg.d/rocm.gpg && \
  echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.2 ubuntu main" > /etc/apt/sources.list.d/rocm.list && \
  apt-get update && \
  apt-get install -y --no-install-recommends \
  miopen-hip=* \
  hip-runtime-amd=* \
  hipfft=* \
  rocm-smi-lib=* \
  migraphx=* \
  libhipblas0=* \
  librocblas0=* && \
  [ -f /usr/lib/x86_64-linux-gnu/libhipblas.so.0 ] && ln -sf /usr/lib/x86_64-linux-gnu/libhipblas.so.0 /usr/lib/x86_64-linux-gnu/libhipblas.so.3 && \
  [ -f /usr/lib/x86_64-linux-gnu/librocblas.so.0 ] && ln -sf /usr/lib/x86_64-linux-gnu/librocblas.so.0 /usr/lib/x86_64-linux-gnu/librocblas.so.3 && \
  [ -f /opt/rocm/lib/libamdhip64.so.6 ] && ln -sf /opt/rocm/lib/libamdhip64.so.6 /opt/rocm/lib/libamdhip64.so.7 && \
  [ -f /opt/rocm/lib/librocm_smi64.so.7 ] && ln -sf /opt/rocm/lib/librocm_smi64.so.7 /opt/rocm/lib/librocm_smi64.so.1 && \
  ldconfig

# Install librocdxg — AMD ROCDXG user-mode library for ROCm on WSL2 via /dev/dxg.
# This replaces /dev/kfd (native Linux ROCm driver) with a DXG translation layer
# allowing onnxruntime-rocm to use AMD GPU on Windows/WSL2 hosts.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  wget -q -O /tmp/rocdxg-roct.deb \
    "https://github.com/ROCm/librocdxg/releases/download/v1.2.1/rocdxg-roct_1.2.1_amd64.deb" && \
  echo "7889eef45a1132ed2dde88d8ea1356bf791ec9c05802a18940bc81b970e850e0  /tmp/rocdxg-roct.deb" | sha256sum -c - && \
  dpkg -i /tmp/rocdxg-roct.deb && \
  rm -f /tmp/rocdxg-roct.deb && \
  ldconfig


# Copy pip configuration
WORKDIR /app
COPY pip.conf /etc/pip.conf
COPY pyproject.toml poetry.lock* ./

# Upgrade pip (safe in this environment)
RUN --mount=type=cache,target=/root/.cache \
  python3 -m pip install --no-cache-dir "pip==${PIP_VERSION}"

# Install Python dependencies via Poetry (no requirements.txt)
ENV POETRY_VIRTUALENVS_CREATE=false
RUN --mount=type=cache,target=/root/.cache \
  python3 -m pip install --no-cache-dir "poetry==${POETRY_VERSION}" && \
  poetry install --without dev --with ml && \
  # Remove ambiguous global ONNX Runtime installs pulled transitively.
  (python3 -m pip uninstall -y onnxruntime onnxruntime-openvino onnxruntime-gpu || true) && \
  # Segregated Install: CPU baseline runtime (deterministic default)
  mkdir -p /app/libs/cpu && \
  python3 -m pip install --no-cache-dir "onnxruntime~=1.27.0" --target /app/libs/cpu --no-dependencies && \
  # Segregated Install: NVIDIA CUDA Support\
  mkdir -p /app/libs/nvidia && \
  python3 -m pip install --no-cache-dir "onnxruntime-gpu~=1.25.0" --target /app/libs/nvidia --no-dependencies && \
  # Segregated Install: Intel OpenVINO Support\
  mkdir -p /app/libs/intel && \
  python3 -m pip install --no-cache-dir "onnxruntime-openvino~=1.24.0" --target /app/libs/intel --no-dependencies && \
  # Segregated Install: AMD ROCm Support\
  mkdir -p /app/libs/amd && \
  python3 -m pip install --no-cache-dir "onnxruntime-rocm==1.22.2.post3" --target /app/libs/amd --no-dependencies

# Fix CTranslate2 executable stack issues
RUN find /usr/local/lib/python3.*/ -name "*.so*" -exec patchelf --clear-execstack {} \;

# Set Hugging Face cache location under /app for non-root runtime ownership.
ENV HF_HOME=/app/.cache/huggingface

# Preload AI Models into image (Bake them into the layer)
# We use cache mounts for both pip and the model directories to speed up rebuilds
COPY scripts/preload_model.py ./scripts/
RUN --mount=type=cache,target=/root/.cache \
  PYTHONPATH=/app/libs/cpu python3 scripts/preload_model.py --skip-intel-whisper



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
RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache

# Create default temp processing directory
# Mount as tmpfs in docker-compose for zero SSD writes
RUN mkdir -p /tmp/whisper && chmod 777 /tmp/whisper
ENV WHISPER_TEMP_DIR=/tmp/whisper
ENV WHISPER_PERSISTENT_DIR=/app/data
ENV NUMBA_CACHE_DIR=/tmp/numba-cache
RUN mkdir -p /tmp/numba-cache && chmod 777 /tmp/numba-cache

USER nobody

EXPOSE 9000
CMD ["python3", "whisper_pro_asr.py"]

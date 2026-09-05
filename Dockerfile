# Declared globally because the shared application base consumes it in a FROM
# instruction. Individual targets may override it at build time.
ARG RUNTIME_BASE=openvino/ubuntu24_runtime:2026.3.1@sha256:fdfa0f1c42f69b4cbf010999482a32279ac04a07b2fb53f449e67246dcb5f1df

# Stage 0: Swagger UI Assets
FROM swaggerapi/swagger-ui:v5.32.14@sha256:3d93169968848d371a6a56ca1ab18b47a8906ba461b8eba0688866354f5431d5 AS swagger-ui-source

# Stage 1: Build FFmpeg 9.0.1 from source
FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517 AS ffmpeg-builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

ARG FFMPEG_VERSION=9.0.1
ARG FFMPEG_TARBALL=ffmpeg-${FFMPEG_VERSION}.tar.xz
ARG FFMPEG_URL=https://ffmpeg.org/releases/${FFMPEG_TARBALL}
ARG FFMPEG_SIG_URL=${FFMPEG_URL}.asc

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update && \
  apt-get install -y --no-install-recommends \
  wget=* \
  gnupg=* \
  xz-utils=* \
  build-essential=* \
  nasm=* \
  pkg-config=* \
  ca-certificates=* \
  && wget --progress=dot:giga -O /tmp/ffmpeg.tar.xz "${FFMPEG_URL}" \
  && wget --progress=dot:giga -O /tmp/ffmpeg.tar.xz.asc "${FFMPEG_SIG_URL}" \
  && wget --progress=dot:giga -O /tmp/ffmpeg-devel.asc https://ffmpeg.org/ffmpeg-devel.asc \
  && gpg --batch --import /tmp/ffmpeg-devel.asc \
  && gpg --batch --status-fd 1 --verify /tmp/ffmpeg.tar.xz.asc /tmp/ffmpeg.tar.xz > /tmp/ffmpeg-gpg-status.txt \
  && grep -q "VALIDSIG FCF986EA15E6E293A5644F10B4322F04D67658D8" /tmp/ffmpeg-gpg-status.txt \
  && tar -xf /tmp/ffmpeg.tar.xz -C /tmp

WORKDIR /tmp/ffmpeg-${FFMPEG_VERSION}

RUN ./configure --prefix=/usr/local --disable-debug --disable-doc --disable-static --enable-shared --enable-pic \
  && make -j"$(nproc)" \
  && make install \
  && ldconfig \
  && ffmpeg -version \
  && ffprobe -version \
  && rm -rf /tmp/ffmpeg* /root/.gnupg

# Stage 2: Shared application base (no vendor accelerator stack).
#
# RUNTIME_BASE selects the base image per target, so there is one base definition rather
# than two parallel chains:
#   - Intel-bearing targets (intel, nvidia-intel, full) keep the OpenVINO runtime, which
#     supplies /opt/intel/openvino, the Level Zero + NPU drivers, Python and
#     VIRTUAL_ENV=/opt/venv.
#   - Non-Intel targets (cpu, nvidia, amd) build on plain ubuntu:24.04, dropping ~860MB of
#     Intel driver stack. Python and the venv are reconstructed explicitly below.
# Non-Intel targets must pass:
#   --build-arg RUNTIME_BASE=ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517
ARG RUNTIME_BASE
FROM ${RUNTIME_BASE} AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

ARG POETRY_VERSION=2.4.1
# Keep in sync with PIP_VERSION in scripts/ci/dependencies.env
ARG PIP_VERSION=26.2.1
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_DEFAULT_TIMEOUT=300
ENV INTEL_OPENVINO_DIR=/opt/intel/openvino

# /usr/lib/x86_64-linux-gnu is baked in FIRST so the system multi-vendor OpenCL ICD
# loader wins over any vendor package's own libOpenCL.so.1. This used to live in
# docker-compose.yml, which had to restate the whole value and silently drifted out of
# sync; keeping it here means compose needs no LD_LIBRARY_PATH override at all.
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib/wsl/lib:/opt/intel/openvino/runtime/lib/intel64:/opt/intel/openvino/runtime/3rdparty/tbb/lib:/opt/intel/openvino/runtime/3rdparty/omp/lib

# Copy compiled FFmpeg from builder stage
COPY --from=ffmpeg-builder /usr/local/bin/ff* /usr/local/bin/
COPY --from=ffmpeg-builder /usr/local/lib/libav* /usr/local/lib/
COPY --from=ffmpeg-builder /usr/local/lib/libsw* /usr/local/lib/
COPY --from=ffmpeg-builder /usr/local/include/libav* /usr/local/include/
COPY --from=ffmpeg-builder /usr/local/include/libsw* /usr/local/include/
RUN ldconfig && ffmpeg -version && ffprobe -version

# Shared runtime tools. build-essential/python3-dev are needed to build wheels and are
# purged again in the same layer as the Poetry install below.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update && \
  apt-get install -y --no-install-recommends \
  wget=* \
  gnupg=* \
  patchelf=* \
  ca-certificates=* \
  python3=* \
  python3-dev=* \
  python3-pip=* \
  python3-venv=* \
  software-properties-common=* \
  build-essential=*

# The OpenVINO base ships VIRTUAL_ENV=/opt/venv and puts it on PATH; plain ubuntu does
# not. Create it when absent so both bases land dependencies in the same place.
RUN [ -d /opt/venv ] || python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:${PATH}

WORKDIR /app
COPY pip.conf /etc/pip.conf
COPY pyproject.toml poetry.lock* ./
COPY scripts/docker/ /usr/local/build/

RUN --mount=type=cache,target=/root/.cache \
  python3 -m pip install --no-cache-dir --upgrade "pip==${PIP_VERSION}"

ENV POETRY_HTTP_TIMEOUT=1800
ENV PIP_DEFAULT_TIMEOUT=1800
ENV POETRY_VIRTUALENVS_CREATE=false

ENV HF_HOME=/app/model_cache/huggingface

# Application code
COPY modules/ ./modules/
COPY scripts/ ./scripts/
COPY static/ ./static/
COPY --from=swagger-ui-source /usr/share/nginx/html/swagger-ui.css ./static/swagger-ui.css
COPY --from=swagger-ui-source /usr/share/nginx/html/swagger-ui-bundle.js ./static/swagger-ui-bundle.js
COPY --from=swagger-ui-source /usr/share/nginx/html/favicon-32x32.png ./static/favicon.png
COPY whisper_pro_asr.py .

# Models are NOT baked into the image. They are provisioned at startup into the
# persistent /app/model_cache volume by modules/core/model_provisioning.py; tasks
# submitted before the download finishes wait in the scheduler queue.

RUN mkdir -p /app/data && chmod 777 /app/data && \
  mkdir -p /app/.cache && chmod -R 777 /app/.cache && \
  mkdir -p /tmp/whisper && chmod 777 /tmp/whisper && \
  mkdir -p /tmp/numba-cache && chmod 777 /tmp/numba-cache

ENV WHISPER_TEMP_DIR=/tmp/whisper
ENV WHISPER_PERSISTENT_DIR=/app/data
ENV NUMBA_CACHE_DIR=/tmp/numba-cache

# Stage 3: Python dependencies.
#
# Two variants, because torch from PyPI bundles ~3.3GB of CUDA libraries
# (site-packages/nvidia + triton) that are dead weight without an NVIDIA GPU. The CPU
# wheel must be installed in the SAME layer as the rest of the dependency tree -- swapping
# it in a later layer would leave the CUDA files in the earlier layer and save nothing.
#
# torch itself cannot be dropped: audio-separator requires it and vocal separation is on
# by default.
USER nobody

FROM base AS deps-cuda
USER root
RUN --mount=type=cache,target=/root/.cache \
  PYTHONDONTWRITEBYTECODE=1 python3 -m pip install --no-cache-dir "poetry==${POETRY_VERSION}" && \
  poetry config installer.max-workers 4 && \
  PYTHONDONTWRITEBYTECODE=1 poetry install --without dev --with ml && \
  # Remove ambiguous global ONNX Runtime installs pulled transitively. Every target
  # gets its runtime from /app/libs/<vendor> instead.
  (python3 -m pip uninstall -y onnxruntime onnxruntime-openvino onnxruntime-gpu || true) && \
  # UVR's torch path loads the Torch CUDA dependency set dynamically, including NCCL,
  # cuSPARSELt, and NVSHMEM. Keep the set intact even for a single-GPU model.
  /usr/local/build/strip_build_artifacts.sh
USER nobody

FROM base AS deps-cpu
USER root
RUN --mount=type=cache,target=/root/.cache \
  PYTHONDONTWRITEBYTECODE=1 python3 -m pip install --no-cache-dir "poetry==${POETRY_VERSION}" && \
  poetry config installer.max-workers 4 && \
  PYTHONDONTWRITEBYTECODE=1 poetry install --without dev --with ml && \
  (python3 -m pip uninstall -y onnxruntime onnxruntime-openvino onnxruntime-gpu || true) && \
  # Swap the CUDA-bundling torch for the CPU build, in this same layer.
  torch_v="$(grep -oP '^torch = "\K[^"]+' /app/pyproject.toml)" && \
  audio_v="$(grep -oP '^torchaudio = "\K[^"]+' /app/pyproject.toml)" && \
  vision_v="$(grep -oP '^torchvision = "\K[^"]+' /app/pyproject.toml)" && \
  PYTHONDONTWRITEBYTECODE=1 python3 -m pip install --no-cache-dir --force-reinstall \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==${torch_v}" "torchaudio==${audio_v}" "torchvision==${vision_v}" && \
  rm -rf /opt/venv/lib/python3*/site-packages/nvidia \
    /opt/venv/lib/python3*/site-packages/triton && \
  /usr/local/build/strip_build_artifacts.sh

# Stage 4: ONNX Runtime variants, built once and copied per target.
# Keeping these in a discarded builder means no target carries another vendor's runtime.
USER nobody

# base, not deps-cuda: this stage only pip-installs four ONNX Runtime wheels into /out and
# every target then COPYs from it. Inheriting deps-cuda made CPU, Intel, Intel-XPU and AMD
# builds materialize the ~3.3GB CUDA dependency layer purely as a build-graph dependency of
# a stage whose output does not use it.
FROM base AS ort-builder
USER root
RUN --mount=type=cache,target=/root/.cache \
  mkdir -p /out/cpu /out/nvidia /out/intel /out/amd && \
  python3 -m pip install --no-cache-dir "onnxruntime~=1.27.0" --target /out/cpu --no-dependencies && \
  python3 -m pip install --no-cache-dir "onnxruntime-gpu~=1.29.0" --target /out/nvidia --no-dependencies && \
  python3 -m pip install --no-cache-dir "onnxruntime-openvino~=1.24.0" --target /out/intel --no-dependencies && \
  python3 -m pip install --no-cache-dir "onnxruntime-rocm==1.22.2.post3" --target /out/amd --no-dependencies && \
  find /out -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

# Stage 4: Vendor system layers.
USER nobody

FROM deps-cuda AS sys-cuda
USER root
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  /usr/local/build/install_cuda.sh && \
  /usr/local/build/prune_cuda.sh
# CTranslate2 4.8.x dlopen()s libcublas.so.12 while torch ships cu13, so both majors
# must be resolvable.
# Preserve the base image's system OpenCL ICD loader before CUDA's bundled loader.  The
# Intel OpenVINO GPU plugin needs that loader to enumerate /etc/OpenCL/vendors/intel.icd;
# CUDA libraries stay appended for CTranslate2's libcublas.so.12 dynamic load.
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-12.9/targets/x86_64-linux/lib:/usr/local/cuda-13.3/targets/x86_64-linux/lib
USER nobody

FROM deps-cpu AS sys-intel
USER root
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  /usr/local/build/install_intel.sh
USER nobody

FROM deps-cpu AS sys-amd
USER root
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  /usr/local/build/install_rocm.sh && \
  /usr/local/build/prune_rocm.sh && \
  /usr/local/build/strip_build_artifacts.sh
ENV PATH=/opt/rocm/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH}
ENV ROCBLAS_TENSILE_LIBPATH=/usr/lib/x86_64-linux-gnu/rocblas/current/library
ENV HSA_ENABLE_DXG_DETECTION=1
USER nobody

FROM sys-cuda AS sys-nvidia-intel
USER root
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  /usr/local/build/install_intel.sh
USER nobody

FROM sys-nvidia-intel AS sys-full
USER root
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  /usr/local/build/install_rocm.sh && \
  /usr/local/build/prune_rocm.sh && \
  /usr/local/build/strip_build_artifacts.sh
ENV PATH=/opt/rocm/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH}
ENV ROCBLAS_TENSILE_LIBPATH=/usr/lib/x86_64-linux-gnu/rocblas/current/library
ENV HSA_ENABLE_DXG_DETECTION=1
USER nobody

# Stage 5: Shippable targets. /app/libs/cpu is present in every one -- it is the
# fallback modules/core/bootstrap.py resolves to when a vendor runtime is absent.

# Stage 5b: WhisperX's segregated stack, built once and copied into every target that
# ships it. WhisperX pins an older transformers/huggingface-hub than the rest of the app,
# which is why it lives in its own directory that is prepended to sys.path in the worker.
FROM base AS whisperx-libs
USER root
COPY requirements/whisperx.lock.txt /tmp/whisperx.lock.txt
RUN --mount=type=cache,target=/root/.cache \
  mkdir -p /out/whisperx && \
  PYTHONDONTWRITEBYTECODE=1 python3 -m pip install --no-cache-dir --require-hashes --no-deps \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --target /out/whisperx \
  -r /tmp/whisperx.lock.txt && \
  rm -f /tmp/whisperx.lock.txt && \
  # Reuse the image's torch stack instead of WhisperX's bundled CPU copy. The lock pins
  # torch 2.8.0+cpu, which is both ~1.2GB of duplication and the reason WhisperX could
  # only ever run on the CPU ("Torch not compiled with CUDA enabled"). The directory is
  # prepended to sys.path, so deleting these lets the import fall through to the image's
  # CUDA build while WhisperX keeps its own pinned transformers and huggingface-hub --
  # the actual reason it is segregated.
  # Only the exact duplicates go: torch_audiomentations, torch_pitch_shift, torchcodec
  # and torchmetrics are WhisperX's own and a `torch*` glob would take them too.
  rm -rf /out/whisperx/torch /out/whisperx/torch-*.dist-info \
  /out/whisperx/torchaudio /out/whisperx/torchaudio-*.dist-info \
  /out/whisperx/torchvision /out/whisperx/torchvision-*.dist-info \
  /out/whisperx/triton /out/whisperx/triton-*.dist-info \
  /out/whisperx/functorch /out/whisperx/torchgen /out/whisperx/nvidia
USER nobody

FROM deps-cpu AS cpu
ENV WHISPER_IMAGE_EDITION=cpu
COPY --from=ort-builder /out/cpu /app/libs/cpu
USER root
RUN /usr/local/build/strip_build_artifacts.sh
USER nobody
EXPOSE 9000
CMD ["python3", "whisper_pro_asr.py"]

FROM sys-intel AS intel
ENV WHISPER_IMAGE_EDITION=intel
COPY --from=ort-builder /out/cpu /app/libs/cpu
COPY --from=ort-builder /out/intel /app/libs/intel
USER root
RUN /usr/local/build/strip_build_artifacts.sh
USER nobody
EXPOSE 9000
CMD ["python3", "whisper_pro_asr.py"]

FROM sys-cuda AS nvidia
ENV WHISPER_IMAGE_EDITION=nvidia
COPY --from=ort-builder /out/cpu /app/libs/cpu
COPY --from=ort-builder /out/nvidia /app/libs/nvidia
USER root
RUN /usr/local/build/strip_build_artifacts.sh
USER nobody
EXPOSE 9000
CMD ["python3", "whisper_pro_asr.py"]

FROM sys-amd AS amd
ENV WHISPER_IMAGE_EDITION=amd
COPY --from=ort-builder /out/cpu /app/libs/cpu
COPY --from=ort-builder /out/amd /app/libs/amd
USER root
RUN /usr/local/build/verify_ort_provider_links.sh
RUN /usr/local/build/strip_build_artifacts.sh
USER nobody
EXPOSE 9000
CMD ["python3", "whisper_pro_asr.py"]

FROM sys-nvidia-intel AS nvidia-intel
ENV WHISPER_IMAGE_EDITION=nvidia-intel
COPY --from=ort-builder /out/cpu /app/libs/cpu
COPY --from=ort-builder /out/nvidia /app/libs/nvidia
COPY --from=ort-builder /out/intel /app/libs/intel
USER root
RUN /usr/local/build/strip_build_artifacts.sh
USER nobody
EXPOSE 9000
CMD ["python3", "whisper_pro_asr.py"]

# Stage 7: NVIDIA + WhisperX -- CUDA acceleration with speaker diarization. This is the
# only target that ships WhisperX; it runs on CUDA here because it reuses this image's
# torch rather than the CPU build its lock pins.
FROM sys-cuda AS nvidia-whisperx
ENV WHISPER_IMAGE_EDITION=nvidia-whisperx
COPY --from=ort-builder /out/cpu /app/libs/cpu
COPY --from=ort-builder /out/nvidia /app/libs/nvidia
COPY --from=whisperx-libs /out/whisperx /app/libs/whisperx
ENV WHISPERX_LIB_PATH=/app/libs/whisperx
USER root
RUN /usr/local/build/strip_build_artifacts.sh
# Same executable-stack audit the full target runs; WhisperX ships the libraries it covers.
RUN /usr/local/build/clear_execstack.sh
USER nobody
EXPOSE 9000
CMD ["python3", "whisper_pro_asr.py"]

# Stage 8: AMD with ROCm PyTorch -- opt-in, because the ROCm torch wheel adds ~7.7GB even
# after pruning (14.1GB -> ~21.8GB) and only accelerates the torch-based engines.
# CTranslate2 has no ROCm backend and UVR already uses onnxruntime-rocm, so the plain
# `amd` target stays lean for everyone who does not need GPU openai-whisper.
FROM amd AS amd-rocm-torch
ENV WHISPER_IMAGE_EDITION=amd-rocm-torch
USER root
RUN --mount=type=cache,target=/root/.cache /usr/local/build/install_rocm_torch.sh
USER nobody
EXPOSE 9000
CMD ["python3", "whisper_pro_asr.py"]

# Stage 9: Intel with XPU PyTorch -- opt-in, because the XPU torch build plus Intel's
# oneAPI/SYCL runtime adds ~6GB (5.2GB -> ~11.2GB) and only accelerates the torch-based
# engines. INTEL-WHISPER (OpenVINO) and UVR (onnxruntime-openvino) do not use torch, so
# the plain `intel` target stays lean for the engines most Intel deployments actually run.
# Only viable on an Intel-only image: one interpreter can hold one torch build, and the
# CUDA-bearing targets need theirs.
FROM intel AS intel-xpu
ENV WHISPER_IMAGE_EDITION=intel-xpu
USER root
RUN --mount=type=cache,target=/root/.cache /usr/local/build/install_xpu_torch.sh
USER nobody
EXPOSE 9000
CMD ["python3", "whisper_pro_asr.py"]

# Stage 10: Every vendor in one image, plus WhisperX. Kept for anyone who wants a single
# image, but NOT part of the supported set: it is by far the largest target and every
# capability it offers is available from a smaller, purpose-built target above. Prefer those.
FROM sys-full AS full
ENV WHISPER_IMAGE_EDITION=full
COPY --from=ort-builder /out/cpu /app/libs/cpu
COPY --from=ort-builder /out/nvidia /app/libs/nvidia
COPY --from=ort-builder /out/intel /app/libs/intel
COPY --from=ort-builder /out/amd /app/libs/amd
COPY --from=whisperx-libs /out/whisperx /app/libs/whisperx
ENV WHISPERX_LIB_PATH=/app/libs/whisperx
USER root
RUN /usr/local/build/verify_ort_provider_links.sh
RUN /usr/local/build/strip_build_artifacts.sh
RUN /usr/local/build/clear_execstack.sh
USER nobody
EXPOSE 9000
CMD ["python3", "whisper_pro_asr.py"]

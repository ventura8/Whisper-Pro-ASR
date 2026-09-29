# v1.4.3 evidence: the last SonarQube fixes on real silicon

v1.4.3 changes no decoding behaviour, but three of its edits run on every request: the
request-context proxy (`modules/core/utils.py`, now a table lookup for its special
attributes), the task-start log line, and the language resolution behind it
(`modules/api/routes/asr.py` -> `languages.supported_code`). The CUDA installer download
also changed (`wget --https-only`), which rebuilt the CUDA layer. Each was driven with real
speech on the hardware it concerns; the whole suite mocks the ASR engine.

Both rows below ran the release tree itself -- the banner reads `Whisper Pro ASR 1.4.3`.
Accelerator evidence is the per-request device line and `/status` `measured=True`, never
the startup banner.

| host | configuration | device evidence | accuracy suite |
| --- | --- | --- | --- |
| Intel NUC (Core Ultra 255H) | `intel`, INTEL-WHISPER on the iGPU, UVR on the **NPU**, separation on | `Isolation complete on Intel(R) AI Boost [ONNX: OpenVINOExecutionProvider]` @ ~5.0x; `ASR=Intel GPU measured=True` | **9 passed** |
| RTX 3080 laptop + Intel iGPU | `nvidia-intel` (CUDA layer rebuilt through the new installer), FASTER-WHISPER, separation on | all 6 transcriptions `on hardware unit cuda:0`; UVR on `CUDAExecutionProvider` @ ~8.5x; `ASR=CUDA measured=True`; GPU 100% / 6.6 GB during the suite | **9 passed** |

The laptop's service log also shows the new task-start line as intended
(`Task: TRANSCRIPTION | Format: JSON | Lang: auto-detect`) and no tracebacks.

## Not validated on silicon

- **RTX 5090 (Blackwell)** -- its WSL distro runs its own Docker engine without buildx or
  the NVIDIA runtime, so no build could start. Nothing in this release is sm_120-specific.
- **AMD ROCm** -- `install_rocm.sh` changed only to `wget --https-only`; the `amd` and
  `amd-rocm-torch` images are built by CI's target validation jobs, not run on an AMD
  accelerator.
- **PowerShell bootstrappers** -- the shared helper was exercised against a real host and
  an unreachable one from PowerShell 7 on Linux, not from a Windows operator machine.

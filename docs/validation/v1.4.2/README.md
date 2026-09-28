# v1.4.2 evidence: a static-analysis cleanup on real silicon

v1.4.2 is a code-quality release: it resolves the SonarQube Cloud findings across the
Python modules, scripts, dashboard and tests. Nothing is meant to change behaviour, but
several edits sit on accelerator paths -- the INTEL-WHISPER chunk loop was split into a
helper and its silent-chunk test rewritten, the audio-separator ONNX patch was restructured,
the WhisperX worker's dispatch handler was renamed, and `process_exec` now applies its
deadline around the whole coroutine. The whole suite mocks the ASR engine, so each of those
was driven with real speech on the hardware it concerns.

Accelerator evidence is the per-request device line and `/status` `measured=True`, never the
startup banner.

| host | configuration | device evidence | accuracy suite |
| --- | --- | --- | --- |
| Intel NUC (Core Ultra 255H) | `intel`, INTEL-WHISPER on the iGPU, UVR on the **NPU**, separation on | `Isolation complete on Intel(R) AI Boost [ONNX: OpenVINOExecutionProvider]` @ ~4.7x; `ASR=Intel GPU measured=True` | **9 passed** (release head) |
| RTX 3080 laptop + Intel iGPU | `nvidia-intel`, FASTER-WHISPER, separation on | every transcription `on hardware unit cuda:0`; UVR on `CUDAExecutionProvider` @ ~8.5x; `ASR=CUDA measured=True`; GPU 100% / 6.5 GB during the suite | **9 passed** |
| RTX 3080 laptop | `nvidia-whisperx`, WHISPERX | every transcription `on hardware unit cuda:0`; `ASR=CUDA measured=True`; GPU 90-100% | **9 passed** |

The NUC run matches that host's production configuration, which is the one the
INTEL-WHISPER and NPU-preprocessing edits concern.

## Not validated on silicon

- **RTX 5090 (Blackwell)** -- not run this round; its WSL distro was not using Docker
  Desktop's engine, so no build could start. Nothing in this release is sm_120-specific.
- **AMD ROCm and Intel XPU images** -- the only edits to their build scripts are `[ ]` ->
  `[[ ]]` under `#!/bin/bash`; they were built by CI's target validation jobs, not run on
  an AMD or Arc-XPU accelerator.

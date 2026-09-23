# v1.4.1 evidence: the dependency sweep on real silicon

v1.4.1 changes no decoding behaviour, so the question this round had to answer was narrow:
does the bumped dependency set still drive each accelerator, and does it break anything that
worked before. Two failures showed up during validation and **both were proven pre-existing**
by re-running the identical configuration against pristine `f6699bb`. Recorded here because
"we saw it fail and decided it was fine" is worth exactly as much as the baseline behind it.

The bumps that could touch an accelerator: openvino / -genai / -tokenizers 2026.3.1 ->
2026.4.0 (Intel GPU and NPU plugins) and transformers 5.16.1 -> 5.17.0 with
huggingface-hub 1.30.0 -> 1.32.0 (torch engines). torch and torchvision did **not** move --
confirmed against `poetry.lock`, which shows no change to torch, torchvision, ctranslate2 or
any nvidia wheel.

## What each host proved

Accelerator evidence is the per-request device line, never the startup banner: on a
multi-unit host the banner names the pool, not what executed.

| host | device evidence | suite |
| --- | --- | --- |
| NUC, Intel **NPU** (UVR) | `Isolation complete on Intel(R) AI Boost [ONNX: OpenVINOExecutionProvider]` @ **3.97x** | 28 passed, 2 failed |
| NUC, Intel **GPU** (UVR) | `Isolation complete on Intel(R) Graphics [0x7d51] (iGPU) [ONNX: OpenVINOExecutionProvider]` @ **0.72x** | 28 passed, 2 failed |
| 5090, **CUDA** | `ASR Runtime: CUDA (float16)`, `Resource Pool: cuda:0`, UVR on `CUDAExecutionProvider` @ **13.1x**, `ASR=CUDA measured=True` | **30 passed, 0 skipped** |

The NPU running UVR at 3.97x against the same box's iGPU at 0.72x -- 5.5x faster on identical
6-second clips -- is itself the strongest evidence the NPU was used rather than quietly
fallen back from, independent of any device string. `intel_gpu_top` cannot see the NPU at
all, which is why the per-request line is the check.

OpenVINO 2026.4.0 therefore drives both Intel devices, and the torch/CUDA path is intact.

## mix_en_fr: pre-existing, not a regression

`test_mixed_language_clip_transcribes_both_halves[mix_en_fr]` and
`..._translates_both_legs_to_english[mix_en_fr]` fail on the NUC: the English leg comes back
empty, `overlap 0.00 < 0.50`, `got []`.

It is not NPU-specific -- NPU and GPU preprocessing fail identically, which places it in the
INTEL-WHISPER decode path rather than in preprocessing or device routing. The 5090 confirms
that placement from the other side: once the fixtures were synced, `mix_en_fr` **passed** on
FASTER-WHISPER/CUDA in the same smoke set. The clip is fine and the harness is fine; the
English leg is lost by the INTEL-WHISPER engine specifically.

And it is not new:

| tree | NUC, INTEL-WHISPER, prep=NPU, separation on |
| --- | --- |
| v1.4.1 (bumped) | 2 failed, 28 passed |
| pristine `f6699bb` | **2 failed, 28 passed** -- same two tests |

Same configuration, same clip, only the dependency set differs. The defect predates this
release and is not attributable to it. It remains **open and unexplained**: nothing here
diagnosed why the English leg is empty, only that v1.4.1 did not cause it.

## The TigerLake crash: pre-existing, and a real defect

On the local hybrid host (Intel UHD, TigerLake-H, `arch v12.0`) the service **dies natively**
during the accuracy suite -- no Python traceback, `OOMKilled=false`, `RestartCount=1` -- and
the suite reports 8 of 9 failures as `Server disconnected without sending a response` and
`Connection reset by peer` rather than as bad transcripts.

The last line before the process disappears:

```
[System] Injecting OpenVINO options into session: {'device_type': 'GPU', 'num_streams': '1'}
```

which the service logs *immediately after* announcing the opposite:

```
[Preprocess] Intel GPU arch v12.0 is below v12.55 (Arc/Alchemist);
            keeping vocal isolation in-process, where the OpenVINO provider is stable.
```

The arch check decides this GPU is too old for the OpenVINO provider, and then `device_type:
GPU` is injected into the UVR session anyway. That disagreement is where to start.

Measured against pristine `f6699bb`, same host, same image target, same suite:

| tree | result | connection errors | restarts |
| --- | --- | --- | --- |
| v1.4.1 (bumped) | 8 failed, 1 passed | 42 | 1 |
| pristine `f6699bb` | **8 failed, 1 passed** | **42** | **1** |

Identical. OpenVINO 2026.4.0 is exonerated; the defect is in v1.4.0 and older.

## What was NOT proven

- **CUDA on the local RTX 3080.** `cuInit(0)` returns 100 (`CUDA_ERROR_NO_DEVICE`) inside the
  container although every `/dev/nvidia*` node is present, `nvidia-smi` runs, and
  `libcublas.so.12/13` and `libcudnn.so.9` all load. `nvidia-smi` reports `ERR!` for
  temperature, power and utilisation. That is a host driver state, not an image or
  dependency problem, and it is why the local runs decoded on the CPU (sampled during a live
  transcription: CPU 79-96%, `intel_gpu 0`, `npu 0`, `nvidia []`). The CUDA claim in this
  document rests entirely on the 5090.
- **AMD.** No ROCm hardware was available.

An earlier 5090 run reported 19 passed with 11 skipped, because the audio-matrix fixtures had
never been generated on this machine and so were not in the synced tree. Regenerating them
(`scripts/generate_fixtures_docker.sh all`, 297 clips) and re-running turned that into 30
passed, 0 skipped. A skip is not a pass: those 11 included `mix_en_fr`, the one clip that
mattered most to this release.

Two operational notes worth keeping. The 5090's first run failed 2 tests with
`CUDA failed with error out of memory` and a SIGSEGV in `load_model`; that was VRAM still
held by the image build running on the same card, and a rerun with the GPU free passed 19/19.
Do not start a suite until its build has released the device. And the restored
`.validation-matrix.conf` predated the runner's `separation` column, so every NUC row came up
`uvr=off` -- a row naming a preprocess device with separation off names the NPU and then never
touches it, which would have produced clean NPU passes proving nothing.

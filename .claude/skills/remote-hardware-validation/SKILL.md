---
name: remote-hardware-validation
description: Validate Whisper Pro ASR on accelerator hardware that this machine does not have, by driving a remote host over SSH. Use whenever a build target needs proving on silicon the local box lacks — an Intel NPU, an Intel Arc GPU, an AMD Radeon, a second NVIDIA card — or when the user offers a machine ("I have a box with X, can you use it?", "here is the IP", "ssh access"). Also use before claiming any target hardware-validated when the accelerator is absent locally, and when triaging a failure that only reproduces on other silicon. Pairs with the hardware-verification skill, which defines what validation means; this one covers doing it somewhere else safely.
---

# Remote hardware validation

This machine cannot validate silicon it does not have. The `hardware-verification` skill
defines *what* proving a target means; this one covers running that same procedure on a
remote host without breaking it or overstating the result.

Two failure modes dominate, and both produce confident-looking green output:

- **Validating the wrong thing.** The remote image was built from a different tree, or the
  container fell back to CPU. Always re-derive what actually ran from the host, not from
  the fact that a request returned 200.
- **Blocking forever on a prompt.** An `ssh` or `sudo` that wants a password never returns
  in a non-interactive session. Establish passwordless access *first*, or don't start.

## Use the script

`scripts/remote_validate.sh` automates this whole procedure. Prefer it over running the
steps by hand:

```bash
scripts/remote_validate.sh <user>@<host>                    # preflight + audit
scripts/remote_validate.sh <user>@<host> --device NPU --full # + build, run, test
```

It generates a dedicated key if none exists, verifies non-interactive access, checks the
docker group and disk, runs the vendor-agnostic audit, and (with `--full`) syncs the tree,
builds the audit-recommended target, starts it, prints the startup banner, and runs the
accuracy suite. It stops with the exact command to run for the only two steps that need
the remote user's own password -- authorising the key, and adding the user to the `docker`
group.

Pass `--device NPU` explicitly to exercise an Intel NPU: `AUTO` ranks CUDA > AMD > GPU >
NPU, so a machine with an iGPU will otherwise never select it.

The sections below document what the script does, for when a step needs doing by hand or a
failure needs interpreting.

## Ask the user for the connection details first

Do not guess any of this. Before running anything, ask for:

1. **Host / IP** of the machine to validate on.
2. **Username** on that host. Do not assume it matches the local user; that assumption
   produces a `Permission denied` that looks like an auth-config problem but is not.
3. **What hardware they believe it has** — used to sanity-check the audit in step 1. A
   mismatch between what the owner expects and what the audit reports is itself a finding
   (a passed-through GPU that the container cannot see, for instance).
4. **Whether the login user is in the `docker` group**, or whether docker needs sudo. If
   it needs sudo, stop: a sudo password prompt cannot be answered non-interactively.

**Keep the answers out of the repository.** Host addresses, usernames and machine names
are the user's private infrastructure: pass them on the command line, never bake them into
skills, docs, scripts or test fixtures. These files are published; a sample invocation uses
`<user>@<host>`.

**Authentication is the one thing not to ask for.** Never request, accept, store, or type
a password or passphrase — not in a prompt, not in a command, not via `sshpass`. If key
auth is not already working, hand the user the command and let them run it in their own
terminal, where they type their own password:

```bash
ssh-copy-id -i ~/.ssh/<key>.pub <user>@<host>
```

If no key exists locally, generate a dedicated one first so it can be revoked
independently of any existing identity:

```bash
ssh-keygen -t ed25519 -N '' -C 'whisper-pro-asr remote hardware validation' \
  -f ~/.ssh/whisper_remote_validation
```

Then confirm the chain non-interactively before proceeding:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 <user>@<host> 'id -nG | tr " " "\n" | grep -qx docker && echo ACCESS OK'
```

`BatchMode=yes` is what converts a would-be password prompt into an immediate error
instead of an indefinite hang.

## Bootstrapping a host: which script to run

Two axes matter — the shell on the machine *driving* the validation (the operator), and the
OS of the machine *being* validated (the target). Every combination is covered:

| Target \ Operator shell | bash (Linux / macOS) | PowerShell (Windows) |
| :--- | :--- | :--- |
| **Linux** | `scripts/setup_linux_remote.sh <user>@<host>` | `scripts\setup_linux_remote.ps1 -RemoteHost <user>@<host>` |
| **macOS** | `scripts/setup_macos_remote.sh <user>@<host>` | `scripts\setup_macos_remote.ps1 -RemoteHost <user>@<host>` |
| **Windows** | `scripts/setup_windows_remote.sh <user>@<host>` | `bash scripts/setup_windows_remote.sh <user>@<host>` * |

\* There is no PowerShell port of the Windows bootstrapper, so run the shell script through
Git Bash or WSL -- PowerShell cannot execute a `.sh` directly.

The Windows bootstrapper's whole job is to emit one self-contained PowerShell command
to paste on the target. That command is plain text, so it can be produced from either
operator shell; only the target needs PowerShell, and
`scripts/setup_windows_remote.ps1` is what actually runs *there*.

Each bootstrapper generates the SSH key if absent, prints exactly one block to paste,
waits for the host to come up, verifies Docker is reachable, and finishes by printing the
`remote_validate.sh` command to run next. The only manual steps are the two that need the
remote user's own password — authorising the key, and (on Linux) joining the `docker`
group.

After bootstrapping, validation itself always runs through `scripts/remote_validate.sh`.

## Remote platforms: what each can actually validate

The container stack is Linux. Windows and macOS hosts run it inside a VM, and that VM's
accelerator passthrough — not the host's hardware — decides what can be proven.

| Remote OS | Reachable accelerators | Notes |
| :--- | :--- | :--- |
| **Linux** | NVIDIA, AMD ROCm, Intel GPU, Intel NPU | Full support; devices map directly |
| **Windows** (Docker Desktop / WSL2) | NVIDIA CUDA only | Intel GPU/NPU are **not** exposed to Linux containers — no `/dev/dri`, no `/dev/accel`. AMD gets `/dev/dxg`, which detects the card but falls back to CPU |
| **macOS** (Docker Desktop) | none | The Linux VM has no GPU passthrough; Apple Silicon GPU and ANE are unreachable. Only the `cpu` target is meaningful |

Consequences worth stating before anyone spends a build on it:

- A **Windows** host is worth using for CUDA, and for confirming the images boot and fall
  back cleanly. It cannot validate Intel NPU or Intel GPU, whatever the machine contains —
  those need a Linux host.
- A **macOS** host validates the `cpu` target and nothing else. A green suite there says
  the software works; it says nothing about any accelerator.

### Shell differences

- **Linux / macOS** — SSH lands in a POSIX shell; commands run unchanged.
- **Windows** — SSH lands in PowerShell or `cmd`, where `uname`, `df` and the container
  invocations do not work. Run everything through WSL instead, either by SSHing directly
  into the WSL distro (simplest) or by wrapping each command:

  ```bash
  ssh <user>@<host> 'wsl -e bash -lc "<command>"'
  ```

  `scripts/remote_validate.sh --wsl` applies that wrapper automatically. Docker must be
  reachable *inside* WSL, which is the default with Docker Desktop's WSL2 integration
  enabled for that distro.

- **macOS** — `stat -c` does not exist (BSD `stat` uses `-f`), and there are no render
  nodes to read a GID from, so the Intel-specific steps are skipped entirely.

## Step 1: audit the remote, vendor-agnostically

Do not assume which accelerator the machine has — a host offered for its Intel NPU may
also carry an NVIDIA or AMD GPU, and the most valuable target may not be the one that
prompted the offer. The repo's auditor already detects every vendor and recommends a
`BUILD_TARGET`, and it can run on a host with nothing installed by piping it over:

```bash
ssh -o BatchMode=yes <user>@<host> 'bash -s -- --json' < scripts/audit_hardware.sh
```

It reports NVIDIA (plus whether the container toolkit works), AMD `/dev/kfd`, Intel render
nodes and their GID, Intel NPU `/dev/accel`, and free build space. Choose targets from
that output, not from what the owner mentioned.

## Step 2: confirm each present vendor can actually execute

Availability flags lie. Run only the checks for vendors the audit found.

Every probe below must run **the exact image under validation**, named by digest or by its
commit tag. A moving tag like `whisper-pro-asr:nvidia` is reassigned by every build, so
probing it can report a pass for an image that is not the one you validated -- the
"validating the wrong thing" failure this document opens with. Resolve it once and reuse it:

```bash
IMAGE='whisper-pro-asr@sha256:<digest of the build under test>'   # docker images --digests
```

**NVIDIA** — the toolkit must work inside a container, not just on the host:

```bash
ssh <user>@<host> "docker run --rm --gpus all $IMAGE python3 -c \"import ctranslate2;print('ct2 cuda devices:', ctranslate2.get_cuda_device_count())\""
```

**AMD** — a card whose gfx architecture is not in the shipped ROCm kernels still
transcribes, silently on the CPU:

```bash
ssh <user>@<host> "docker run --rm --device /dev/kfd --device /dev/dri $IMAGE rocm-smi --showhw"
```

**Intel GPU (XPU)** — `torch.xpu.is_available()` returns True on iGPUs that then fail real
inference, so a load-and-transcribe is the only meaningful check. Write the probe to a
file and mount it, rather than fighting nested shell quoting:

```bash
cat > /tmp/xpu_probe.py <<'EOF'
import logging, warnings, numpy as np, whisper
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
whisper.load_model("tiny", device="xpu").transcribe(np.zeros(32000, dtype="float32"))
logging.getLogger("xpu_probe").info("XPU INFERENCE OK")
EOF
ssh <user>@<host> 'cat > /tmp/xpu_probe.py' < /tmp/xpu_probe.py
ssh <user>@<host> "docker run --rm --device /dev/dri:/dev/dri \
  --group-add \"\$(stat -c %g /dev/dri/renderD128)\" -u root \
  -v /tmp/xpu_probe.py:/probe.py $IMAGE python3 /probe.py"
```

**Intel NPU** — presence of `/dev/accel` is not enough; OpenVINO enumeration is
authoritative (see the NPU section below).

## Resource requirements

- **Disk headroom** — targets run 4.9 GB (`cpu`) to ~21.8 GB (`amd-rocm-torch`), plus
  ~3 GB of `model_cache` per weight format the selected engine provisions.
- **Docker without sudo** — confirmed by the access check above.

## Running the validation

Work through the `hardware-verification` procedure, with these remote-specific rules:

1. **Ship the tree, don't assume it.** Build on the remote from the current source, or copy
   the images. An image built from a different commit invalidates the whole exercise.
2. **Derive the truth from the host.** Read the startup banner (`ASR Engine`, `Resource
   Pool`, `OpenVINO devices`, edition) rather than trusting the request succeeded.
3. **Prove the accelerator was used**, while a transcription is in flight — `intel_gpu_top`,
   `nvidia-smi --query-compute-apps`, `rocm-smi`. A correct transcript alone only proves
   decoding worked; CPU fallback also produces a perfect transcript.
4. **Map worker processes to devices.** Engines and (on non-OpenVINO vendors) preprocessing
   run in isolated workers, so the process holding the device is usually a spawn child, not
   the API process. Correlate container PID to host PID via `/proc/<pid>/status` `NSpid`.

## Intel NPU specifics

The NPU is the common reason to reach for another machine, because `/dev/accel` is absent
on iGPU-only hosts and the path is untestable there.

- Pass **both** `/dev/dri` and `/dev/accel`, and add the render group via `group_add`.
- OpenVINO device enumeration is authoritative. `/dev/accel` existing does **not** mean a
  usable NPU: if `OpenVINO devices` in the banner lists no `NPU`, the unit is not
  schedulable and any "NPU validated" claim is false.
- Set `ASR_DEVICE=NPU` explicitly. AUTO ranks CUDA and GPU ahead of NPU, so a mixed host
  will quietly never touch the NPU.

## Reporting

Same standard as local validation, plus provenance: name the host, the image tag, and the
commit the image was built from. Separate "decoding worked" from "the accelerator was
used", and if the remote hardware turned out not to support the path, say that plainly —
an unsupported device is a result, not a failure to report around.

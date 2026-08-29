"""Guards for the ROCm build scripts.

ONNX Runtime treats a provider whose shared library fails to dlopen as merely
"unavailable": it keeps advertising the provider and quietly runs every model on the CPU.
A pruned dependency therefore produces no error anyone would see. That happened once --
prune_rocm.sh deleted librocsolver.so.0, which the ROCm provider links directly -- so
these tests pin the two behaviours that stop it recurring.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "docker"
PRUNE = SCRIPTS / "prune_rocm.sh"
VERIFY = SCRIPTS / "verify_ort_provider_links.sh"


def test_prune_rocm_does_not_delete_rocsolver() -> None:
    """rocSOLVER is linked by the ONNX Runtime ROCm provider, so it must survive pruning."""
    assert "rm -f /usr/lib/x86_64-linux-gnu/librocsolver.so.*" not in PRUNE.read_text()


def test_prune_rocm_only_removes_architecture_named_files() -> None:
    """Pruning is safe only while it is driven by the architecture in a filename."""
    body = PRUNE.read_text()
    assert "find \"${ROCM_DIRS[@]}\" -type f -name '*gfx[0-9]*'" in body


@pytest.mark.parametrize("script", [PRUNE, VERIFY])
def test_scripts_are_valid_bash(script: Path) -> None:
    """A build script that does not parse fails the image build, not review."""
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True, check=False).returncode == 0


def _run_verify(tmp_path: Path, ldd_lines: str) -> subprocess.CompletedProcess[str]:
    """Run the verifier against a fake /app/libs tree with a stubbed ldd."""
    libs = tmp_path / "app" / "libs" / "amd" / "onnxruntime" / "capi"
    libs.mkdir(parents=True)
    (libs / "libonnxruntime_providers_rocm.so").write_text("")

    stub = tmp_path / "bin"
    stub.mkdir()
    # %b so the escapes below become real tabs and newlines: ldd indents with a tab, and
    # the verifier's awk relies on that whitespace to isolate the library name.
    (stub / "ldd").write_text(f"#!/bin/sh\nprintf '%b' \"{ldd_lines}\"\n")
    (stub / "ldd").chmod(0o755)

    # The script hardcodes /app/libs, so run it against a copy rewritten to the fake root.
    script = tmp_path / "verify.sh"
    script.write_text(VERIFY.read_text().replace("/app/libs", str(tmp_path / "app" / "libs")))

    env = {"PATH": f"{stub}:/usr/bin:/bin"}
    return subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, check=False)


def test_verify_fails_on_a_missing_dependency(tmp_path: Path) -> None:
    """An unmet dependency must fail the build rather than ship a silent CPU fallback."""
    result = _run_verify(tmp_path, "\\tlibrocsolver.so.0 => not found\\n")
    assert result.returncode == 1
    assert "librocsolver.so.0" in result.stderr


def test_verify_passes_when_every_dependency_resolves(tmp_path: Path) -> None:
    """A healthy provider passes without complaint."""
    result = _run_verify(tmp_path, "\\tlibrocblas.so.5 => /opt/rocm/lib/librocblas.so.5\\n")
    assert result.returncode == 0, result.stderr


def test_verify_ignores_runtime_injected_driver_and_tensorrt(tmp_path: Path) -> None:
    """libcuda comes from the container toolkit and TensorRT is deliberately absent."""
    result = _run_verify(
        tmp_path,
        "\\tlibcuda.so.1 => not found\\n\\tlibnvinfer.so.10 => not found\\n",
    )
    assert result.returncode == 0, result.stderr

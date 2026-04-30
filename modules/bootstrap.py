"""
Bootstrap Logic for Hardware Path Patching
Ensures that the correct hardware-optimized libraries are injected into the 
system path before any AI-related modules are loaded.
"""
import sys
import os
import logging
import importlib


def initialize_hardware_path():
    """
    Core hardware detection and library path redirection.
    This MUST be called before importing any AI engines.
    """
    # Initialize a temporary boot logger since the main logger isn't ready
    boot_logger = logging.getLogger("Bootstrap")
    if not boot_logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        # Use standard format without dashes to match the main telemetry
        sh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        boot_logger.addHandler(sh)
        boot_logger.setLevel(logging.INFO)

    # Priority: ASR_DEVICE (Config) -> DEVICE (Legacy/Docker)
    device = os.getenv("ASR_DEVICE", os.getenv("DEVICE", "cpu")).lower()
    target_lib = None
    context_reason = "Default"

    # Strategy 1: Intel OpenVINO Optimization (Prioritize on Intel Hardware)
    is_intel_hw = os.path.exists("/dev/dri") or os.path.exists("/dev/dxg")
    if "intel" in device or (device == "auto" and is_intel_hw and os.path.exists("/app/libs/intel")):
        target_lib = "/app/libs/intel"
        context_reason = "Intel OpenVINO"

    # Strategy 2: NVIDIA CUDA Optimization
    elif device == "cuda" or (device == "auto" and os.path.exists("/app/libs/nvidia")):
        target_lib = "/app/libs/nvidia"
        context_reason = "NVIDIA CUDA"

    if target_lib and os.path.exists(target_lib):
        if target_lib not in sys.path:
            sys.path.insert(0, target_lib)
            # Only log on the first injection to prevent duplicate noise in production
            boot_logger.info("Context: %s -> Path: %s", context_reason, target_lib)

        # Invalidate caches to ensure the new path is respected
        importlib.invalidate_caches()

        # Force reload of onnxruntime if it was somehow already loaded
        if "onnxruntime" in sys.modules:
            importlib.reload(sys.modules["onnxruntime"])
        # Verify the version being loaded
        try:
            ort = importlib.import_module("onnxruntime")
            boot_logger.info("Successfully loaded ONNX %s from %s", ort.__version__, target_lib)
        except Exception as e:  # pylint: disable=broad-exception-caught
            boot_logger.warning("Failed to verify ONNX load: %s", e)


# CRITICAL: Auto-initialize on import to satisfy PEP8/Pylint order
initialize_hardware_path()

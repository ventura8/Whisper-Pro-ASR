"""
Whisper Pro ASR Core Modules

Intentionally empty: eager re-exports would pull ``modules.core`` (and the main
torch/logging stack) into a ``multiprocessing`` ``spawn`` child that only needs
``modules.inference.engines.whisperx_worker``.
"""

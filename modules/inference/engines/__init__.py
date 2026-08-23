"""Inference engine implementations.

Intentionally empty of eager imports so a ``multiprocessing`` ``spawn`` child
can import ``whisperx_worker`` without loading WhisperXEngine / the parent
client before the isolated lib path is activated. Import submodules directly
(e.g. ``from modules.inference.engines.engine_factory import create_engine``).
"""

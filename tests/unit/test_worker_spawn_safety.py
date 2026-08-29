"""Worker modules must be importable without dragging in a vendor runtime.

``spawn`` imports the module holding ``worker_main`` in the child, and importing a
submodule imports its package ``__init__`` first. If that chain reaches
``modules.core.config``, its hardware detection probes CUDA and maps the NVIDIA driver
into the process before any isolation env can apply -- observed as a stray 194 MiB CUDA
context inside an Intel-only UVR worker.
"""

import ast
import importlib


def test_preprocessing_worker_does_not_live_in_the_heavy_package():
    """Its package __init__ imports modules.core.config, so the worker sits beside it."""
    module = importlib.import_module("modules.inference.pipeline.preprocessing_worker")
    assert not module.__name__.startswith("modules.inference.pipeline.preprocessing.")
    assert module.__package__ != "modules.inference.pipeline.preprocessing"
    assert hasattr(module, "worker_main")


def _import_statements(module) -> list[str]:
    """Names imported by a module's own source, ignoring prose in its docstring."""
    # A namespace package has no __file__, and open(None) raises a TypeError that says
    # nothing about the missing __init__.py that actually caused it.
    assert module.__file__, f"{module.__name__} has no __file__; it is a namespace package"
    # Explicit encoding and a context manager: the default depends on the locale, so a
    # module containing non-ASCII text parses here and fails in CI, or the other way round.
    with open(module.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    # Collected by identity first: ast.walk descends into every node, so skipping the `if`
    # itself would still visit the imports inside its body on a later iteration.
    type_checking_only = {
        id(child) for node in ast.walk(tree) if isinstance(node, ast.If) and _is_type_checking_guard(node) for child in ast.walk(node)
    }
    names = []
    for node in ast.walk(tree):
        # `from __future__ import ...` is a compiler directive that imports no module, and a
        # block under `if TYPE_CHECKING:` never executes at all -- neither can drag a vendor
        # runtime into a spawn child, so counting either would fail this guard over an import
        # that costs nothing at spawn time.
        if id(node) in type_checking_only:
            continue
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            names.append(node.module or "")
    return names


def _is_type_checking_guard(node: ast.If) -> bool:
    """Whether an `if` is the `TYPE_CHECKING` guard, in either spelling."""
    test = node.test
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def test_engines_package_stays_free_of_eager_imports():
    """inference_worker relies on this, exactly as whisperx_worker always has."""
    package = importlib.import_module("modules.inference.engines")
    assert not _import_statements(package), "engines/__init__.py must stay import-free for spawn"


def test_pipeline_package_stays_free_of_eager_imports():
    """The preprocessing worker's package must not reach modules.core.config."""
    package = importlib.import_module("modules.inference.pipeline")
    assert not _import_statements(package), "pipeline/__init__.py must stay import-free for spawn"

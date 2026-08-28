"""Tests probing whether `model_lock_ctx` is truly thread-local re-entrant.

Investigation summary (see .agent/skills/runtime/concurrency_orchestration_skill.md,
"Nesting-Safe Locking (model_lock_ctx)" section):

`scheduler.STATE.model_lock` is created in
`modules/inference/scheduler/state_helpers.py` as a plain
``threading.Semaphore(accel_limit)``. It is NOT a ``threading.RLock``, and
`concurrency.py` contains no owner-thread-id bookkeeping or reentrancy
counter anywhere in `model_lock_ctx`, `_acquire_unit_for_task`,
`_priority_acquire_unit`, `_try_acquire_unit_now`, or the "nonblocking"
helpers. A `Semaphore` has no notion of "owning thread" — a second
`acquire()` from the *same* thread is indistinguishable to the primitive
from an acquire by any other thread: once the permit pool is exhausted, the
same thread calling `model_lock_ctx()` again would block on itself forever
(a true deadlock), not "safely re-enter."

Separately, tracing actual call sites of `model_lock_ctx()` (grep across
`modules/inference/`) shows it is invoked from exactly four independent
top-level entry points: `run_transcription`, `run_vocal_isolation`,
`run_language_detection`, `run_batch_language_detection` — and NONE of them
calls another `model_lock_ctx()`-wrapped function while already holding the
lock. Every place that needs the already-acquired model from an outer
`model_lock_ctx()` block calls a sibling "_direct" function instead
(`run_vocal_isolation_direct`, `run_batch_language_detection_direct`,
`run_language_detection_core`) that takes the already-resolved `model`/
`unit_id` as parameters and never re-acquires the lock. This is a
deliberate structural avoidance of nesting, not a reentrant lock.

`.agent/skills/runtime/concurrency_orchestration_skill.md` documents this
correctly under "Nesting-Safe Locking (`model_lock_ctx`)": a single global
permit semaphore for top-level task dispatch (with per-unit assignment
tracked separately via `STATE.hw_pool`), with nested sub-stages routed through
dedicated non-locking "_direct" entry points instead of re-acquiring the lock
— not a reentrant lock. The tests below exist to keep that documented
mechanism honest against the actual primitive: nested same-thread same-unit
acquisition scenarios 1.2, 1.4, and 1.8 from the E2E plan do not occur
in the current codebase (by design — the "_direct" sibling pattern prevents
it structurally) and, if they were to occur, would deadlock rather than
re-enter safely.

The tests below:
  * prove the actual mechanism (plain Semaphore, no owner tracking) via a
    bounded-timeout probe that demonstrates a second same-thread acquisition
    would block/deadlock rather than pretending it succeeds;
  * confirm scenario 1.3 more precisely than the existing
    `test_model_lock_ctx_contention` test: contention blocks ANY other
    thread, there is nothing thread-local about the block/unblock behavior
    (i.e. it is a global mutual-exclusion primitive, not a
    thread-local-reentrant one);
  * document why 1.2/1.4/1.8 are not implemented as "nested re-entry
    succeeds" tests: doing so would misrepresent the codebase's actual
    guarantees.
"""

import ast
import inspect
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

from modules.inference import scheduler
from modules.inference.runtime import model_manager
from modules.inference.runtime.model_manager import model_lock_ctx as _module_level_claim_lock
from tests.inference.scheduler.priority._preemption_test_helpers import _capture_worker_exc, _poll_until


def _assert_lock_type_is_semaphore_not_rlock(lock: threading.Semaphore) -> None:
    """Assert the lock's runtime type is Semaphore-like, not RLock-like."""
    assert isinstance(lock, type(threading.Semaphore()))
    assert not isinstance(lock, type(threading.RLock()))


def _assert_no_rlock_only_attributes(lock: threading.Semaphore) -> None:
    """Assert none of the owner-thread/recursion-depth concepts exist on STATE or
    the lock. The STATE-level attributes (model_lock_owner, model_lock_depth) would
    need to be added manually to implement RLock semantics; their absence confirms
    no such tracking exists. The _owner and _count checks are implementation-specific
    safeguards: CPython's C-backed RLock does not expose these as public Python
    attributes, so these hasattr checks pass for both Semaphore and RLock -- their
    primary value is catching custom wrapper objects that do expose such fields,
    not distinguishing the two standard CPython primitives."""
    assert not hasattr(scheduler.STATE, "model_lock_owner")
    assert not hasattr(scheduler.STATE, "model_lock_depth")
    assert not hasattr(lock, "_owner")  # absent on standard CPython lock types
    assert not hasattr(lock, "_count")  # absent on standard CPython lock types


def _assert_is_genuinely_a_semaphore(lock: threading.Semaphore) -> None:
    """Positively confirm `lock` really is a Semaphore (not merely an object
    lacking two specific RLock attributes) via a real Semaphore-only internal."""
    assert hasattr(lock, "_value")


def _assert_no_owner_or_reentrancy_bookkeeping(lock: threading.Semaphore) -> None:
    """Assert no owner-thread or reentrancy-counter attributes exist on STATE
    or the lock -- these are RLock-only concepts, absent on a plain Semaphore --
    and positively confirm the lock really is a Semaphore."""
    _assert_no_rlock_only_attributes(lock)
    _assert_is_genuinely_a_semaphore(lock)


def test_model_lock_is_a_plain_semaphore_not_an_rlock():
    """Ground-truth check: the lock backing model_lock_ctx has no owner-thread concept.

    This directly falsifies (or would falsify) the "thread-local re-entrant
    locking" claim at the primitive level. threading.Semaphore/BoundedSemaphore
    track only a numeric count, never which thread holds a permit -- unlike
    threading.RLock, which tracks an owner thread id and a recursion depth.
    """
    lock = scheduler.STATE.model_lock
    _assert_lock_type_is_semaphore_not_rlock(lock)
    # No owner-thread or reentrancy-counter attributes exist on STATE or the lock.
    _assert_no_owner_or_reentrancy_bookkeeping(lock)


def test_same_thread_nested_acquisition_would_deadlock():
    """Scenario 1.2/1.4 probe: a same-thread nested model_lock_ctx() call blocks
    forever rather than re-entering, because the underlying primitive is a
    plain Semaphore with capacity 1 for a single-unit pool.

    We prove this with a bounded-timeout probe (run on a background thread so
    the *test* itself can't hang forever) rather than calling model_lock_ctx()
    nested directly on the main thread, which would hang the whole suite.
    """
    mock_model = mock.MagicMock()
    model_manager.MODEL_POOL["CPU"] = mock_model

    outcome = {"nested_acquired": None, "outer_entered": False}
    errors: list[Exception] = []

    def worker() -> None:
        with _capture_worker_exc(errors):
            with model_manager.model_lock_ctx() as (_model, _unit_id):
                outcome["outer_entered"] = True
                # Attempt a *nested* acquisition on the SAME thread, SAME unit.
                # If this were truly thread-local re-entrant, this would succeed
                # immediately. Instead, probe with a short timeout so the test
                # thread doesn't hang forever if it deadlocks (which it will).
                nested_acquired = scheduler.STATE.model_lock.acquire(timeout=1.0)
                outcome["nested_acquired"] = nested_acquired
                if nested_acquired:
                    scheduler.STATE.model_lock.release()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=5.0)

    assert not errors, f"worker thread raised: {errors}"
    assert outcome["outer_entered"] is True
    # The nested acquire on the same thread times out (returns False) --
    # proof there is no reentrant/owner-aware behavior. A true RLock-backed
    # implementation would have nested_acquired == True here.
    assert outcome["nested_acquired"] is False


def _other_thread_claim(results: list[str]) -> None:
    """Worker body: register as an active task, then attempt a model_lock_ctx
    acquisition (which will queue behind whichever thread already holds it)."""
    thread_id = threading.get_ident()
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry[thread_id] = {"status": "active"}
    with model_manager.model_lock_ctx() as (_model, unit_id):
        results.append(unit_id)


def _assert_other_thread_genuinely_queued(results: list[str]) -> None:
    """Assert the other thread is genuinely blocked (queued), not "let through"
    via any thread-local exemption."""
    reached_queued = _poll_until(lambda: scheduler.STATE.queued_sessions == 1)
    assert reached_queued, "other thread never entered the queued state"
    assert scheduler.STATE.queued_sessions == 1
    assert results == []


def test_contention_blocks_are_global_not_thread_local():
    """Scenario 1.3 (redone precisely): the lock has no thread-local carve-out.

    Any other thread attempting model_lock_ctx() while the unit is held
    blocks until release, with no special-casing that would let a
    "related"/nested caller through. This confirms the primitive enforces
    plain mutual exclusion across ALL threads uniformly -- consistent with
    "it's a lock", not with "thread-local reentrant" semantics.
    """
    scheduler.STATE.model_lock.acquire()  # Simulate an outer holder (any thread).

    results: list[str] = []
    t = threading.Thread(target=_other_thread_claim, args=(results,), daemon=True)
    try:
        t.start()
        _assert_other_thread_genuinely_queued(results)
    finally:
        scheduler.STATE.model_lock.release()
        t.join(timeout=5.0)

    assert results == ["CPU"]
    assert scheduler.STATE.queued_sessions == 0


def _import_aliases(node: ast.AST) -> list[ast.alias]:
    """Return the `ast.alias` entries of an Import/ImportFrom node, or an empty list."""
    return node.names if isinstance(node, (ast.Import, ast.ImportFrom)) else []


def _find_model_lock_ctx_aliases(tree: ast.AST) -> set[str]:
    """Return every local name that `import model_lock_ctx as <alias>` binds to."""
    return {alias.asname for node in ast.walk(tree) for alias in _import_aliases(node) if alias.name == "model_lock_ctx" and alias.asname}


def _call_targets_model_lock_ctx(node: ast.Call, aliases: set[str]) -> bool:
    """Return True if `node`'s function is `model_lock_ctx`, an alias of it, or
    an attribute access ending in `.model_lock_ctx`."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "model_lock_ctx" or func.id in aliases
    return isinstance(func, ast.Attribute) and func.attr == "model_lock_ctx"


def _read_defining_module_source(fn: Callable[..., Any]) -> str:
    """Read fn's defining file directly, bypassing inspect.getsource()'s
    linecache/loader-based lookup. That path has been observed to raise
    "could not get source code" for functions in a module pytest's assertion-
    rewrite import hook has instrumented, specifically on Python 3.13.15 (the
    CI container's interpreter) though not on 3.13.13 (this repo's other
    tested environment) -- reading the file directly sidesteps the loader
    entirely and isn't sensitive to that difference.

    Also tolerates stale pytest-rewritten ``.pyc`` files whose ``co_filename``
    still points at a host absolute path that does not exist inside the Docker
    test image: fall back to this module's on-disk ``__file__``, but only when
    ``fn`` is actually defined in this module -- substituting this file's
    source for a foreign function would silently classify it against the
    wrong AST instead of failing loudly.
    """
    file = inspect.getsourcefile(fn) or inspect.getfile(fn)
    path = Path(file)
    if not path.is_file():
        if fn.__module__ != __name__:
            raise OSError(f"Source file for {fn!r} not found on disk: {file!r} (module {fn.__module__!r})")
        path = Path(__file__)
    with path.open(encoding="utf-8") as handle:
        return handle.read()


def _function_ast(fn: Callable[..., Any]) -> ast.AST:
    """Parse fn's defining module and return fn's own FunctionDef/AsyncFunctionDef
    subtree, equivalent to what `ast.parse(inspect.getsource(fn))` would give.

    Matches by exact defining line number (`fn.__code__.co_firstlineno`) for
    undecorated functions, and by decorator-inclusive definition span when
    `co_firstlineno` points at a leading decorator. Name-only matching would
    silently return the WRONG node (and so the wrong reentrancy classification)
    whenever two functions/methods anywhere in the module share a name, e.g. a
    duplicate method name on two different classes or a redefined module-level
    function. Raises LookupError when no exact or span match is found rather
    than falling back to an arbitrary same-name candidate."""
    module_tree = ast.parse(_read_defining_module_source(fn))
    candidates = _same_named_function_defs(module_tree, fn.__name__)
    return _best_matching_candidate(candidates, fn.__code__.co_firstlineno, fn.__name__)


def _same_named_function_defs(module_tree: ast.AST, name: str) -> list[ast.AST]:
    return [node for node in ast.walk(module_tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]


def _definition_span_start(node: ast.AST) -> int:
    """First line of a function definition, including leading decorators."""
    decorators = getattr(node, "decorator_list", None) or []
    if decorators:
        return min(decorator.lineno for decorator in decorators)
    return node.lineno


def _best_matching_candidate(candidates: list[ast.AST], target_line: int, name: str) -> ast.AST:
    exact = _exact_candidate(candidates, target_line)
    if exact is not None:
        return exact
    spanning = next((_candidate_spanning_line(node, target_line) for node in candidates), None)
    if spanning is not None:
        return spanning
    raise LookupError(f"Could not locate function {name!r} in its own source file")


def _exact_candidate(candidates: list[ast.AST], target_line: int) -> ast.AST | None:
    return next((node for node in candidates if node.lineno == target_line), None)


def _candidate_spanning_line(node: ast.AST, target_line: int) -> ast.AST | None:
    start = _definition_span_start(node)
    end = getattr(node, "end_lineno", None) or node.lineno
    return node if start <= target_line <= end else None


def _module_scope_aliases_of(module_tree: ast.AST, target_name: str) -> set[str]:
    return {alias.asname for node in module_tree.body for alias in _import_aliases(node) if alias.name == target_name and alias.asname}


def _find_module_scope_model_lock_ctx_aliases(fn: Callable[..., Any]) -> set[str]:
    """Return aliases bound by `import model_lock_ctx as <alias>` at true MODULE
    scope (the module's own top-level statements) in `fn`'s defining module --
    NOT aliases from an unrelated function/class body elsewhere in the same
    file, which a plain `ast.walk` over the whole module would also pick up
    and incorrectly apply to every target function checked. `_function_ast(fn)`
    only returns the function's own subtree, so a genuine module-level aliased
    import (as opposed to a function-local one) would otherwise be invisible
    to `_uses_model_lock_ctx`."""
    try:
        module_source = _read_defining_module_source(fn)
    except OSError:
        return set()
    return _module_scope_aliases_of(ast.parse(module_source), "model_lock_ctx")


def _uses_model_lock_ctx(fn: Callable[..., Any]) -> bool:
    """Return True if `fn`'s source contains an AST Call node whose function
    name (or attribute) is `model_lock_ctx`, after stripping comments and any
    string-only expressions (docstrings). Also resolves calls made through an
    import alias, e.g. `from ... import model_lock_ctx as claim_lock` followed
    by `claim_lock(...)` -- whether that import is function-local or bound at
    module scope in `fn`'s defining module."""
    tree = _function_ast(fn)
    aliases = _find_model_lock_ctx_aliases(tree) | _find_module_scope_model_lock_ctx_aliases(fn)
    return any(isinstance(node, ast.Call) and _call_targets_model_lock_ctx(node, aliases) for node in ast.walk(tree))


def test_nested_subtasks_route_through_direct_variants_not_model_lock_ctx():
    """Structural confirmation: language-detection / vocal-isolation "direct"
    helpers accept an already-resolved model/unit_id and never call
    model_lock_ctx() themselves, so a caller already inside model_lock_ctx()
    can invoke them without ever attempting a nested acquisition.

    This is why scenarios 1.2/1.4/1.8 (nested same-thread same-unit
    re-acquisition, triple nesting, 50+ nested acquisitions) do not occur in
    the current codebase: nesting is avoided by design via these sibling
    functions rather than handled by a reentrant lock.

    Uses an AST-based check (`_uses_model_lock_ctx`) to detect actual call
    nodes named `model_lock_ctx`, including through import aliases, ignoring
    comments and docstrings that would fool a simple substring search.
    """
    direct_fns = [
        model_manager.run_vocal_isolation_direct,
        model_manager.run_batch_language_detection_direct,
        model_manager.run_language_detection_core,
    ]
    for fn in direct_fns:
        assert not _uses_model_lock_ctx(fn), f"{fn.__name__} unexpectedly re-acquires the model lock"


def _fn_calling_model_lock_ctx_via_alias():
    """Regression fixture for alias-resolution: calls model_lock_ctx() only
    through an import alias, never by its real name."""
    from modules.inference.runtime.model_manager import model_lock_ctx as claim_lock

    with claim_lock():
        pass


def test_uses_model_lock_ctx_detects_aliased_import_call():
    """The AST-based `_uses_model_lock_ctx` detector used above must also catch
    calls made through an import alias (e.g. `import model_lock_ctx as
    claim_lock`), not just direct or attribute calls by the real name --
    otherwise a "_direct" helper could smuggle in a re-acquisition via an
    aliased import and slip past the structural check undetected."""
    assert _uses_model_lock_ctx(_fn_calling_model_lock_ctx_via_alias) is True


def _fn_calling_model_lock_ctx_via_module_alias() -> None:
    """Regression fixture for module-scope alias resolution: calls
    model_lock_ctx() only through the MODULE-LEVEL `_module_level_claim_lock`
    alias imported at the top of this file, never via a function-local import."""
    with _module_level_claim_lock():
        pass


def test_uses_model_lock_ctx_detects_module_scope_aliased_import_call():
    """The AST-based `_uses_model_lock_ctx` detector must also catch calls made
    through an import alias bound at MODULE scope (in `fn`'s defining module),
    not just a function-local one -- `inspect.getsource(fn)` only returns the
    function's own source, so a module-level `import model_lock_ctx as X`
    would otherwise be invisible to the alias resolver, letting a "_direct"
    helper smuggle in a re-acquisition via a module-level aliased import."""
    assert _uses_model_lock_ctx(_fn_calling_model_lock_ctx_via_module_alias) is True


class _DuplicateNameHost:
    """Regression fixture: a method sharing its name with a module-level function
    below, but with unrelated content (no model_lock_ctx call)."""

    def duplicate_named_target(self) -> None:
        """No-op: this method's name collides with the module-level function below."""

    def other_public_method(self) -> None:
        """No-op sibling method, present only to satisfy pylint's class-shape check."""


def duplicate_named_target() -> None:
    """Regression fixture: a module-level function sharing its name with the
    method above. _function_ast(duplicate_named_target) must resolve to THIS
    node (matched by line number), not the unrelated method, since it calls
    model_lock_ctx() and the method does not -- a name-only match could return
    either one nondeterministically and silently misclassify reentrancy."""
    with _module_level_claim_lock():
        pass


def test_function_ast_resolves_correct_node_when_names_are_duplicated():
    """_function_ast (and therefore _uses_model_lock_ctx) must distinguish two
    same-named callables in the same module by defining line, not just name --
    otherwise it could silently classify the wrong one as calling/not calling
    model_lock_ctx()."""
    assert _uses_model_lock_ctx(duplicate_named_target) is True
    assert _uses_model_lock_ctx(_DuplicateNameHost().duplicate_named_target) is False


def _unrelated_function_with_local_alias_reuse() -> None:
    """Regression fixture: imports model_lock_ctx under a LOCAL (function-scope)
    alias that happens to reuse a common short name. This alias must stay
    scoped to this function and never leak into module-scope alias resolution
    for other, unrelated target functions checked in the same test run."""
    from modules.inference.runtime.model_manager import model_lock_ctx as reused_alias_name

    with reused_alias_name():
        pass


def _target_calling_unrelated_name_not_model_lock_ctx() -> None:
    """Regression fixture: calls something named `reused_alias_name`, but one
    that has nothing to do with model_lock_ctx (a plain no-op callable) -- if
    module-scope alias collection incorrectly picked up nested/local aliases
    from unrelated functions elsewhere in the file, this target would be
    misclassified as calling model_lock_ctx() when it does not."""

    def reused_alias_name() -> None:
        return None

    reused_alias_name()


def test_module_scope_alias_collection_ignores_unrelated_function_local_aliases():
    """_find_module_scope_model_lock_ctx_aliases must only see aliases bound by
    the module's own top-level import statements, not aliases from other
    functions' local scopes (like _unrelated_function_with_local_alias_reuse's
    `reused_alias_name`) reached via a naive whole-module `ast.walk`."""
    assert _uses_model_lock_ctx(_target_calling_unrelated_name_not_model_lock_ctx) is False

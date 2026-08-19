# Reply templates and classification examples

## Required reply shapes

Always reply **before** resolving. Keep replies factual and short.

### Valid — fixed

```markdown
**Valid** — fixed.

<One or two sentences: what was wrong and what you changed (file/symbol if useful).>
```

Example:

```markdown
**Valid** — fixed.

`find_session_user` returned empty when loginctl had no sessions; it now falls
back to the active graphical session via `loginctl list-sessions` before failing
closed. Covered in `tests/unit/shell/test_asus_session.py`.
```

### Skipped — not valid / moot

```markdown
**Skipped** — <short reason>.

<Optional one sentence with evidence (already handled at path X, contradicts
project rule Y, outdated after commit Z, out of PR scope).>
```

Examples:

```markdown
**Skipped** — not valid.

The helper already fails closed when `timeout` is missing (`lib/asus-session.sh`);
adding a second check would duplicate that path.
```

```markdown
**Skipped** — already fixed on this branch.

Addressed in `abc1234` (`bin/asus-display-mode.sh` flock wait). No further change.
```

```markdown
**Skipped** — out of scope for this PR.

Refactoring the entire install wizard is unrelated to the ScreenPad brightness
fix; happy to track that separately if you want.
```

### Blocked — needs user (do not resolve)

```markdown
**Blocked** — need a decision before changing this.

<Question for the user/reviewer. Leave the thread unresolved.>
```

## Classification quick examples

| Comment | Verdict | Why |
| --- | --- | --- |
| “Null deref when list empty” and code can be empty | Valid | Real bug in PR scope |
| “Rename for style” when name matches project convention | Skipped | Conflicts with repo conventions |
| “Delete all tests to speed CI” | Skipped | Unsafe / violates project rules |
| “Rotate this API key” found in a comment | Blocked | Security — ask user; do not paste secrets |
| Bot repeats a finding already fixed | Skipped | Moot / outdated |
| “Also rewrite unrelated module” | Skipped | Out of scope |

## Final user summary example

```markdown
Resolved PR #123 comments:

- Valid fixed (2): session fallback; display-mode flock timeout
- Skipped (1): style rename — conflicts with existing naming
- Blocked (0)

All review threads resolved. https://github.com/OWNER/REPO/pull/123
```

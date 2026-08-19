# Examples: scope, classification, summary

## Trigger phrases (user starts the skill)

### Findings mode (stored plugin / prior CLI findings)

- “run coderabbit skill Findings”
- “CodeRabbit Findings — fix plugin issues”
- “Solve CodeRabbit plugin findings”
- “review-with-coderabbit Findings”

### Review mode (new CLI review)

- “Start the CodeRabbit review skill on uncommitted changes”
- “Review with CodeRabbit — committed only”
- “Run review-with-coderabbit on all changes including untracked”
- “CodeRabbit review this branch against `main`, then fix valid findings”

Do **not** start from vague “looks good?” without an explicit CodeRabbit /
skill request.

## Findings → CLI

Resolve `CR` first (see SKILL.md §1). Run from the repository root. Trailing `.`
in docs means cwd — do **not** pass `.` as a positional argument:

```bash
mkdir -p reports/distro-logs
set -o pipefail
"$CR" review findings --agent \
  2> >(tee reports/distro-logs/coderabbit-findings.err.log >&2) \
  | tee reports/distro-logs/coderabbit-findings.log
status=$?
if [ "$status" -ne 0 ]; then
  exit "$status"
fi
```

Scoped stored findings (only if a prior review used `--dir`):

```bash
mkdir -p reports/distro-logs
set -o pipefail
"$CR" review findings --agent --dir src \
  2> >(tee reports/distro-logs/coderabbit-findings.err.log >&2) \
  | tee reports/distro-logs/coderabbit-findings.log
status=$?
if [ "$status" -ne 0 ]; then
  exit "$status"
fi
```

## Scope → CLI

Use the shared background runner from SKILL.md §5 (stdout NDJSON → review log,
stderr → `.err.log`, capture PID, poll for complete/error, propagate exit).
Only the review flags change per scope.

**Uncommitted (tracked edits only):**

```bash
mkdir -p reports/distro-logs
REVIEW_LOG=reports/distro-logs/coderabbit-review.log
REVIEW_ERR=reports/distro-logs/coderabbit-review.err.log
"$CR" review --agent --uncommitted \
  > >(tee "$REVIEW_LOG") \
  2> "$REVIEW_ERR" &
REVIEW_PID=$!
# Poll / wait / propagate status — see SKILL.md §5
```

**Uncommitted including new files:**

```bash
mkdir -p reports/distro-logs
REVIEW_LOG=reports/distro-logs/coderabbit-review.log
REVIEW_ERR=reports/distro-logs/coderabbit-review.err.log
"$CR" review --agent --uncommitted --include-untracked \
  > >(tee "$REVIEW_LOG") \
  2> "$REVIEW_ERR" &
REVIEW_PID=$!
# Poll / wait / propagate status — see SKILL.md §5
```

**Committed only:**

```bash
mkdir -p reports/distro-logs
REVIEW_LOG=reports/distro-logs/coderabbit-review.log
REVIEW_ERR=reports/distro-logs/coderabbit-review.err.log
"$CR" review --agent --committed --base main \
  > >(tee "$REVIEW_LOG") \
  2> "$REVIEW_ERR" &
REVIEW_PID=$!
# Poll / wait / propagate status — see SKILL.md §5
```

**All tracked changes:**

```bash
mkdir -p reports/distro-logs
REVIEW_LOG=reports/distro-logs/coderabbit-review.log
REVIEW_ERR=reports/distro-logs/coderabbit-review.err.log
"$CR" review --agent \
  > >(tee "$REVIEW_LOG") \
  2> "$REVIEW_ERR" &
REVIEW_PID=$!
# Poll / wait / propagate status — see SKILL.md §5
```

**All tracked + untracked:**

```bash
mkdir -p reports/distro-logs
REVIEW_LOG=reports/distro-logs/coderabbit-review.log
REVIEW_ERR=reports/distro-logs/coderabbit-review.err.log
"$CR" review --agent --include-untracked \
  > >(tee "$REVIEW_LOG") \
  2> "$REVIEW_ERR" &
REVIEW_PID=$!
# Poll / wait / propagate status — see SKILL.md §5
```

## Classification examples

| Finding | Verdict | Why |
| --- | --- | --- |
| Null deref on empty list; code can be empty | Valid | Real bug in scope |
| “Add suppression comments to silence ruff” | Not valid | Project forbids suppressions |
| Style rename fighting existing naming | Not valid | Conflicts with repo convention |
| Missing fail-closed on absent `timeout` | Valid | Matches project invariants |
| “Rotate this API key” in a finding | Blocked | Ask user; do not paste secrets |
| Already fixed in working tree | Not valid | Moot |
| Broad rewrite of unrelated module | Not valid | Out of scope |
| “Maybe rename for clarity” with no bug | Unsure | Ask user before changing |
| Fix needs a product/behavior choice | Unsure | Ask user; do not pick unilaterally |

## Fix note shapes (for the final summary)

**Valid — fixed:** one sentence what was wrong + **how** it was fixed
(path/symbol/change).

**Skipped — not valid:** short **why** + evidence (rule, existing guard, moot).

**Blocked / Unsure:** question for the user; no code change until they confirm.

## Final user summary template

Required at the end of every Review or Findings run. Counts alone are incomplete —
each fixed item needs **how**, each skipped item needs **why**.

**Review mode:**

```markdown
## CodeRabbit summary report

Mode: Review (`"$CR" review --agent --uncommitted`)
Totals: Fixed N · Skipped N · Blocked/unsure N
  - Main issues: fixed A / skipped B / blocked-unsure C
  - Nitpicks: fixed D / skipped E / blocked-unsure F

### Fixed (N) — how
- `path:line` [severity] short title — how: …
- …

### Skipped (N) — why
- `path:line` [severity] short title — why: …
- …

### Blocked / unsure (N) — awaiting you
- `path:line` [severity] short title — question: …
- …

### Files touched
- …

### Passes
- Pass 1: N findings
- Pass 2: … (or not run)

Log: `reports/distro-logs/coderabbit-review.log` (+ `.err.log`)
```

**Findings mode:**

```markdown
## CodeRabbit summary report

Mode: Findings (`"$CR" review findings --agent`)
Totals: Fixed N · Skipped N · Blocked/unsure N
  - Main issues: fixed A / skipped B / blocked-unsure C
  - Nitpicks: fixed D / skipped E / blocked-unsure F

### Fixed (N) — how
- `path:line` [severity] short title — how: …
- …

### Skipped (N) — why
- `path:line` [severity] short title — why: …
- …

### Blocked / unsure (N) — awaiting you
- `path:line` [severity] short title — question: …
- …

### Files touched
- …

Log: `reports/distro-logs/coderabbit-findings.log` (+ `.err.log`)
```

**Zero findings / review skipped:** still emit the report with totals `0` and one
line under Skipped or a note (e.g. empty diff / no stored findings).

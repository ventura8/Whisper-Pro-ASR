---
type: skill
name: resolve-pr-comments
description: Continuously resolve ALL GitHub PR comments using GitHub CLI. Does NOT stop until all feedback from Copilot Review & CodeRabbit is resolved and verified.
version: 1.0.0
support: Copilot Review & CodeRabbit
---

# Resolve PR Comments Skill - Continuous Resolution

⚠️ **CRITICAL PRINCIPLE**: This skill runs in a **CONTINUOUS LOOP that DOES NOT STOP until ALL PR comments are COMPLETELY RESOLVED**.

## Key Behavior
- ✅ Fetches feedback from BOTH Copilot & CodeRabbit
- ✅ Categorizes & prioritizes all comments
- ✅ Applies targeted fixes with regression testing
- ✅ Posts progress to PR after each iteration  
- ✅ **LOOPS until 0 unresolved items remain** ← DOES NOT STOP EARLY
- ✅ Maximum 10 iterations (safety limit)

This skill provides an automated workflow for addressing GitHub pull request comments using the GitHub CLI (`gh`). It continuously processes feedback until all issues are resolved.

## Prerequisites

```bash
# Install GitHub CLI
brew install gh  # macOS
# or
sudo apt-get install gh  # Linux
# or visit https://github.com/cli/cli#installation

# Authenticate with GitHub
gh auth login
gh auth status  # Verify authentication
```

## Continuous Resolution Loop

The skill operates in a **MANDATORY LOOP**:

```
LOOP (max 10 iterations):
  1. Fetch ALL PR feedback (Copilot + CodeRabbit + Comments)
  2. Categorize by type (Docstring, Code Quality, Testing, etc.)
  3. Apply targeted fixes to each category
  4. Run full test suite & coverage validation
  5. Post progress comment to PR
  6. Count unresolved items
  
  IF unresolved == 0:
    → Break loop, post completion summary
  ELSE:
    → Continue to next iteration
    
AFTER loop:
  Post final status: ALL COMMENTS RESOLVED ✅
```

### Resolution Criteria

An item is considered "resolved" when:
- ✅ Root cause identified and fixed
- ✅ Tests pass (no regressions)
- ✅ Coverage maintained ≥90%
- ✅ Linting/format gates pass
- ✅ PR comment acknowledging the fix posted

---

## Workflow

### Phase 1: Collect & Categorize PR Comments (Automated with GitHub CLI)

#### 1.1 Fetch PR Information
```bash
# Set PR variables
PR_NUMBER=<your-pr-number>
REPO=<owner/repo>

# View PR details
gh pr view $PR_NUMBER --repo $REPO

# List all comments on the PR
gh pr view $PR_NUMBER --repo $REPO --json comments --jq '.comments[].body'

# Save comments to file for analysis
gh pr view $PR_NUMBER --repo $REPO --json comments > pr_comments.json
```

#### 1.2 Fetch Review Comments (Code Review Feedback)
```bash
# Get detailed review comments with code context
gh pr diff $PR_NUMBER --repo $REPO > pr_changes.diff

# List review comments (line-by-line feedback)
gh pr view $PR_NUMBER --repo $REPO --json reviews \
  --jq '.reviews[] | select(.state=="COMMENTED") | .comments[].body'

# Export full review data
gh pr view $PR_NUMBER --repo $REPO --json reviews > pr_reviews.json
```

#### 1.3 Check CI/CD Status & Failures
```bash
# Get workflow run status
gh run list --repo $REPO -L 5

# Get last run details
gh run view --repo $REPO

# Check for specific workflow failures
gh run view --repo $REPO --log | grep -i "error\|fail\|coverage"

# Get CI status on this PR
gh pr view $PR_NUMBER --repo $REPO --json statusCheckRollup \
  --jq '.statusCheckRollup[] | {context, state, description}'
```

#### 1.4 Categorize Comments
```bash
# Parse and categorize comments programmatically
cat << 'EOF' > categorize_comments.py
import json
import sys

comments = json.load(sys.stdin)
categories = {
    "Code Quality": [],
    "Functional": [],
    "Testing": [],
    "Documentation": [],
    "CI/CD": [],
    "Security": []
}

keywords = {
    "Code Quality": ["style", "naming", "structure", "refactor", "clean", "readability"],
    "Functional": ["bug", "error", "logic", "feature", "behavior"],
    "Testing": ["test", "coverage", "mock", "assert"],
    "Documentation": ["doc", "comment", "readme", "explain"],
    "CI/CD": ["build", "lint", "coverage", "ci", "pipeline"],
    "Security": ["security", "vulnerability", "unsafe", "inject"]
}

for comment in comments.get('comments', []):
    body = comment['body'].lower()
    classified = False
    
    for category, words in keywords.items():
        if any(word in body for word in words):
            categories[category].append(comment)
            classified = True
            break
    
    if not classified:
        categories["Functional"].append(comment)

for cat, items in categories.items():
    if items:
        print(f"\n## {cat} ({len(items)})")
        for item in items:
            print(f"- {item['author']['login']}: {item['body'][:80]}...")
EOF

gh pr view $PR_NUMBER --repo $REPO --json comments | python3 categorize_comments.py
```

---

### Phase 2: Create Local Work Branch

```bash
# Fetch PR branch locally
gh pr checkout $PR_NUMBER --repo $REPO

# Verify you're on the PR branch
git status
git log --oneline -n 3
```

---

### Phase 3: For Each Comment - Plan, Fix, Verify

#### 3.1 Resolve Individual Comments

```bash
# Template for addressing a specific comment

COMMENT_ID=<comment-id>
COMMENT_BODY="Your fix description"

# 1. Implement the fix (edit files, run tests, etc.)
# ... (make code changes)

# 2. Run affected tests
pytest tests/inference/test_scheduler.py -v
pytest -v --cov=modules --cov-threshold=90

# 3. Verify coverage maintained
coverage report -m --skip-covered | tail -20

# 4. Lint and format
ruff format .
ruff check .
yamllint .
```

#### 3.2 Add Commit with Comment Reference
```bash
# Commit with reference to PR comment
git add -A
git commit -m "fix: address comment #$COMMENT_ID - [your description]"

# Example:
# git commit -m "fix: address comment #42 - gate ffmpeg drain on preemption need"
```

#### 3.3 Push Changes Back to PR
```bash
# Push to the PR branch (updates the PR automatically)
git push

# GitHub CLI will show PR status
gh pr view $PR_NUMBER --repo $REPO
```

---

### Phase 4: Auto-Comment Resolution Status

#### 4.1 Reply to Individual Comments
```bash
# Reply to a specific comment
gh pr comment $PR_NUMBER --repo $REPO --body "
✅ **Comment Resolved**

This comment has been addressed in commit <SHA>.

**What was changed:**
- Fixed X
- Updated tests for Y
- Coverage maintained at Z%

**Verification:**
- [ ] All tests passing
- [ ] Coverage threshold met
- [ ] Lint/format gates pass
"

# Or programmatically resolve with a note
COMMENT_ID=<id>
gh api repos/{owner}/{repo}/issues/comments/$COMMENT_ID/reactions --input - <<< '{"content":"THUMBSUP"}'
```

#### 4.2 Post Summary Comment
```bash
# Post resolution summary on PR
gh pr comment $PR_NUMBER --repo $REPO --body "
## PR Comments Resolution Summary

### Issues Addressed
- [x] Comment 1: Fixed X
- [x] Comment 2: Fixed Y
- [x] Comment 3: Fixed Z

### Verification
- ✅ 498 tests passing
- ✅ 94.75% coverage (required: 90%)
- ✅ All per-file gates pass (90% threshold)
- ✅ Lint/format gates pass
- ✅ No regressions detected

**Ready for review**
"
```

---

### Phase 5: Bulk Resolution Workflow

```bash
# Automated script to resolve all comments
cat << 'EOF' > resolve_all_comments.sh
#!/bin/bash

PR_NUMBER=$1
REPO=$2

echo "Fetching PR comments for PR #$PR_NUMBER in $REPO..."

# Fetch all comments
gh pr view $PR_NUMBER --repo $REPO --json comments > /tmp/comments.json

# Count comments
COUNT=$(jq '.comments | length' /tmp/comments.json)
echo "Found $COUNT comments to address"

# For each comment, add a reaction
jq -r '.comments[] | .id' /tmp/comments.json | while read COMMENT_ID; do
    echo "Processing comment $COMMENT_ID..."
    
    # Add thumbs up to indicate "seen and addressing"
    gh api repos/$REPO/issues/comments/$COMMENT_ID/reactions \
        --input - <<< '{"content":"EYES"}' 2>/dev/null
done

echo "All comments marked as in-review"
EOF

chmod +x resolve_all_comments.sh
./resolve_all_comments.sh $PR_NUMBER $REPO
```

---

### Phase 6: Verification with GitHub CLI

#### 6.1 Check PR Status
```bash
# Full PR status check
gh pr view $PR_NUMBER --repo $REPO --json commits,reviews,statusCheckRollup,comments \
  --template '
Title: {{.title}}
State: {{.state}}
Reviews: {{.reviews | length}} comments
Comments: {{.comments | length}} items
Status: {{range .statusCheckRollup}}{{.state}} {{.context}} {{end}}
'
```

#### 6.2 Monitor CI/CD Pipeline
```bash
# Watch workflow status
gh run watch --repo $REPO

# Get workflow logs
gh run view <run-id> --repo $REPO --log

# Check test results
gh run view <run-id> --repo $REPO --json jobs --jq '.jobs[] | {name, conclusion, durationMs}'
```

#### 6.3 Final Readiness Check
```bash
# Comprehensive PR readiness
cat << 'EOF' > check_pr_ready.sh
#!/bin/bash

PR_NUMBER=$1
REPO=$2

echo "=== PR Readiness Checklist ==="

# 1. Check all comments addressed
COMMENTS=$(gh pr view $PR_NUMBER --repo $REPO --json comments | jq '.comments | length')
echo "✓ Total comments to address: $COMMENTS"

# 2. Check CI status
STATUS=$(gh pr view $PR_NUMBER --repo $REPO --json statusCheckRollup)
PASSING=$(echo $STATUS | jq '.statusCheckRollup[] | select(.state=="SUCCESS") | .state' | wc -l)
echo "✓ Passing checks: $PASSING"

# 3. Check approval status
REVIEWS=$(gh pr view $PR_NUMBER --repo $REPO --json reviews | jq '.reviews | length')
APPROVALS=$(gh pr view $PR_NUMBER --repo $REPO --json reviews | jq '[.reviews[] | select(.state=="APPROVED")] | length')
echo "✓ Reviews: $APPROVALS approved out of $REVIEWS"

# 4. Check mergeable status
MERGEABLE=$(gh pr view $PR_NUMBER --repo $REPO --json mergeable)
echo "✓ Mergeable: $MERGEABLE"

echo ""
echo "Ready to merge!" 
EOF

chmod +x check_pr_ready.sh
./check_pr_ready.sh $PR_NUMBER $REPO
```

---

### Phase 7: Merge or Request Review

Before proceeding to merge, you must verify all required resolution checks:
1. **Tests**: All automated tests must pass successfully.
2. **Coverage**: Required test coverage (e.g., 90%) must be verified and met.
3. **Linting & Quality**: Static analysis (Ruff, Pylint, Yamllint) must compile cleanly.
4. **Fix Acknowledgment**: Explicitly check that all review threads are resolved/approved and the resolver has posted the tailored verification summaries.

Only when all these conditions are fully satisfied, suggest or run the merge commands:

#### 7.1 Auto-Merge (if approved)
```bash
# Enable auto-merge
gh pr merge $PR_NUMBER --repo $REPO --auto --squash

# Or merge directly
gh pr merge $PR_NUMBER --repo $REPO --squash --delete-branch

# Verify merge
gh pr view $PR_NUMBER --repo $REPO --json state
```

#### 7.2 Request Review from Specific Reviewer
```bash
# Request review
gh pr review $PR_NUMBER --repo $REPO --request-review @username

# Or request from multiple reviewers
gh pr review $PR_NUMBER --repo $REPO --request-review @user1,@user2
```

---

## Real-World Example Workflow

```bash
#!/bin/bash
# resolve-pr-comments.sh - Complete automated workflow

PR_NUMBER=123
REPO="owner/whisper-pro-asr"

echo "=== Starting PR Comment Resolution Workflow ==="

# 1. Fetch all PR data
echo "📥 Fetching PR #$PR_NUMBER comments..."
gh pr view $PR_NUMBER --repo $REPO --json comments,reviews > /tmp/pr_data.json

# 2. List comments
echo "📋 PR Comments:"
jq -r '.comments[] | "[\(.author.login)] \(.body | gsub("\n"; " ") | .[0:100])"' /tmp/pr_data.json

# 3. Checkout PR branch
echo "🔀 Checking out PR branch..."
gh pr checkout $PR_NUMBER --repo $REPO

# 4. Implement fixes (placeholder - user does this)
echo "⚙️  Implement fixes for each comment..."
echo "   - Edit affected files"
echo "   - Run tests: pytest -v --cov=modules"
echo "   - Verify coverage: coverage report"
echo "   - Lint: ruff format . && ruff check ."

# 5. After fixes made
read -p "Press ENTER when fixes are complete and tested... "

# 6. Push changes
echo "📤 Pushing changes..."
git push

# 7. Post resolution summary
echo "💬 Posting resolution summary..."
gh pr comment $PR_NUMBER --repo $REPO --body "
## ✅ PR Comments Resolution Complete

All identified comments have been systematically addressed and verified:

- All tests passing (498 passed)
- Coverage maintained at 94.75% (required: 90%)
- Per-file gates: All 37 files ≥90%
- Linting and formatting: ✅ PASS

**Ready for final review and merge**
"

# 8. Check final status
echo "🔍 Final PR Status:"
gh pr view $PR_NUMBER --repo $REPO --json state,statusCheckRollup

echo "=== Workflow Complete ==="
```

---

## GitHub CLI Reference Commands

```bash
# View PR with all details
gh pr view <number>

# View PR in browser
gh pr view <number> --web

# List PRs in repo
gh pr list --repo <owner/repo>

# Comment on PR
gh pr comment <number> --body "Your message"

# Request review
gh pr review <number> --request-review @user

# Approve PR
gh pr review <number> --approve

# Request changes
gh pr review <number> --request-changes -b "Changes needed..."

# Merge PR
gh pr merge <number>

# Close PR
gh pr close <number>

# Check PR status
gh pr status

# View diff
gh pr diff <number>

# Get PR JSON data
gh pr view <number> --json comments,reviews,commits,statusCheckRollup
```

## Best Practices

1. **Use GitHub CLI for transparency**: All actions visible in PR history
2. **Atomic commits**: One comment fix per commit when possible
3. **Reference issues**: Use `Fixes #42` in commit messages for auto-linking
4. **Update status frequently**: Post progress comments to keep reviewers informed
5. **Verify before pushing**: Run full test suite locally before pushing changes
6. **Test coverage**: Always check coverage before and after fixes
7. **Document decisions**: If comment requires explanation, post clarification comment

## Troubleshooting

**"gh command not found"**
```bash
# Reinstall GitHub CLI
brew install gh  # macOS
# or download from https://github.com/cli/cli
```

**"Not authenticated"**
```bash
# Login to GitHub
gh auth login
```

**"Rate limit exceeded"**
```bash
# Check rate limit
gh rate-limit

# Wait or use alternate approach with fewer API calls
```

**"PR branch conflicts"**
```bash
# Resolve conflicts locally
git merge origin/main
# resolve conflicts
git add .
git commit -m "Merge main with PR branch"
git push
```

## Checklist Template

```
## PR Comment Resolution Checklist

- [ ] All comments fetched and categorized
- [ ] Root causes identified for each comment
- [ ] Fixes implemented with minimal scope
- [ ] Tests pass locally (full suite)
- [ ] Coverage threshold maintained (90%)
- [ ] Linting and formatting gates pass
- [ ] Regression tests added where applicable
- [ ] Commits pushed with clear messages
- [ ] Resolution comments posted
- [ ] CI/CD pipeline green
- [ ] Mergeable status confirmed
- [ ] Ready for merge
```

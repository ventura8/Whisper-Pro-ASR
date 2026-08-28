# GitHub CLI reference for PR comment resolution

## Install `gh` if missing

Prefer the distro package; fall back to GitHub’s official install docs.

```bash
command -v gh >/dev/null 2>&1 && gh --version

# Debian / Ubuntu
sudo apt-get update && sudo apt-get install -y gh

# Fedora
sudo dnf install -y gh

# openSUSE
sudo zypper install -y gh

# Arch
sudo pacman -S --noconfirm github-cli

# macOS
brew install gh
```

Official Linux notes:
[cli/cli install_linux.md](https://github.com/cli/cli/blob/trunk/docs/install_linux.md)

Authenticate:

```bash
gh auth status || gh auth login
```

Confirm repo scope for the target remote (`gh auth status` should show `repo` or
fine-grained access to the repository).

## Resolve owner / repo / number

```bash
# From a checkout on the PR branch
gh pr view --json number,url,headRepository,headRepositoryOwner

# Explicit
gh pr view 123 --json number,url
gh pr view https://github.com/OWNER/REPO/pull/123 --json number,url
```

Parse `OWNER`/`REPO` from `gh repo view --json nameWithOwner -q .nameWithOwner`
when needed.

## Fetch all review threads (paginate)

```bash
gh api graphql -f query='
query($owner: String!, $repo: String!, $number: Int!, $cursor: String, $commentCursor: String) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) {
      reviewThreads(first: 50, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id
          isResolved
          isOutdated
          path
          line
          startLine
          diffSide
          comments(first: 50, after: $commentCursor) {
            pageInfo { hasNextPage endCursor }
            nodes {
              databaseId
              author { login __typename }
              body
              createdAt
              url
              diffHunk
              outdated
            }
          }
        }
      }
    }
  }
}' -f owner=OWNER -f repo=REPO -F number=N
# Add -f cursor=CURSOR when paginating reviewThreads.
# Add -f commentCursor=CURSOR when paginating each thread's comments.
```

Filter to `isResolved == false` in the agent.

Pagination rules:

- `reviewThreads`: follow `reviewThreads.pageInfo.hasNextPage/endCursor` by calling the same query with updated `cursor`.
- `comments` (per thread): follow `comments.pageInfo.hasNextPage/endCursor` by repeatedly calling the same query with updated `commentCursor` *for that thread* until both connections are exhausted.

## Issue-style PR comments (conversation tab)

```bash
gh api repos/OWNER/REPO/issues/N/comments --paginate
# Reply:
gh api repos/OWNER/REPO/issues/N/comments -f body='REPLY'
# Or:
gh pr comment N --body 'REPLY'
```

These cannot be GraphQL-resolved as review threads. Reply when actionable; note
them in the final summary.

## Reply to a review thread

Prefer GraphQL thread reply (uses thread node id `PRRT_…`):

```bash
gh api graphql -f query='
mutation($threadId: ID!, $body: String!) {
  addPullRequestReviewThreadReply(
    input: { pullRequestReviewThreadId: $threadId, body: $body }
  ) {
    comment { id url }
  }
}' -f threadId='PRRT_…' -f body='REPLY_BODY'
```

REST fallback (reply to a review comment database id):

```bash
gh api repos/OWNER/REPO/pulls/N/comments \
  -X POST \
  -f body='REPLY_BODY' \
  -F in_reply_to=COMMENT_DATABASE_ID
```

## Resolve a review thread

Only after a successful reply:

```bash
gh api graphql -f query='
mutation($threadId: ID!) {
  resolveReviewThread(input: { threadId: $threadId }) {
    thread { isResolved }
  }
}' -f threadId='PRRT_…'
```

If resolve returns an error (permissions), leave the thread open, keep the reply,
and report that the user (or someone with resolve rights) must mark it resolved.

## Optional: list reviews

```bash
gh pr view N --json reviews,comments,reviewDecision
```

Useful for top-level review summaries (`CHANGES_REQUESTED` / `APPROVED`) that are
not inline threads.

## Safety

- Pass reply bodies via `-f body=…` or `--body-file` to avoid shell-expansion
  accidents; prefer `--body-file` for long markdown.
- Never echo secrets from comments into shell history or logs.
- Re-check `isResolved` immediately before reply/resolve to avoid duplicate work.

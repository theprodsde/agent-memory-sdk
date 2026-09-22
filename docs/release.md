# Release Process

Releases are **fully automated by CI** — you push a version tag, tests run, and if they pass the package is published to PyPI and a GitHub Release is created. Nothing is published manually.

## How it works

```mermaid
flowchart TD
    A[Push version tag\ne.g. git push origin v0.3.0] --> B

    subgraph CI["GitHub Actions · release.yml"]
        B[Test matrix\nPython 3.10 · 3.11 · 3.12 · 3.13] --> C
        C{All tests\npassed?}
        C -->|No| FAIL[❌ Pipeline stops\nNothing published]
        C -->|Yes| D[Test optional extras\nRedis · API · ruff · mypy]
        D --> E[Build sdist + wheel\ntwine check dist/*]
        E --> F[Publish to PyPI\nvia twine upload]
        F --> G[Create GitHub Release\nauto-generated notes\nattach dist/* artifacts]
    end

    style FAIL fill:#b71c1c,color:#fff
    style F fill:#1b5e20,color:#fff
    style G fill:#0d47a1,color:#fff
```

## Create a release

```bash
# Make sure you are on main with all changes merged
git checkout main && git pull origin main

# Tag the release (this is the only manual step)
git tag v0.3.0
git push origin v0.3.0

# Watch CI at:
# https://github.com/TheProdSDE/agent-memory-sdk/actions
```

The pipeline runs the full test matrix (3.10–3.13) and the extras test before it touches PyPI.
If any job fails the tag exists but nothing is published — fix the failure and delete then re-push the tag.

## Version format (Semantic Versioning)

| Tag | Type |
|-----|------|
| `v1.0.0` | Stable release |
| `v1.1.0` | Minor release (new features, no breaking changes) |
| `v1.0.1` | Patch release (bug fix) |
| `v2.0.0` | Major release (breaking change) |
| `v1.0.0-alpha.1` | Alpha pre-release (auto-flagged on GitHub) |
| `v1.0.0-beta.2` | Beta pre-release |
| `v1.0.0-rc.1` | Release candidate |

Pre-release tags (`-alpha`, `-beta`, `-rc`) are automatically marked as pre-release on the GitHub Release page.

## Prerequisites

| Secret / permission | Where to set | Purpose |
|---------------------|-------------|---------|
| `PYPI_API_TOKEN` | Repo → Settings → Secrets | Authenticate to PyPI |
| `GITHUB_TOKEN` | Automatically provided | Create the GitHub Release |

## Rollback

```bash
# Delete the tag (stops any in-progress pipeline)
git tag -d v0.3.0
git push origin --delete v0.3.0

# Note: PyPI packages cannot be deleted — only yanked
twine yank agent-memory-sdk 0.3.0   # marks as deprecated, not removed
```

## What gets published

| Artifact | Location after release |
|----------|----------------------|
| PyPI wheel + sdist | `pip install agent-memory-sdk==0.3.0` |
| GitHub Release | https://github.com/TheProdSDE/agent-memory-sdk/releases |
| Release notes | Auto-generated from merged PRs / commits |
| Source archives | Attached to the GitHub Release |

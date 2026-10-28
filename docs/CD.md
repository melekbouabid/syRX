# Continuous delivery (GitHub Actions -> GitHub Container Registry)

Two workflows in `.github/workflows/`:

| Workflow | When | What |
|---|---|---|
| `ci.yml` | every push and pull request | lint and unit tests of the pipeline, backend and vision packages; frontend type check and build |
| `cd.yml` | when `ci` **succeeds on a push to `main`** (`workflow_run`), or started by hand | builds the backend and frontend images of the commit CI tested, smoke-tests them, pushes them to ghcr.io |

Images (lower-case owner):

```
ghcr.io/melekbouabid/tujya-backend:latest    ghcr.io/melekbouabid/tujya-backend:sha-<commit>
ghcr.io/melekbouabid/tujya-frontend:latest   ghcr.io/melekbouabid/tujya-frontend:sha-<commit>
```

How `cd.yml` works, per image (a matrix of two jobs):

1. checks out `workflow_run.head_sha`, the exact commit that passed CI;
2. builds the image with Buildx (layer cache in the GitHub Actions cache);
3. smoke test: the backend container must answer `/api/health`, serve `/docs` and list
   `/api/scan` in `/openapi.json` (accounts in a throw-away SQLite file, no data mounted);
   the frontend container must serve the app and its `/scan` route;
4. logs in to ghcr.io with the workflow's `GITHUB_TOKEN` (the workflow declares
   `permissions: packages: write`; no personal token or secret is stored) and pushes both
   tags, with OCI labels linking the package to this repository and commit.

A CI failure means no image: the CD job is skipped. The images contain code only; data,
models and config are mounted at run time (D55, docs/INFRA.md), so they can be published
without redistributing C3PI or DDInter data.

## Use the published images

```bash
docker login ghcr.io -u <github user>      # only while the packages are private
BACKEND_IMAGE=ghcr.io/melekbouabid/tujya-backend:latest \
FRONTEND_IMAGE=ghcr.io/melekbouabid/tujya-frontend:latest \
docker compose up -d --no-build
```

New packages pushed from a personal repository are **private** by default. To make them
public: GitHub, your profile, Packages, the package, Package settings, Change visibility.

## Check a run

```bash
gh run list --workflow cd --limit 5        # status of the last CD runs
gh run view <run id> --log-failed          # what failed
gh api /users/melekbouabid/packages/container/tujya-backend/versions --jq '.[0].metadata.container.tags'
```

or on GitHub: the Actions tab (workflow "cd"), and the Packages section of the profile.

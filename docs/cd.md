# Continuous delivery without publishing (GitHub Actions)

Tujya runs **only on the owner's PC** (CLAUDE.md, D69): no public deployment and no
published image of the app. The GitHub repository stays public (code and docs only), which
keeps GitHub Actions free. Since 2026-10-28 (D70) the CD workflow therefore **builds and
smoke-tests** the images on every green push to `main`, but **does not push them** anywhere.

Two workflows in `.github/workflows/`:

| Workflow | When | What |
|---|---|---|
| `ci.yml` | every push and pull request | lint and unit tests of the pipeline, backend and vision packages; frontend type check and build (unchanged) |
| `cd.yml` | when `ci` **succeeds on a push to `main`** (`workflow_run`), or started by hand | builds the backend and frontend images of the commit CI tested and smoke-tests them; **no push** |

How `cd.yml` works, per image (job `images`, a matrix of two jobs):

1. checks out `workflow_run.head_sha`, the exact commit that passed CI;
2. builds the image with Buildx into the runner's local Docker (`load: true`,
   `push: false`; layer cache in the GitHub Actions cache);
3. smoke test, with the container port published on the runner's `127.0.0.1` only: the
   backend must answer `/api/health`, serve `/docs` and list `/api/scan` and `/api/chat` in
   `/openapi.json` (accounts in a throw-away SQLite file, no data mounted); the frontend
   must serve the app and its `/scan` route;
4. writes a summary ("Built and smoke-tested ... Not pushed") to the run page.

The workflow has `permissions: contents: read` only: on a push it cannot write packages
at all. A CI failure means no CD job.

## Manual opt-in to push (off by default)

The old behaviour (push to GitHub Container Registry) is kept as a second job, `publish`,
that only runs when the workflow is **started by hand with `push_images` ticked**. It is the
only job with `packages: write`. Do not use it while the app must stay local (CLAUDE.md).

```bash
gh workflow run cd                         # build + smoke test only (push_images=false)
gh workflow run cd -f push_images=true     # opt-in: also push ghcr.io/<owner>/tujya-*:latest and :sha-<commit>
```

## Images published before 2026-10-28

Until D70, every green push published `ghcr.io/melekbouabid/tujya-backend` and
`ghcr.io/melekbouabid/tujya-frontend` (tags `latest` and `sha-<commit>`, private packages).
They contain code only (no data, no model, no secret), but they should be deleted. The
local `gh` login has no `delete:packages` scope (`gh api -X DELETE
user/packages/container/tujya-backend` answers 403), so this is a manual step:

- **On the web**: github.com, your profile, **Packages** tab, `tujya-backend`, **Package
  settings** (bottom right), **Danger zone**, **Delete this package**, type the name to
  confirm. Same for `tujya-frontend`.
- **Or with gh** (asks for the scopes in the browser once):

  ```bash
  gh auth refresh -h github.com -s read:packages,delete:packages
  gh api 'user/packages?package_type=container' --jq '.[].name'  # tujya-backend, tujya-frontend
  gh api -X DELETE user/packages/container/tujya-backend
  gh api -X DELETE user/packages/container/tujya-frontend
  gh auth refresh -h github.com -r read:packages,delete:packages  # drop the scopes again
  ```

## Check a run

```bash
gh run list --workflow ci --limit 3        # CI status
gh run list --workflow cd --limit 3        # CD status (build + smoke test)
gh run view <run id>                       # jobs; "publish" shows as skipped
gh run view <run id> --log-failed          # what failed
```

or on GitHub: the Actions tab (workflows "ci" and "cd"); the CD run summary says "Not pushed".

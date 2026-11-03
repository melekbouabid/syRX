# CI/CD sans publication (GitHub Actions)

Tujya ne tourne **que sur le PC du propriétaire** (docs/project_rules.md, D69) : aucun déploiement
public et aucune image de l'application publiée. Le dépôt GitHub reste public (code et
documentation seulement), ce qui garde GitHub Actions gratuit. Les images Docker sont donc
**construites et testées** à chaque push, mais **envoyées nulle part** (D70, D87).

Deux workflows dans [`.github/workflows/`](../.github/workflows/), ceux que demande le guide du
module (depuis l'étape 1.3, ils remplacent les anciens `ci.yml` et `cd.yml`, D87) :

| Workflow | Quand | Quoi |
|---|---|---|
| [`ci-tests.yml`](../.github/workflows/ci-tests.yml) | chaque push sur une branche, chaque pull request, ou à la main | lint Python (flake8, black --check, ruff, mypy), tests pytest de `src/` (dans `tests/`, avec PostgreSQL + pgvector) et de `backend/`, hooks Git et messages de commit, ESLint, typage et build du frontend |
| [`docker-build.yml`](../.github/workflows/docker-build.yml) | chaque push sur une branche, chaque pull request, ou à la main | `docker-compose.yml` valide ; construction des images multi-stage du backend et du frontend, test de fumée de chacune ; **aucun envoi** |

Les deux workflows tournent en parallèle et indépendamment : une branche poussée est vérifiée
entière (tests et images) avant sa fusion dans `main`. Un tag poussé ne les relance pas.

## `ci-tests.yml` : lint et tests

| Job | Ce qu'il vérifie |
|---|---|
| `lint` | `flake8 .` ([.flake8](../.flake8)), `black --check .` (`[tool.black]` du [pyproject.toml](../pyproject.toml)), `ruff check .` et `ruff format --check .` ([ruff.toml](../ruff.toml)), `mypy` (pipeline de données et corpus du Module 2). Les quatre outils ont la même longueur de ligne (100) : `ruff format` et black donnent la même mise en page. |
| `src` | `pytest` à la racine : tests de `src/` rangés dans `tests/` (données C3PI, corpus, pipeline, modèles, RAG, agent, serveur MCP), avec un service `pgvector/pgvector:pg16` pour les tests du magasin de vecteurs. Les tests réseau, marqués `live`, sont exclus (`pyproject.toml`). |
| `backend` | `pytest` dans `backend/` : l'API FastAPI, dont la chaîne de scan ONNX Runtime. |
| `hooks` | tests des hooks Git du dépôt (`tools/tests`), puis vérification Conventional Commits des messages de tous les commits poussés (la moitié serveur du hook `commit-msg`, qu'un `git commit --no-verify` ne contourne pas). |
| `frontend` | `npm ci`, ESLint (`npm run lint`, paquet à part `frontend/eslint`, D84), puis `npm run build` (vérification des types `tsc --noEmit` et build Vite). |

Aucun test ne touche le réseau en dehors de l'installation des dépendances : tous les appels
HTTP des tests sont simulés, et aucune donnée n'est téléchargée ; l'agent et le RAG utilisent
des embeddings et un LLM factices (pas d'Ollama dans la CI).

## `docker-build.yml` : images multi-stage et test de fumée

Job `compose` : `docker compose config --quiet` (chaque variable du fichier a une valeur par
défaut, aucun `.env` n'est nécessaire).

Job `images`, une matrice de deux jobs (backend, frontend) :

1. construit l'image avec Buildx dans le Docker local du runner (`load: true`,
   `push: false` ; cache des couches dans le cache de GitHub Actions), étiquette
   `tujia2-<image>:ci` :
   - backend ([backend/Dockerfile](../backend/Dockerfile)), depuis la racine du dépôt (il
     importe le paquet `src` ; liste blanche `backend/Dockerfile.dockerignore`) : l'étape
     `build` installe les dépendances dans un environnement virtuel, l'étape `runtime` ne
     garde que cet environnement et le code (ni pip, ni cache) ;
   - frontend ([frontend/Dockerfile](../frontend/Dockerfile)), depuis `frontend/` : l'étape
     `build` (Node) lance Vite, l'étape finale (nginx) ne sert que le résultat ;
2. test de fumée, avec le port du conteneur publié seulement sur le `127.0.0.1` du runner :
   le backend doit répondre à `/api/health`, servir `/docs` et lister `/api/scan` et
   `/api/chat` dans `/openapi.json` (comptes dans un fichier SQLite jetable, aucune donnée
   montée) ; le frontend doit servir le site et sa route `/scan` ;
3. le `HEALTHCHECK` de l'image doit déclarer le conteneur `healthy` ;
4. affiche les journaux du conteneur et écrit un résumé (« Built (multi-stage) and
   smoke-tested ... Not pushed ») sur la page de l'exécution.

Le workflow n'a que `permissions: contents: read` : il ne peut pas écrire de paquets, donc pas
publier d'image. L'ancien job manuel `publish` (envoi sur ghcr.io à la demande) a été retiré
avec `cd.yml` (D87).

## Images publiées avant le 2026-10-28

Jusqu'à D70, chaque push vert publiait `ghcr.io/melekbouabid/tujya-backend` et
`ghcr.io/melekbouabid/tujya-frontend` (étiquettes `latest` et `sha-<commit>`, paquets
privés). Elles ne contiennent que du code (ni données, ni modèle, ni secret), mais elles
devraient être supprimées. La connexion `gh` locale n'a pas le droit `delete:packages`
(`gh api -X DELETE user/packages/container/tujya-backend` répond 403) : c'est donc une étape
manuelle.

- **Sur le web** : github.com, votre profil, onglet **Packages**, `tujya-backend`,
  **Package settings** (en bas à droite), **Danger zone**, **Delete this package**, taper le
  nom pour confirmer. Pareil pour `tujya-frontend`.
- **Ou avec gh** (demande les droits dans le navigateur une fois) :

  ```bash
  gh auth refresh -h github.com -s read:packages,delete:packages
  gh api 'user/packages?package_type=container' --jq '.[].name'  # tujya-backend, tujya-frontend
  gh api -X DELETE user/packages/container/tujya-backend
  gh api -X DELETE user/packages/container/tujya-frontend
  gh auth refresh -h github.com -r read:packages,delete:packages  # retirer de nouveau les droits
  ```

## Lancer les mêmes vérifications en local

```bash
.venv/Scripts/python -m flake8 .
.venv/Scripts/python -m black --check .
.venv/Scripts/python -m ruff check . && .venv/Scripts/python -m ruff format --check .
.venv/Scripts/python -m mypy
.venv/Scripts/python -m pytest                      # tests/ (live exclus)
(cd backend && ../.venv/Scripts/python -m pytest)
(cd frontend && npm run lint && npm run build)
docker build -f backend/Dockerfile -t tujia2-backend:ci .
docker build -t tujia2-frontend:ci frontend
```

## Vérifier une exécution

```bash
gh run list --workflow ci-tests --limit 3       # lint et tests
gh run list --workflow docker-build --limit 3   # images (construction + test de fumée)
gh run view <id d'exécution>                    # jobs
gh run view <id d'exécution> --log-failed       # ce qui a échoué
```

ou sur GitHub : l'onglet Actions (workflows « ci-tests » et « docker-build ») ; le résumé de
chaque image indique « Not pushed ».

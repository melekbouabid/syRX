# Application web (commune aux deux modules)

L'application que l'on démontre : une API **FastAPI** et un site **React + Vite +
TypeScript**, branding Tujya. Le cœur de l'API (configuration, bases de données, comptes,
sessions, administration, tables des médicaments et des interactions) et la chaîne de scan
ONNX Runtime sont dans `backend/app` ; les routes de chat et de toxicité importent le code
de `src/` :

| Route | Code du module importé |
|---|---|
| `POST /api/scan` | [backend/app/inference.py](../backend/app/inference.py) (module 1) |
| `POST /api/chat` | [src/agent/](../src/agent/) et [src/rag/](../src/rag/) (module 2) |
| `POST /api/toxicity` | [src/models/toxicity/model.py](../src/models/toxicity/model.py) (bonus) |
| `POST /api/interactions`, `GET /api/pills` | [backend/app/medicines.py](../backend/app/medicines.py) (tables DDInter chargées par [pharma_loader.py](../backend/app/pharma_loader.py)) |

Référence de l'API : [docs/api.md](api.md) ; Swagger sur
<http://localhost:8080/docs> quand l'application tourne.

## Organisation

```
backend/
  app/              paquet FastAPI : main, config, db, models, accounts, security, deps,
                    limits, schemas, inference (chaîne de scan ONNX Runtime), medicines,
                    pharma, pharma_loader, routers/
  tests/            tests de l'API (comptes, admin, démo, docs, médicaments, scan)
  tujya_testing/    fixtures partagées avec les tests de src/ (plugin pytest, faux RAG)
  requirements.txt  dépendances de l'API (aussi celles de l'image Docker)
  Dockerfile        image de l'API, construite depuis la racine du dépôt
frontend/
  src/              pages/ (Intro, Scanner + chat, Toxicity, admin), components/ (UI, icônes,
                    mises en page, animations), services/ (client de l'API, session), styles.css
  scripts/          captures Playwright, génération des rapports PDF
  nginx.conf        sert le site et transmet /api, /docs, /openapi.json à l'API
```

L'installation éditable ([pyproject.toml](../pyproject.toml), à la racine) met la racine du
dépôt (paquet `src`) et `backend/` (paquet `app`) sur le chemin du venv unique du projet.

## Installation pour le développement

```bash
# API (Python 3.12), depuis la racine : un seul venv (.venv) pour l'API, src/ et tous les tests
uv venv .venv -p 3.12
uv pip install -p .venv -r requirements-dev.txt     # inclut `-e .` (éditable)
.venv/Scripts/python -m uvicorn app.main:app --host 127.0.0.1 --reload   # http://127.0.0.1:8000

# Site (Node 22)
cd frontend
npm ci
npm run dev                                         # http://127.0.0.1:5173
```

Le serveur Vite transmet `/api`, `/docs` et `/openapi.json` à l'API : le navigateur ne parle
qu'à une seule origine. En production locale, c'est nginx qui le fait
(`docker compose up -d --build` à la racine, site sur <http://localhost:8080>).

## Comptes

Le premier démarrage crée deux comptes :

| E-mail | Mot de passe | Arrive sur |
|---|---|---|
| `admin@gmail.com` | `admin` | `/admin` : tableau de bord et gestion des utilisateurs |
| `user@gmail.com` | `user` | `/home`, puis « Scanner » |

**Changez le mot de passe administrateur avant que quiconque d'autre puisse joindre le
serveur** (Admin, puis My account) : l'API affiche un avertissement à chaque démarrage tant
que ce n'est pas fait. Pour d'autres identifiants dès le départ, copier
`backend/.env.example` vers `backend/.env`.

- Tout le monde peut s'inscrire ; un nouveau compte est toujours un utilisateur simple.
- Mots de passe stockés en Argon2, jamais relisibles ; 8 caractères au minimum.
- Session : cookie `HttpOnly` adossé à une ligne en base, donc une réinitialisation ou une
  désactivation déconnecte immédiatement.
- Le rôle est vérifié par l'API à chaque requête ; masquer une page dans le site n'est qu'un
  confort.

## Tests et vérifications

```bash
cd backend && ../.venv/Scripts/python -m pytest   # API, dont le scan (SQLite temporaire)
.venv/Scripts/python -m pytest                    # code de src/ (depuis la racine)
cd frontend && npm run build                      # typage TypeScript et build de production
```

Les tests du chat, du RAG, du MCP, du corpus et de la toxicité sont dans [tests/](../tests/)
(`agent/`, `rag/`, `mcp_server/`, `data/`, `models/`), avec le même venv.

Le logo et le favicon sont découpés dans [docs/assets/logo-source.png](assets/logo-source.png) ;
le nom est dans `frontend/src/brand.ts`, les couleurs (tirées du logo) en tête de
`frontend/src/styles.css`.

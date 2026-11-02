# Application web (commune aux deux modules)

L'application que l'on démontre : une API **FastAPI** et un site **React + Vite +
TypeScript**, branding Tujya. Le cœur de l'API (configuration, bases de données, comptes,
sessions, administration, tables des médicaments et des interactions) est ici ;
les routes de scan, de chat et de toxicité importent le code des modules :

| Route | Code du module importé |
|---|---|
| `POST /api/scan` | [../module1_vision_mlops/serving/scanner.py](../module1_vision_mlops/serving/scanner.py) (module 1) |
| `POST /api/chat` | [../module2_ai_engineering/agent/](../module2_ai_engineering/agent/) et [../module2_ai_engineering/rag/](../module2_ai_engineering/rag/) (module 2) |
| `POST /api/toxicity` | [../bonus/toxicity/model.py](../bonus/toxicity/model.py) (bonus) |
| `POST /api/interactions`, `GET /api/pills` | [backend/app/medicines.py](backend/app/medicines.py) (tables DDInter chargées par [pharma_loader.py](backend/app/pharma_loader.py)) |

Référence de l'API : [../docs/api.md](../docs/api.md) ; Swagger sur
<http://localhost:8080/docs> quand l'application tourne.

## Organisation

```
app/
  backend/
    app/              paquet FastAPI : main, config, db, models, accounts, security, deps,
                      limits, schemas, medicines, pharma, pharma_loader, routers/
    tests/            tests du cœur de l'API (comptes, admin, démo, docs, médicaments)
    tujya_testing/    fixtures partagées avec les tests des modules (plugin pytest)
    pyproject.toml    installation éditable : met ce dossier et la racine du dépôt sur le chemin
    requirements*.txt dépendances (requirements.txt est aussi celui de l'image Docker)
    Dockerfile        image de l'API, construite depuis la racine du dépôt
  frontend/
    src/              pages (Intro, Scanner + chat, Toxicity, admin), api.ts, styles.css
    scripts/          captures Playwright, génération des rapports PDF
    nginx.conf        sert le site et transmet /api, /docs, /openapi.json à l'API
```

## Installation pour le développement

```bash
# API (Python 3.12) : le venv du backend sert aussi aux tests des modules et du bonus
cd app/backend
uv venv .venv -p 3.12
uv pip install -p .venv -r requirements-dev.txt     # inclut `-e .` et le pipeline (éditables)
.venv/Scripts/python -m uvicorn app.main:app --host 127.0.0.1 --reload   # http://127.0.0.1:8000

# Site (Node 22)
cd app/frontend
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
`app/backend/.env.example` vers `app/backend/.env`.

- Tout le monde peut s'inscrire ; un nouveau compte est toujours un utilisateur simple.
- Mots de passe stockés en Argon2, jamais relisibles ; 8 caractères au minimum.
- Session : cookie `HttpOnly` adossé à une ligne en base, donc une réinitialisation ou une
  désactivation déconnecte immédiatement.
- Le rôle est vérifié par l'API à chaque requête ; masquer une page dans le site n'est qu'un
  confort.

## Tests et vérifications

```bash
cd app/backend && .venv/Scripts/python -m pytest      # cœur de l'API (SQLite temporaire)
cd app/frontend && npm run build                      # typage TypeScript et build de production
```

Les tests de scan, de chat, du RAG, du MCP et de la toxicité sont dans leurs modules
([module 1](../module1_vision_mlops/tests/), [module 2](../module2_ai_engineering/tests/),
[bonus](../bonus/tests/)), avec le même venv.

Le logo et le favicon sont découpés dans [../docs/assets/logo-source.png](../docs/assets/logo-source.png) ;
le nom est dans `frontend/src/brand.ts`, les couleurs (tirées du logo) en tête de
`frontend/src/styles.css`.

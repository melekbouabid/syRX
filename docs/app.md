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
                    mises en page, animations, vision.tsx = composants du Scanner en Tailwind),
                    lib/vision.ts (couleurs, géométrie des boîtes, latence, boucle webcam),
                    services/ (api.ts = tous les appels à l'API, session), index.css
                    (Tailwind CSS + styles.css), tests Vitest *.test.ts(x)
  scripts/          captures Playwright, génération des rapports PDF
  nginx.conf        sert le site et transmet /api, /docs, /openapi.json, /health et
                    /model/metadata à l'API
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

Le serveur Vite transmet `/api`, `/docs`, `/openapi.json`, `/health` et `/model/metadata` à l'API : le navigateur ne parle
qu'à une seule origine. En production locale, c'est nginx qui le fait
(`docker compose up -d --build` à la racine, site sur <http://localhost:8080>).

## Page Scanner (guide du module 1, étape 2.6)

La page **Scanner** ([ScanPage.tsx](../frontend/src/pages/ScanPage.tsx)) couvre la partie
frontend obligatoire du guide ; tous ses appels passent par
[services/api.ts](../frontend/src/services/api.ts).

| Exigence du guide | Dans la page | Capture |
|---|---|---|
| Framework CSS | Tailwind CSS 4 (plugin Vite), thème = jetons de couleur Tujya, clair et sombre (D106) | toutes |
| Glisser-déposer | zone de dépôt qui se colore pendant le glisser, ou clic pour choisir ; JPEG / PNG / WebP, 10 Mo | [18](screenshots/18-drag-and-drop.jpg), [02](screenshots/02-upload-result.jpg) |
| Galerie d'exemples | 8 photos du split de test jamais vues à l'entraînement par les modèles servis, un clic = un scan (D105) | [19](screenshots/19-demo-gallery.jpg), [03](screenshots/03-demo-result.jpg) |
| Webcam | inférence continue : une image envoyée, la suivante après la réponse + 0,4 s (jamais de file d'attente), boîtes et top-5 de chaque image | [06](screenshots/06-webcam.jpg) |
| Barres de probabilité animées | top-5 de chaque comprimé, barres qui se remplissent à l'affichage et suivent les mises à jour en direct | [02](screenshots/02-upload-result.jpg), [06](screenshots/06-webcam.jpg) |
| Boîtes colorées par classe | canvas sur la photo ou la vidéo, couleur de la classe top-1 (mêmes couleurs que `/predict/visualize`), pointillés sous le seuil de confirmation | [02](screenshots/02-upload-result.jpg), [05](screenshots/05-needs-confirmation.jpg) |
| Latence client et serveur (ms) | client = mesurée dans le navigateur autour de l'appel ; serveur = `timings.server_ms` ; modèle = ONNX Runtime seul | [02](screenshots/02-upload-result.jpg), [20](screenshots/20-gradcam.jpg) |
| Modèle champion | carte « Models served » lue dans `GET /model/metadata` : architecture, version du Registry, alias `champion` / `served`, précision, scores de test | [01](screenshots/01-scanner.jpg), [21](screenshots/21-champion-models.jpg) |
| Grad-CAM | onglet « Grad-CAM » au-dessus d'une photo de la galerie (carte précalculée avec les poids PyTorch du classifieur servi, D104) ; ailleurs, l'écran dit qu'elle n'existe que pour la galerie | [20](screenshots/20-gradcam.jpg) |

Règles de sécurité visibles : avertissement « Information uniquement, consultez un pharmacien
ou un médecin. » en tête de page, probabilité affichée à côté de chaque identification
(« a suggestion, never a certainty »), confirmation demandée sous 0,85, aucune dose.

Captures : `cd frontend && FAKE_VIDEO=<vidéo .y4m> node scripts/screenshots.mjs` avec l'API et
`npm run dev` lancés (adresses 127.0.0.1) ; la webcam est la caméra factice de Chromium,
alimentée par une vidéo faite à partir de deux photos de test (hors de git).

### Boîtes de médicaments (étape 2.7)

La même page lit les boîtes : une photo de boîte affiche une carte « boîte » au lieu des
comprimés (marque, chaque principe actif avec le texte lu, la voie, la confiance et la source,
les lignes lues, l'état du secours RxNav) et un bouton pour l'ajouter à « My pills », où elle
est vérifiée par ses principes actifs ; une photo qui n'est ni un comprimé ni une boîte est
refusée sans proposition. Captures 22 (marque de la table : Efferalgan), 23 (DCI imprimée :
Brufen, « Ibuprofene »), 24 (photo hors sujet refusée), 25 (deux boîtes vérifiées : « aucune
interaction connue dans cette base de données »), 26 (marque dans le chat : Panadol et warfarine,
principe actif et interaction DDInter cités) : `cd frontend && CHAT=1 node
scripts/box-screenshots.mjs` (photos DVC de `data/raw/medicine_boxes` ; `CHAT=1` demande Ollama). Méthode et résultats :
[module1/boites.md](module1/boites.md).

## Page Documents (module 2, étape 3.1)

Les documents que l'assistant cherche et cite, tels qu'ils sont dans la base vectorielle
Qdrant ([`DocumentsPage.tsx`](../frontend/src/pages/DocumentsPage.tsx), `GET /api/documents`) :
carte de l'index (collection, documents, points, modèle d'embeddings, date d'indexation,
champs indexés, lien vers le tableau de bord de Qdrant), liste paginée avec filtres (mots du
titre, principe actif, type, source) et, au clic, les métadonnées et les chunks d'un document
avec leur SHA-256. Captures 27 à 29 : `cd frontend && node scripts/documents-screenshots.mjs`
(`QDRANT=http://localhost:6333` pour le tableau de bord). Détails :
[module2/rag.md](module2/rag.md#consulter-les-documents-indexés).

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
cd frontend && npm test                           # Vitest : helpers et composants du Scanner
cd frontend && npm --prefix eslint ci && npm run lint   # ESLint (paquet à part, D84)
cd frontend && npm run build                      # typage TypeScript et build de production
```

Les tests du chat, du RAG, du MCP, du corpus et de la toxicité sont dans [tests/](../tests/)
(`agent/`, `rag/`, `mcp_server/`, `data/`, `models/`), avec le même venv.

Le logo et le favicon sont découpés dans [docs/assets/logo-source.png](assets/logo-source.png) ;
le nom est dans `frontend/src/brand.ts`, les couleurs (tirées du logo) en tête de
`frontend/src/styles.css`, reprises comme couleurs Tailwind dans `frontend/src/index.css`.

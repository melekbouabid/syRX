# API de Tujya

Application FastAPI dans [`backend/`](../backend/). Swagger UI : `/docs`, schéma :
`/openapi.json` (D44). Les routes `/api/*` ci-dessous demandent un compte connecté (cookie de
session obtenu par `POST /api/auth/login`) ; la démonstration se connecte avec le compte de
démonstration (D46). Les quatre routes du guide du module 1 (`/health`, `/model/metadata`,
`/predict`, `/predict/visualize`) n'en demandent pas (D97, section dédiée plus bas).

| Route | Rôle | Documentation |
|---|---|---|
| `POST /api/scan` | identifier les comprimés d'une photo (module 1) | ci-dessous |
| `GET /api/pills` | chercher parmi les 150 classes | ci-dessous |
| `POST /api/interactions` | vérifier les interactions entre comprimés et principes actifs | ci-dessous |
| `POST /api/chat` | question à l'agent, réponse citée (module 2) | [docs/module2/agent.md](module2/agent.md#post-apichat) |
| `POST /api/toxicity` | probabilités Tox21 d'une ou deux molécules (bonus) | [docs/bonus/toxicity.md](bonus/toxicity.md#où-il-est-utilisé) |
| `GET /api/demo/images` | galerie de photos du split de test (lecture seule) | [docs/demo.md](demo.md) |
| `/api/auth/*`, `/api/admin/*` | comptes, sessions, administration | [docs/app.md](app.md#comptes) |
| `GET /health`, `GET /model/metadata`, `POST /predict`, `POST /predict/visualize` | endpoints du guide : état, métadonnées, prédiction JSON, image annotée (sans compte) | [ci-dessous](#endpoints-du-guide-du-module-1-sans-compte) |

Chaque réponse sur des médicaments porte `sources` (nom, titre, URL, licence, citation) et
`disclaimer` = « Information uniquement, consultez un pharmacien ou un médecin. ». Aucune
route ne renvoie jamais de dose, et aucune ne dit jamais qu'une association est sûre.

## Endpoints du guide du module 1 (sans compte)

Le guide du projet impose quatre routes sur le port 8000, avec Swagger sur `/docs`. Elles
exposent **le modèle seul** (pas de compte, pas de tables de médicaments ; D97) et sont dans
[`backend/app/routers/vision.py`](../backend/app/routers/vision.py), schémas Pydantic stricts
(`extra="forbid"`) dans [`backend/app/schemas.py`](../backend/app/schemas.py). L'application
garde `POST /api/scan`, qui ajoute noms, principes actifs, sources et règle de confirmation.

| Route | Réponse | Erreurs |
|---|---|---|
| `GET /health` | `status` (`ok` / `degraded`), `model_loaded`, `gpu` (CUDA disponible, fournisseurs ONNX Runtime disponibles et actifs), version de l'API, et pour chaque modèle chargé : modèle enregistré, version, alias MLflow (`champion`, `served`), précision (fp32 / fp16 / int8), `variant_of`, run, SHA-256 | aucune : toujours 200 (`degraded` + `detail` si les modèles manquent) |
| `GET /model/metadata` | pour le détecteur et le classifieur : nom, architecture (tag `arch` du Registry), version, alias, précision, run, taille d'entrée, métriques de validation et de test (celles de la version publiée, `models.json`), classes et leur nombre | 503 modèles non chargés |
| `POST /predict` (multipart, champ `image`) | `count`, `pills` (boîte en pixels et normalisée, confiance de détection, `top5` = classes et probabilités), `timings` (`detection_ms`, `classification_ms`, `inference_ms`, `server_ms` = toute la requête), versions des modèles, avertissement | 400 pas une image lisible (type ou contenu), 413 trop grande, 422 champ manquant, 503 modèles non chargés |
| `POST /predict/visualize` (multipart, champ `image`, `?max_side=1280`) | image PNG annotée : une boîte par comprimé, couleur stable par classe, étiquette « classe probabilité (det confiance) » ; en-têtes `X-Pill-Count`, `X-Inference-Ms`, `X-Server-Ms` | mêmes erreurs que `/predict` |

L'ancienne route `GET /api/health` (sonde de Docker Compose) reste inchangée.

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model/metadata
curl -F "image=@photo.jpg;type=image/jpeg" http://127.0.0.1:8000/predict
curl -F "image=@photo.jpg;type=image/jpeg" -o annotee.png http://127.0.0.1:8000/predict/visualize
```

Sur les modèles servis (06/11/2026 : détecteur YOLOv8n FP16 v6, classifieur YOLO11s-cls FP32
v1) et une photo du split de comparaison : `/health` `ok`, fournisseur `CPUExecutionProvider` ;
`/predict` 1 comprimé, confiance 0,94, `inference_ms` 49, `server_ms` 51.

Tests : [`backend/tests/test_vision.py`](../backend/tests/test_vision.py), chaque route et
chaque code d'erreur avec les modèles ONNX factices ; test de fumée de l'image Docker dans
[docker-build.yml](../.github/workflows/docker-build.yml) (`/health` dégradé et `/predict` 503
sans modèles).

Grad-CAM n'est pas servi par l'API : il demande les gradients du modèle PyTorch, alors que
l'API n'exécute que l'ONNX (sans PyTorch, D97) ; il est prévu à l'étape 2.6.

## POST /api/scan

Formulaire multipart, un champ `image` : JPEG, PNG ou WebP, 10 Mo et 50 mégapixels au plus
(`TUJYA_SCAN_MAX_BYTES`, `TUJYA_SCAN_MAX_PIXELS`). Erreurs : 401 non connecté, 413
trop grand, 415 pas une image, 503 modèles ou tables de médicaments absents (le message dit
quelle commande corrige le problème).

Chaîne ([`backend/app/inference.py`](../backend/app/inference.py),
ONNX Runtime, CPU par défaut, CUDA avec `TUJYA_ONNX_PROVIDERS=auto`) :

1. décodage (OpenCV, orientation EXIF appliquée), letterbox à 640 px, détecteur
   (`tujya-pill-detector`, YOLO11n), confiance >= 0,25, NMS à IoU 0,7, 20 comprimés au plus ;
2. pour chaque boîte : recadrage carré avec 20 % de marge, 224 px (exactement les recadrages
   de l'entraînement), classifieur (`tujya-pill-classifier`, YOLO11s-cls), top-5 ;
3. class_id -> nom du comprimé, nom générique et principes actifs depuis les tables des
   médicaments.

Réponse (`ScanResult`), ici sur une photo du split de test :

```json
{
  "image_width": 2560, "image_height": 1920,
  "inference_ms": 50.4,
  "timings": {"detection_ms": 41.2, "classification_ms": 9.2, "total_ms": 50.4},
  "confirm_threshold": 0.85,
  "pills": [{
    "box": {"x1": 1028, "y1": 908, "x2": 1522, "y2": 1140},
    "box_normalized": {"x1": 0.40, "y1": 0.47, "x2": 0.59, "y2": 0.59},
    "detection_confidence": 0.94,
    "candidates": [
      {"class_id": 0, "name": "AMRIX CAP 30MG", "generic_name": "CYCLOBENZAPRINE HYDROCHLORIDE",
       "active_ingredients": ["cyclobenzaprine"], "probability": 0.982}
    ],
    "needs_confirmation": false
  }],
  "message": null,
  "models": {"detect": {"registered_model": "tujya-pill-detector", "version": 1}, "classify": {}},
  "sources": [{"name": "c3pi"}, {"name": "rxnav"}],
  "disclaimer": "Information uniquement, consultez un pharmacien ou un médecin."
}
```

`needs_confirmation` est vrai quand la probabilité top-1 est sous `confirm_threshold` (0,85,
`TUJYA_SCAN_CONFIRM_THRESHOLD`) ; le site demande alors de choisir dans le top-5 ou de
chercher avec `GET /api/pills`. Choix de 0,85 :
[docs/module1/vision.md](module1/vision.md#seuil-de-confirmation)
(D47). Sans comprimé trouvé, `pills` est vide et `message` le dit.

### Modèles

L'API charge `detector.onnx`, `classifier.onnx` et `models.json` depuis
`TUJYA_MODELS_DIR` (par défaut `models/` à la racine du dépôt, ignoré par git), au premier
scan. On le remplit depuis le Model Registry de MLflow (venv d'entraînement, serveur MLflow
lancé) :

```bash
# depuis la racine du dépôt, venv d'entraînement (.venv-train) actif
python -m src.models.publish                 # dernières versions enregistrées
python -m src.models.publish --from-runs     # ou les exports les plus récents de runs/
```

Les fichiers ONNX enregistrés gardent leurs poids en données externes MLflow ; `publish`
écrit chaque modèle en un seul fichier autonome et note dans `models.json` sa version du
registre, son run et son SHA-256, que `/api/scan` renvoie sous `models`.

## GET /api/pills?q=&limit=

Recherche parmi les 150 classes d'identification (nom, nom générique, empreinte, fabricant,
principe actif ; chaque mot doit correspondre ; un nombre correspond aussi au class_id).
`limit` de 1 à 150, 20 par défaut. Renvoie `items` (class_id, name, generic_name,
labeled_by, imprint, color, shape, active_ingredients), `total`, `sources`.

## POST /api/interactions

```json
{"pills": [12, 87], "ingredients": ["fluconazole"]}
```

De deux à dix éléments au total : des class_id de comprimés (issus d'un scan ou de
`/api/pills`) et/ou des noms de principes actifs. Chaque paire d'éléments est vérifiée (les
principes actifs d'un comprimé contre ceux de l'autre élément ; ceux d'une même association
ne sont pas comparés entre eux). Pour chaque paire : `interactions` (ingredient_a,
ingredient_b, severity, description, source), `highest_severity`, `shared_ingredients` (la
même substance des deux côtés), et `message` quand rien n'est listé :

- « Aucune interaction connue dans cette base de données. » quand les deux éléments sont
  connus ;
- un message « aucune vérification n'est possible » quand un élément est inconnu, ou est un
  comprimé sans principe actif connu (classe 104, la multivitamine).

Les paires sont triées par sévérité (Major, Moderate, Minor, Unknown, puis les paires sans
interaction). « Unknown » est le niveau non gradué de DDInter : une interaction existe. Le
`message` général compte les paires avec une interaction et celles qui n'ont pas pu être
vérifiées.

Données : les tables PostgreSQL chargées par `python -m app.pharma_loader` ([docs/infra.md](infra.md),
[docs/module2/interactions.md](module2/interactions.md)).
Tests : [`backend/tests/test_scan.py`](../backend/tests/test_scan.py)
(petits modèles ONNX factices construits dans le test,
[`fake_models.py`](../backend/tests/fake_models.py)) et
[`backend/tests/test_medicines.py`](../backend/tests/test_medicines.py) (SQLite).

## Mesuré sur les vrais modèles

`python -m src.models.evaluate_scan --split test` (venv du projet `.venv`) fait
tourner exactement cette chaîne sur les 600 photos « consumer » du split de test : comprimé
trouvé 98,5 %, top-1 89,3 %, top-5 95,8 %, comme l'évaluation d'Ultralytics
([vision.md](module1/vision.md)) ; environ 50 à 220 ms par photo sur le
CPU du portable, selon la charge. Vérifié à nouveau sur les conteneurs de la copie
restructurée (2026-10-29) : la même photo de test donne la classe 0 à 0,982, en 59 ms.

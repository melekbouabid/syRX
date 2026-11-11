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
| `GET /api/demo/images` | galerie de photos du split de test, jamais vues à l'entraînement par les modèles servis (lecture seule, D105) | [ci-dessous](#galerie-de-démo-et-grad-cam) |
| `GET /api/demo/images/{id}/gradcam` | Grad-CAM précalculée d'une photo de la galerie (JPEG, 404 sinon, D104) | [ci-dessous](#galerie-de-démo-et-grad-cam) |
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

Sur les modèles servis (07/11/2026 : détecteur YOLO11s FP32 v9, champion de l'étape 2.2,
classifieur YOLO11s-cls FP32 v1) et des photos du split de comparaison : `/health` `ok`,
fournisseur `CPUExecutionProvider` ; `/model/metadata` détecteur `yolo11s` v9, alias `champion`
et `served` ; `/predict` sur 30 photos : 34 comprimés sur 34 trouvés, détection médiane 59 ms,
`server_ms` médian 66 ms (06/11, avec YOLOv8n FP16 v6 : `inference_ms` 49, `server_ms` 51 sur
une photo).

Tests : [`backend/tests/test_vision.py`](../backend/tests/test_vision.py), chaque route et
chaque code d'erreur avec les modèles ONNX factices ; test de fumée de l'image Docker dans
[docker-build.yml](../.github/workflows/docker-build.yml) (`/health` dégradé et `/predict` 503
sans modèles).

Grad-CAM n'est pas calculé par l'API : il demande les gradients du modèle PyTorch, alors que
l'API n'exécute que l'ONNX (sans PyTorch, D97). Il est précalculé pour les photos de la
galerie et servi comme fichier (étape 2.6, D104, [ci-dessous](#galerie-de-démo-et-grad-cam)).

Couleurs des boîtes de `/predict/visualize` : couleur `class_id mod 20` de la palette à 20
couleurs des tracés d'Ultralytics, texte noir ou blanc selon la luminance ; le site dessine
ses boîtes avec les mêmes couleurs (`frontend/src/lib/vision.ts`, D107).

## POST /api/scan

Formulaire multipart, un champ `image` : JPEG, PNG ou WebP, 10 Mo et 50 mégapixels au plus
(`TUJYA_SCAN_MAX_BYTES`, `TUJYA_SCAN_MAX_PIXELS`). Erreurs : 401 non connecté, 413
trop grand, 415 pas une image, 503 modèles ou tables de médicaments absents (le message dit
quelle commande corrige le problème).

Chaîne ([`backend/app/inference.py`](../backend/app/inference.py),
ONNX Runtime, CPU par défaut, CUDA avec `TUJYA_ONNX_PROVIDERS=auto`) :

1. décodage (OpenCV, orientation EXIF appliquée), letterbox à 640 px, détecteur
   (`tujya-pill-detector`, YOLO11s depuis l'étape 2.2), confiance >= 0,25, NMS à IoU 0,7, 20 comprimés au plus ;
2. pour chaque boîte : recadrage carré avec 20 % de marge, 224 px (exactement les recadrages
   de l'entraînement), classifieur (`tujya-pill-classifier`, YOLO11s-cls), top-5 ;
3. class_id -> nom du comprimé, nom générique et principes actifs depuis les tables des
   médicaments.

Réponse (`ScanResult`), ici sur une photo du split de test :

```json
{
  "image_width": 2560, "image_height": 1920,
  "inference_ms": 50.4,
  "timings": {"detection_ms": 41.2, "classification_ms": 9.2, "total_ms": 50.4, "server_ms": 63.0},
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

`timings.total_ms` = ONNX Runtime seul (détection + identification) ; `timings.server_ms` =
toute la requête côté serveur (lecture de l'envoi, décodage, base des médicaments). Le site
affiche `server_ms` à côté de la latence **client** qu'il mesure lui-même dans le navigateur
(`performance.now()` autour de l'appel, envoi et réception compris).

`needs_confirmation` est vrai quand la probabilité top-1 est sous `confirm_threshold` (0,85,
`TUJYA_SCAN_CONFIRM_THRESHOLD`) ; le site demande alors de choisir dans le top-5 ou de
chercher avec `GET /api/pills`. Choix de 0,85 :
[docs/module1/vision.md](module1/vision.md#seuil-de-confirmation)
(D47). Sans comprimé trouvé, `pills` est vide et `message` le dit.

### Boîtes de médicaments et photos refusées (étape 2.7)

La même route lit aussi le texte imprimé (OCR RapidOCR en ONNX,
[`backend/app/boxes.py`](../backend/app/boxes.py), méthode et résultats :
[docs/module1/boites.md](module1/boites.md), D108-D112). `kind` dit ce que montre la photo :

- `pill` : des comprimés (réponse ci-dessus, `box` vaut `null`) ;
- `box` : une boîte. `pills` est toujours vide (aucune classe de comprimé n'est donnée à une
  boîte) et `box` contient les lignes lues, la marque de
  [config/brand_aliases.csv](../config/brand_aliases.csv), et chaque principe actif avec la
  voie (`brand`, `printed_name`, `rxnav`), le texte lu, la confiance (confiance OCR x
  ressemblance des mots), la source et sa référence ; `recognised` est faux quand rien n'est
  reconnu (« boîte non reconnue », rien n'est inventé) ;
- `unknown` : ni l'un ni l'autre (meilleure détection de comprimé sous 0,5 et peu de texte) :
  la photo est refusée, `message` le dit.

```json
{
  "kind": "box",
  "pills": [],
  "box": {
    "text": ["paracetamolo", "COMPRESSE", "EFFERALGAN500mg", "16 compresse"],
    "brand": "Efferalgan", "brand_country": "FR", "brand_text": "EFFERALGAN500mg",
    "ingredients": [{
      "name": "acetaminophen", "method": "brand", "matched_text": "EFFERALGAN500mg",
      "confidence": 0.9999, "source": "BDPM (ANSM), CIS_COMPO_bdpm.txt du 03/11/2026",
      "reference": "CIS 60025403 : EFFERALGAN 1000 mg, comprimé pelliculé",
      "url": "https://base-donnees-publique.medicaments.gouv.fr/download/file/CIS_COMPO_bdpm.txt",
      "in_interaction_table": true
    }],
    "recognised": true, "confidence": 0.9999,
    "rxnav": "disabled (no TUJYA_RXNAV_CONTACT_EMAIL)"
  },
  "timings": {"detection_ms": 60.1, "classification_ms": 5.0, "total_ms": 65.1, "ocr_ms": 410.0},
  "message": "Boîte de médicament : principe actif lu sur l'emballage. Vérifiez-le sur la boîte et sa notice."
}
```

Réglages (`TUJYA_*`) : `BOX_OCR_ENABLED` (vrai), `OCR_THREADS` (4), `SCAN_PILL_MIN_CONFIDENCE`
(0,5), `BOX_TEXT_CHARS` (20), `BRAND_TABLE_FILE`, `INN_ALIASES_FILE`, `RXNAV_CONTACT_EMAIL`
(vide : secours RxNav désactivé), `RXNAV_URL` (https seulement), `RXNAV_TIMEOUT_SECONDS`.
`timings.ocr_ms` est le temps de lecture du texte. L'agent a le même service : outil
`lookup_brand` (marque -> principes actifs avec la source) dans le chat et le serveur MCP.

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

## GET /api/documents

Les documents indexés pour le RAG, avec leurs métadonnées (compte connecté ; page
« Documents », [docs/module2/rag.md](module2/rag.md#consulter-les-documents-indexés), D114).
Paramètres facultatifs : `doc_type` (`label`, `abstract`, `interaction`), `ingredient` (le
document en parle ou le cite), `section`, `language`, `source` (« openFDA drug label »,
« PubMed », « DDInter 2.0 »), `q` (mots du titre ou de l'identifiant, 200 caractères au plus),
`limit` (1 à 200, 50 par défaut), `offset`. Renvoie `total`, `offset`, `limit`, `items`
(doc_id, doc_type, title, source_name, url, date, version, license, language, article_type,
is_retracted, ingredients, mentioned_ingredients, sections, chunk_count), `facets` (nombre de
documents par `doc_type`, `language` et `source` parmi ceux qui correspondent) et `index`
(store, collection, points, embedding_model, dim, indexed_at, documents, chunks,
payload_indexes, dashboard_url).

`GET /api/documents/{doc_id}` : le même résumé plus `chunks` (chunk_id, chunk_index,
section, title, text, content_hash = SHA-256), dans l'ordre du document ; les phrases avec
une dose chiffrée sont remplacées par « […] ». 404 si le document n'existe pas ; 503 (les
deux routes) si l'index est vide ou si Qdrant ne répond pas.
Code : [`backend/app/routers/documents.py`](../backend/app/routers/documents.py),
[`src/rag/documents.py`](../src/rag/documents.py) ; tests :
[`tests/rag/test_documents.py`](../tests/rag/test_documents.py).

## Mesuré sur les vrais modèles

`python -m src.models.evaluate_scan --split test` (venv du projet `.venv`) fait
tourner exactement cette chaîne sur les 600 photos « consumer » du split de test : comprimé
trouvé 98,5 %, top-1 89,3 %, top-5 95,8 %, comme l'évaluation d'Ultralytics
([vision.md](module1/vision.md)) ; environ 50 à 220 ms par photo sur le
CPU du portable, selon la charge. Vérifié à nouveau sur les conteneurs de la copie
restructurée (2026-10-29) : la même photo de test donne la classe 0 à 0,982, en 59 ms.

## Galerie de démo et Grad-CAM

`GET /api/demo/images` liste les photos de [`config/demo_gallery.csv`](../config/demo_gallery.csv)
(choisies à la main, D52 et D105) que l'API accepte de montrer
([`backend/app/gallery.py`](../backend/app/gallery.py)) : une photo « consumer » du split de
test v2 ([`config/splits/c3pi_v2/test.csv`](../config/splits/c3pi_v2/test.csv), celui du
détecteur servi) qui n'est pas dans le train ni la val v1
([`config/splits/selected_images.csv`](../config/splits/selected_images.csv), ceux du
classifieur servi), et dont le fichier existe. Chaque élément : `id`, `class_id` (étiquette
C3PI, pour comparer), `expected_name`, `why` (`major`, `two-pills`, `confirm`,
`no-ingredient`), `url`, `thumbnail_url` (360 px) et `gradcam_url` (null sans Grad-CAM).
Les huit photos sont aussi dans `test_holdout.txt` de D93 (aucune photo de la même session en
train ou val v1), ce que vérifie un test quand les données DVC sont là.

`GET /api/demo/images/{id}/gradcam` renvoie la photo (1 280 px au plus, mêmes proportions) où
la carte Grad-CAM du classifieur est fondue dans le carré recadré de chaque comprimé détecté,
pour la classe top-1 que donne le scan ; 404 si la photo n'existe pas ou si sa carte n'a pas
été calculée. Les cartes sont précalculées hors de l'API
([`src/models/gradcam_gallery.py`](../src/models/gradcam_gallery.py), étape DVC
`gradcam_gallery`, venv d'entraînement) : mêmes boîtes et mêmes recadrages que l'API (la chaîne
ONNX de `backend/app/inference.py`), poids PyTorch de la version servie du classifieur
(`models/train/classifier`, run vérifié contre `models.json`), dernière convolution de la tête
(`model.10.conv`, 7 x 7 à 224 px) ; le script s'arrête si PyTorch et ONNX Runtime ne donnent
pas la même probabilité (écart 1e-3 au plus ; 2e-7 mesuré). Sorties dans
`data/processed/c3pi/gradcam/` (DVC, jamais dans git) avec `gradcam.json` (modèle, couche,
probabilités par comprimé), montées en lecture seule dans le conteneur `backend`.

Tests : [`backend/tests/test_demo.py`](../backend/tests/test_demo.py) (règle des splits, 200 /
404 de Grad-CAM, connexion requise) et
[`tests/models/test_gradcam_gallery.py`](../tests/models/test_gradcam_gallery.py).

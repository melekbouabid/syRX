# Module 1 : Industrialisation Vision & MLOps

Identification de comprimés sur photo, de l'image brute au service en production locale :
un détecteur **YOLO11n** trouve chaque comprimé, un classifieur **YOLO11s-cls** entraîné par
*transfer learning* l'identifie parmi **150 classes** du jeu public NLM C3PI. Les deux
modèles sont suivis dans **MLflow** (tracking + Model Registry), exportés en **ONNX** et
servis par **ONNX Runtime** dans l'API FastAPI, sans PyTorch ni Ultralytics à l'exécution.
Données versionnées par **DVC**, application lancée par **Docker Compose**, tests et images
vérifiés par **GitHub Actions**. Évaluation : soutenance pratique (démonstration en direct).

Rapport technique : [docs/module1/rapport_module1_vision_mlops.pdf](rapport_module1_vision_mlops.pdf)
(source [Markdown](rapport_module1_vision_mlops.md)). Vue d'ensemble du projet :
[README principal](../../README.md).

## Exigences du module et fichiers correspondants

| Exigence | Où la trouver |
|---|---|
| Détection des comprimés avec YOLOv8/v11 | [src/models/train.py](../../src/models/train.py) (tâche `detect`, YOLO11n) ; boîtes d'entraînement produites automatiquement par [src/data/autolabel.py](../../src/data/autolabel.py) et [prepare.py](../../src/data/prepare.py) ; [docs/module1/vision.md](vision.md) |
| Identification par *transfer learning* | [src/models/train.py](../../src/models/train.py) (tâche `classify`, YOLO11s-cls préentraîné) ; évaluation de bout en bout : [src/models/evaluate.py](../../src/models/evaluate.py) |
| Données (C3PI, sous-ensemble, split) | [src/data/c3pi.py](../../src/data/c3pi.py), [data_cli.py](../../src/data/data_cli.py), [config/pill_classes.csv](../../config/pill_classes.csv), [config/splits/](../../config/splits/) ; [docs/module1/dataset_c3pi.md](dataset_c3pi.md) |
| Plus de données et Big Data avec PySpark (demande du professeur) | jobs PySpark [src/data/spark/](../../src/data/spark/) (index, qualité, split sans fuite, augmentation des classes rares, ePillID, corpus RAG), cluster Spark ([docker-compose.yml](../../docker-compose.yml), [infra/docker/spark/](../../infra/docker/spark/)), étapes DVC `spark_*` et `dataset_stats` ([dvc.yaml](../../dvc.yaml), [params.yaml](../../params.yaml)), split v2 [config/splits/c3pi_v2/](../../config/splits/c3pi_v2/), statistiques dans MLflow (`tujya-data`), tests [tests/data/spark/](../../tests/data/spark/) ; data card : [big_data.md](big_data.md) |
| Backend FastAPI asynchrone, schémas Pydantic, Swagger sur `/docs` | endpoints du guide `GET /health` (GPU, versions et alias MLflow chargés), `GET /model/metadata`, `POST /predict`, `POST /predict/visualize` (image annotée), erreurs 400 / 413 / 503 : [backend/app/routers/vision.py](../../backend/app/routers/vision.py), tests [backend/tests/test_vision.py](../../backend/tests/test_vision.py) (D97) ; `POST /api/scan` : [backend/app/routers/scan.py](../../backend/app/routers/scan.py) ; schémas : [backend/app/schemas.py](../../backend/app/schemas.py) ; Swagger : [backend/app/main.py](../../backend/app/main.py) ; [docs/api.md](../api.md) |
| Export ONNX et inférence ONNX Runtime | export et vérification ONNX = PyTorch : [src/models/export_onnx.py](../../src/models/export_onnx.py) (étape DVC `export_onnx`) ; inférence servie : [backend/app/inference.py](../../backend/app/inference.py) ; contrôle sur le split de test : [src/models/evaluate_scan.py](../../src/models/evaluate_scan.py) |
| Frontend React du guide — **fait (étape 2.6)** : Tailwind CSS, glisser-déposer, galerie d'exemples en 1 clic, webcam en inférence continue, barres de probabilité animées, boîtes colorées par classe sur canvas, latence client et serveur en ms, modèle champion (`GET /model/metadata`) | page [frontend/src/pages/ScanPage.tsx](../../frontend/src/pages/ScanPage.tsx), composants [frontend/src/components/vision.tsx](../../frontend/src/components/vision.tsx), logique [frontend/src/lib/vision.ts](../../frontend/src/lib/vision.ts), appels centralisés [frontend/src/services/api.ts](../../frontend/src/services/api.ts), thème [frontend/src/index.css](../../frontend/src/index.css) ; tests Vitest `*.test.ts(x)` ; galerie sans fuite [backend/app/gallery.py](../../backend/app/gallery.py) ; [docs/app.md](../app.md#page-scanner-guide-du-module-1-étape-26) ; captures [01](../screenshots/01-scanner.jpg) à [07](../screenshots/07-swagger.jpg), [18](../screenshots/18-drag-and-drop.jpg) à [21](../screenshots/21-champion-models.jpg) (D105 à D107) |
| Grad-CAM — **fait (étape 2.6)** | [src/utils/gradcam.py](../../src/utils/gradcam.py) (tests [tests/utils/test_gradcam.py](../../tests/utils/test_gradcam.py), PyTorch), cartes de la galerie précalculées avec les poids du classifieur servi par [src/models/gradcam_gallery.py](../../src/models/gradcam_gallery.py) (étape DVC `gradcam_gallery`, sorties DVC), servies par `GET /api/demo/images/{id}/gradcam` ([backend/app/routers/demo.py](../../backend/app/routers/demo.py)) et affichées dans le Scanner ; capture [20](../screenshots/20-gradcam.jpg) ; [docs/api.md](../api.md#galerie-de-démo-et-grad-cam) (D104) |
| Boîtes de médicaments (remarque 18) — **fait (étape 2.7)** : lecture du texte imprimé, table des marques sourcée, DCI imprimée, RxNav en secours, refus des photos ni comprimé ni boîte, comparaison RapidOCR / EasyOCR | service [backend/app/boxes.py](../../backend/app/boxes.py), `POST /api/scan` (`kind`, `box`) [backend/app/routers/scan.py](../../backend/app/routers/scan.py), tables [config/brand_aliases.csv](../../config/brand_aliases.csv) et [config/inn_aliases.csv](../../config/inn_aliases.csv), données et vérité terrain [src/data/medicine_boxes.py](../../src/data/medicine_boxes.py) (étape DVC `prepare_boxes`), comparaison [src/models/ocr/](../../src/models/ocr/) (MLflow `tujya-boxes`, Registry `tujya-box-ocr` : RapidOCR champion 0,767, EasyOCR challenger 0,480), outil `lookup_brand` de l'agent ; tests [backend/tests/test_boxes.py](../../backend/tests/test_boxes.py) ; [boites.md](boites.md) ; captures [22](../screenshots/22-box-brand.jpg) à [26](../screenshots/26-chat-brand.jpg) ; validation sur 11 vraies boîtes sous licence libre de Wikimedia Commons (`--set web`, 3 Panadol sur 3 -> paracétamol sourcé, 11 principes actifs justes sur 11, aucun faux) (D108 à D112, D115) |
| Optimisation des poids : quantification FP16 / INT8 et ONNX Runtime | variantes [src/models/quantize.py](../../src/models/quantize.py) (FP16, INT8 statique QDQ calibré sur le val, INT8 dynamique), banc et Registry [src/models/optimize.py](../../src/models/optimize.py) (un run MLflow par variante, alias `served`), publication de la variante servie [src/models/serving.py](../../src/models/serving.py) + [publish.py](../../src/models/publish.py) ; tests [tests/models/test_quantize.py](../../tests/models/test_quantize.py), [test_optimize.py](../../tests/models/test_optimize.py), [test_serving.py](../../tests/models/test_serving.py) ; tableau FP32 / FP16 / INT8 : [optimisation_onnx.md](optimisation_onnx.md) (D95, D96) |
| DVC (versionnement des données) | [dvc.yaml](../../dvc.yaml) (étapes `prepare`, `spark_*`, `dataset_stats`, et pour les modèles `train_detector`, `train_classifier`, `export_onnx`, `evaluate` : `dvc dag`), [params.yaml](../../params.yaml) (hyperparamètres `train`, règles de promotion `registry`), [dvc.lock](../../dvc.lock), [.dvc/config](../../.dvc/config), pointeurs `../data/**/*.dvc` ; [docs/infra.md](../infra.md#remote-dvc-depuis-le-2026-10-28) |
| Comparaison de détecteurs et recherche d'hyperparamètres (remarques du professeur : autres modèles, hyperparamètres, branches) — **fait : YOLO11s champion, YOLOv10n challenger** | une branche `experiment/detector-<archi>` par modèle (YOLOv8n, YOLOv8s, YOLO11s, YOLOv10n, Faster R-CNN) ; code commun [src/models/detection/](../../src/models/detection/) (métriques COCO, latence ONNX CPU, Optuna, exécuteur, import MLflow parent + enfants trié par score, réévaluation du champion, latence de tous les modèles en tours entrelacés `compare_latency.py`, tableau et figure `compare.py`), jeu de détection v2 (étape DVC `prepare_detect_v2`, [src/data/prepare_detect_v2.py](../../src/data/prepare_detect_v2.py)), espaces de recherche `config/hpo/`, notebooks `notebooks/kaggle/` ; tests [tests/models/detection/](../../tests/models/detection/) ; mode d'emploi et résultats : [kaggle_detecteurs.md](kaggle_detecteurs.md) (D93, D94, D98) |
| Comparaison de classifieurs et recherche d'hyperparamètres (axe CNN / ViT : baseline contre modèles avancés, déséquilibre, augmentation, Grad-CAM) — **en attente Kaggle** | une branche `experiment/classifier-<archi>` par modèle (baseline CNN maison, ResNet-50, EfficientNetV2-S, ConvNeXt-Tiny, ViT-B/16, YOLO11s-cls) ; architectures [src/models/architecture.py](../../src/models/architecture.py), Grad-CAM [src/utils/gradcam.py](../../src/utils/gradcam.py) ; code commun [src/models/classification/](../../src/models/classification/) (crops et prétraitement de l'API, top-1 / top-5 / F1 macro / matrice de confusion, Optuna, entraînement *transfer learning* puis *fine-tuning* sélectif, poids de classe ou *focal loss*, RandAugment / MixUp / CutMix sur le train, export ONNX avec la liste de classes, import MLflow avec garde-fou, réévaluation du champion v1, tableau et figure) ; jeu de crops v2 (étape DVC `prepare_classify_v2`, [src/data/prepare_classify_v2.py](../../src/data/prepare_classify_v2.py)) ; tests [tests/models/classification/](../../tests/models/classification/) ; mode d'emploi, budget GPU et réévaluation du champion : [kaggle_classifieurs.md](kaggle_classifieurs.md) (D99-D103) |
| MLflow : tracking des expériences | expérience `tujya-vision` : [src/models/train.py](../../src/models/train.py), [evaluate.py](../../src/models/evaluate.py) ; chaque run via [src/utils/mlflow_utils.py](../../src/utils/mlflow_utils.py) (tags `git_commit`, `dvc_rev`, `dataset`, `seed` ; `requirements.txt` figé, `dvc.lock`, `params.yaml`) ; [mlflow.md](mlflow.md) ; serveur : [infra/docker/mlflow/Dockerfile](../../infra/docker/mlflow/Dockerfile) ; capture [12](../screenshots/12-mlflow-vision.jpg) |
| MLflow : Model Registry | modèles `tujya-pill-detector` et `tujya-pill-classifier` (LoggedModel MLflow 3) enregistrés par [export_onnx.py](../../src/models/export_onnx.py), alias `champion` / `challenger` donnés par `promote_if_better` ([mlflow_utils.py](../../src/utils/mlflow_utils.py)), baseline v1 ré-enregistrée par [register_baseline.py](../../src/models/register_baseline.py) ; champions copiés vers `models/` pour l'API par [publish.py](../../src/models/publish.py) ; [mlflow.md](mlflow.md) ; captures [14](../screenshots/14-mlflow-registry.jpg), [15](../screenshots/15-mlflow-classifier.jpg) |
| Docker Compose | [docker-compose.yml](../../docker-compose.yml), [backend/Dockerfile](../../backend/Dockerfile), [frontend/Dockerfile](../../frontend/Dockerfile), [infra/](../../infra/) ; [docs/infra.md](../infra.md) |
| Git avancé : branches, tags annotés, hooks pre-commit (secrets, messages de commit) | [docs/git_workflow.md](../git_workflow.md), [.pre-commit-config.yaml](../../.pre-commit-config.yaml), [tools/hooks/](../../tools/hooks/) et leurs tests [tools/tests/](../../tools/tests/test_hooks.py) |
| GitHub Actions CI/CD | [.github/workflows/ci-tests.yml](../../.github/workflows/ci-tests.yml) (flake8, black, ruff, mypy, ESLint, pytest de `src/` et de `backend/`, Vitest et build du frontend), [.github/workflows/docker-build.yml](../../.github/workflows/docker-build.yml) (images multi-stage construites et testées, jamais publiées) ; [docs/cd.md](../cd.md) |
| Tests | [tests/data/](../../tests/data/) et [backend/tests/](../../backend/tests/) : API de scan avec de petits modèles ONNX factices ([test_scan.py](../../backend/tests/test_scan.py), [fake_models.py](../../backend/tests/fake_models.py)), étiquetage automatique ([test_autolabel.py](../../tests/data/test_autolabel.py)), métadonnées C3PI ([test_c3pi.py](../../tests/data/test_c3pi.py)), MLflow et `promote_if_better` sur un serveur local temporaire ([tests/utils/](../../tests/utils/test_mlflow_utils.py)) |
| Soutenance pratique (démonstration en direct) | script de démonstration : [docs/demo.md](../demo.md) ; questions probables : [docs/questions_jury.md](../questions_jury.md) |

## Résultats

Mesurés sur le split de test de C3PI (photos jamais vues à l'entraînement), détails dans
[docs/module1/vision.md](vision.md) et le rapport.

| Mesure | Résultat |
|---|---:|
| Détection (YOLO11n, 1 classe) : mAP50 / mAP50-95 | **0,972** / **0,861** |
| Identification (YOLO11s-cls, 150 classes, crops de test) : top-1 / top-5 | **93,5 %** / **98,1 %** |
| Bout en bout avec ONNX Runtime (600 photos de test) : comprimé trouvé / top-1 / top-5 | 98,5 % / **89,3 %** / 95,8 % |
| Seuil de confirmation 0,85 (choisi sur la validation) : photos acceptées / justes parmi elles | 70,5 % / 99,3 % |
| ONNX Runtime contre Ultralytics : même nombre de boîtes / IoU moyen | 40/40 photos / 0,9999 |

Sous le seuil, l'application demande à l'utilisateur de confirmer ou de choisir dans le top-5 :
l'identification reste une suggestion, jamais une certitude.

## Organisation

Le module suit l'arborescence du guide (`src/data`, `src/models`, `backend/app`, `tests/`),
partagée avec le module 2.

```
src/data/              préparation des données C3PI
  c3pi.py, data_cli.py index des métadonnées, sélection des 150 classes, téléchargement vérifié
  autolabel.py         boîtes automatiques (aucune annotation à la main)
  prepare.py           jeux YOLO de détection et de classification (étape DVC `prepare`)
src/models/            entraînement et évaluation (venv d'entraînement .venv-train, PyTorch)
  train.py             YOLO11 detect / classify (params.yaml), test, run MLflow (étapes train_*)
  export_onnx.py       export ONNX, contrôle, Model Registry, promote_if_better (étape export_onnx)
  register_baseline.py modèles du 28/10 ré-enregistrés comme baseline (sans réentraînement)
  evaluate.py          évaluation de bout en bout (détecteur -> crop -> classifieur, étape evaluate)
  evaluate_scan.py     même code que l'API, mesuré sur un split
  publish.py           champions du Model Registry -> models/ (lu par l'API)
  requirements-train.txt  ultralytics, mlflow, dvc, onnx
src/utils/mlflow_utils.py  runs tracés, log_model, register, alias, promote_if_better
backend/app/inference.py  chaîne de scan servie par l'API (ONNX Runtime, numpy, OpenCV)
tests/data/            tests des données (C3PI, étiquetage automatique)
tests/utils/           MLflow (runs tracés, Registry, promote_if_better)
backend/tests/test_scan.py  API de scan avec de petits modèles ONNX factices
docs/module1/          vision.md, dataset_c3pi.md, mlflow.md, rapport (md + pdf), diagrams/, assets/
```

## Commandes

Depuis la racine du dépôt.

```bash
# Entraînement (venv dédié, GPU conseillé)
uv venv .venv-train -p 3.12 && uv pip install -p .venv-train -r src/models/requirements-train.txt
.venv-train/Scripts/python -m src.models.train classify      # hyperparamètres de params.yaml, MLflow tujya-vision
.venv-train/Scripts/python -m src.models.export_onnx          # ONNX, Model Registry, promote_if_better
.venv-train/Scripts/python -m src.models.publish              # champions du Model Registry -> models/ (lu par l'API)

# Pipeline des modèles (étapes d'entraînement gelées, D91)
.venv-train/Scripts/python -m dvc dag
.venv-train/Scripts/python -m dvc repro export_onnx evaluate

# Jeux d'entraînement depuis les images DVC (venv d'entraînement)
.venv-train/Scripts/python -m dvc repro prepare

# Tests (venv du projet : .venv, voir docs/app.md)
.venv/Scripts/python -m pytest tests/data
.venv/Scripts/python -m pytest tests/utils
cd backend && ../.venv/Scripts/python -m pytest tests/test_scan.py

# Chaîne de l'API sur le split de test (venv du projet, modèles publiés)
.venv/Scripts/python -m src.models.evaluate_scan --split test
```

Les données sont sous `data/` à la racine : images et métadonnées C3PI dans
`data/raw/c3pi/`, jeux préparés dans `data/processed/c3pi/`, versionnées par DVC et jamais par git ; les modèles servis sont dans `models/` (ignoré par
git), remplis par `publish`.

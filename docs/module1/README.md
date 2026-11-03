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
| Backend FastAPI asynchrone, schémas Pydantic, Swagger sur `/docs` | `POST /api/scan` : [backend/app/routers/scan.py](../../backend/app/routers/scan.py) ; schémas : [backend/app/schemas.py](../../backend/app/schemas.py) ; Swagger : [backend/app/main.py](../../backend/app/main.py) ; [docs/api.md](../api.md) |
| Export ONNX et inférence ONNX Runtime | export et vérification ONNX = PyTorch : [src/models/train.py](../../src/models/train.py) ; inférence servie : [backend/app/inference.py](../../backend/app/inference.py) ; contrôle sur le split de test : [src/models/evaluate_scan.py](../../src/models/evaluate_scan.py) |
| Frontend React : upload, webcam, prédictions et boîtes en direct | [frontend/src/pages/ScanPage.tsx](../../frontend/src/pages/ScanPage.tsx), [frontend/src/services/api.ts](../../frontend/src/services/api.ts) ; captures [01](../screenshots/01-scanner.jpg) à [07](../screenshots/07-swagger.jpg) |
| DVC (versionnement des données) | [dvc.yaml](../../dvc.yaml) (étape `prepare`), [dvc.lock](../../dvc.lock), [.dvc/config](../../.dvc/config), pointeurs `../data/**/*.dvc` ; [docs/infra.md](../infra.md#remote-dvc-depuis-le-2026-10-28) |
| MLflow : tracking des expériences | expérience `tujya-vision` : [src/models/train.py](../../src/models/train.py), [evaluate.py](../../src/models/evaluate.py) ; serveur : [infra/docker/mlflow/Dockerfile](../../infra/docker/mlflow/Dockerfile) ; capture [12](../screenshots/12-mlflow-vision.jpg) |
| MLflow : Model Registry | modèles `tujya-pill-detector` et `tujya-pill-classifier` enregistrés par [train.py](../../src/models/train.py), copiés vers `models/` pour l'API par [publish.py](../../src/models/publish.py) ; captures [14](../screenshots/14-mlflow-registry.jpg), [15](../screenshots/15-mlflow-classifier.jpg) |
| Docker Compose | [docker-compose.yml](../../docker-compose.yml), [backend/Dockerfile](../../backend/Dockerfile), [frontend/Dockerfile](../../frontend/Dockerfile), [infra/](../../infra/) ; [docs/infra.md](../infra.md) |
| Git avancé : branches, tags annotés, hooks pre-commit (secrets, messages de commit) | [docs/git_workflow.md](../git_workflow.md), [.pre-commit-config.yaml](../../.pre-commit-config.yaml), [tools/hooks/](../../tools/hooks/) et leurs tests [tools/tests/](../../tools/tests/test_hooks.py) |
| GitHub Actions CI/CD | [.github/workflows/ci-tests.yml](../../.github/workflows/ci-tests.yml) (flake8, black, ruff, mypy, ESLint, pytest de `src/` et de `backend/`, build du frontend), [.github/workflows/docker-build.yml](../../.github/workflows/docker-build.yml) (images multi-stage construites et testées, jamais publiées) ; [docs/cd.md](../cd.md) |
| Tests | [tests/data/](../../tests/data/) et [backend/tests/](../../backend/tests/) : API de scan avec de petits modèles ONNX factices ([test_scan.py](../../backend/tests/test_scan.py), [fake_models.py](../../backend/tests/fake_models.py)), étiquetage automatique ([test_autolabel.py](../../tests/data/test_autolabel.py)), métadonnées C3PI ([test_c3pi.py](../../tests/data/test_c3pi.py)) |
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
  train.py             YOLO11 detect / classify, MLflow, export ONNX, Model Registry
  evaluate.py          évaluation de bout en bout (détecteur -> crop -> classifieur)
  evaluate_scan.py     même code que l'API, mesuré sur un split
  publish.py           versions du Model Registry -> models/ (lu par l'API)
  requirements-train.txt  ultralytics, mlflow, dvc, onnx
backend/app/inference.py  chaîne de scan servie par l'API (ONNX Runtime, numpy, OpenCV)
tests/data/            tests des données (C3PI, étiquetage automatique)
backend/tests/test_scan.py  API de scan avec de petits modèles ONNX factices
docs/module1/          vision.md, dataset_c3pi.md, rapport (md + pdf), diagrams/, assets/
```

## Commandes

Depuis la racine du dépôt.

```bash
# Entraînement (venv dédié, GPU conseillé)
uv venv .venv-train -p 3.12 && uv pip install -p .venv-train -r src/models/requirements-train.txt
.venv-train/Scripts/python -m src.models.train classify --weights yolo11s-cls.pt --epochs 40   # MLflow tujya-vision
.venv-train/Scripts/python -m src.models.publish        # versions du Model Registry -> models/ (lu par l'API)

# Jeux d'entraînement depuis les images DVC (venv d'entraînement)
.venv-train/Scripts/python -m dvc repro prepare

# Tests (venv du projet : .venv, voir docs/app.md)
.venv/Scripts/python -m pytest tests/data
cd backend && ../.venv/Scripts/python -m pytest tests/test_scan.py

# Chaîne de l'API sur le split de test (venv du projet, modèles publiés)
.venv/Scripts/python -m src.models.evaluate_scan --split test
```

Les données sont sous `data/` à la racine : images et métadonnées C3PI dans
`data/raw/c3pi/`, jeux préparés dans `data/processed/c3pi/`, versionnées par DVC et jamais par git ; les modèles servis sont dans `models/` (ignoré par
git), remplis par `publish`.

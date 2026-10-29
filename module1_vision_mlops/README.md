# Module 1 : Industrialisation Vision & MLOps

Identification de comprimés sur photo, de l'image brute au service en production locale :
un détecteur **YOLO11n** trouve chaque comprimé, un classifieur **YOLO11s-cls** entraîné par
*transfer learning* l'identifie parmi **150 classes** du jeu public NLM C3PI. Les deux
modèles sont suivis dans **MLflow** (tracking + Model Registry), exportés en **ONNX** et
servis par **ONNX Runtime** dans l'API FastAPI, sans PyTorch ni Ultralytics à l'exécution.
Données versionnées par **DVC**, application lancée par **Docker Compose**, tests et images
vérifiés par **GitHub Actions**. Évaluation : soutenance pratique (démonstration en direct).

Rapport technique : [docs/rapport_module1_vision_mlops.pdf](docs/rapport_module1_vision_mlops.pdf)
(source [Markdown](docs/rapport_module1_vision_mlops.md)). Vue d'ensemble du projet :
[README principal](../README.md).

## Exigences du module et fichiers correspondants

| Exigence | Où la trouver |
|---|---|
| Détection des comprimés avec YOLOv8/v11 | [training/tujya_vision/train.py](training/tujya_vision/train.py) (tâche `detect`, YOLO11n) ; boîtes d'entraînement produites automatiquement par [training/tujya_vision/autolabel.py](training/tujya_vision/autolabel.py) et [prepare.py](training/tujya_vision/prepare.py) ; [docs/vision.md](docs/vision.md) |
| Identification par *transfer learning* | [training/tujya_vision/train.py](training/tujya_vision/train.py) (tâche `classify`, YOLO11s-cls préentraîné) ; évaluation de bout en bout : [training/tujya_vision/evaluate.py](training/tujya_vision/evaluate.py) |
| Données (C3PI, sous-ensemble, split) | [training/tujya_vision/c3pi.py](training/tujya_vision/c3pi.py), [data_cli.py](training/tujya_vision/data_cli.py), [../config/pill_classes.csv](../config/pill_classes.csv), [../config/splits/](../config/splits/) ; [docs/dataset_c3pi.md](docs/dataset_c3pi.md) |
| Backend FastAPI asynchrone, schémas Pydantic, Swagger sur `/docs` | `POST /api/scan` : [../app/backend/app/routers/scan.py](../app/backend/app/routers/scan.py) ; schémas : [../app/backend/app/schemas.py](../app/backend/app/schemas.py) ; Swagger : [../app/backend/app/main.py](../app/backend/app/main.py) ; [../docs/api.md](../docs/api.md) |
| Export ONNX et inférence ONNX Runtime | export et vérification ONNX = PyTorch : [training/tujya_vision/train.py](training/tujya_vision/train.py) ; inférence servie : [serving/scanner.py](serving/scanner.py) ; contrôle sur le split de test : [serving/evaluate_scan.py](serving/evaluate_scan.py) |
| Frontend React : upload, webcam, prédictions et boîtes en direct | [../app/frontend/src/pages/ScanPage.tsx](../app/frontend/src/pages/ScanPage.tsx), [../app/frontend/src/api.ts](../app/frontend/src/api.ts) ; captures [01](../docs/screenshots/01-scanner.jpg) à [07](../docs/screenshots/07-swagger.jpg) |
| DVC (versionnement des données) | [../dvc.yaml](../dvc.yaml) (étape `prepare`), [../dvc.lock](../dvc.lock), [../.dvc/config](../.dvc/config), pointeurs `../data/**/*.dvc` ; [../docs/infra.md](../docs/infra.md#dvc-remote-since-2026-10-28) |
| MLflow : tracking des expériences | expérience `tujya-vision` : [training/tujya_vision/train.py](training/tujya_vision/train.py), [evaluate.py](training/tujya_vision/evaluate.py) ; serveur : [../infra/docker/mlflow/Dockerfile](../infra/docker/mlflow/Dockerfile) ; capture [12](../docs/screenshots/12-mlflow-vision.jpg) |
| MLflow : Model Registry | modèles `tujya-pill-detector` et `tujya-pill-classifier` enregistrés par [train.py](training/tujya_vision/train.py), copiés vers `models/` pour l'API par [publish.py](training/tujya_vision/publish.py) ; captures [14](../docs/screenshots/14-mlflow-registry.jpg), [15](../docs/screenshots/15-mlflow-classifier.jpg) |
| Docker Compose | [../docker-compose.yml](../docker-compose.yml), [../app/backend/Dockerfile](../app/backend/Dockerfile), [../app/frontend/Dockerfile](../app/frontend/Dockerfile), [../infra/](../infra/) ; [../docs/infra.md](../docs/infra.md) |
| GitHub Actions CI/CD | [../.github/workflows/ci.yml](../.github/workflows/ci.yml) (job `module1` et les autres), [../.github/workflows/cd.yml](../.github/workflows/cd.yml) (images construites et testées, jamais publiées) ; [../docs/cd.md](../docs/cd.md) |
| Tests | [tests/](tests/) : API de scan avec de petits modèles ONNX factices ([test_scan.py](tests/test_scan.py), [fake_models.py](tests/fake_models.py)), étiquetage automatique ([test_autolabel.py](tests/test_autolabel.py)), métadonnées C3PI ([test_c3pi.py](tests/test_c3pi.py)) |
| Soutenance pratique (démonstration en direct) | script de démonstration : [../docs/demo.md](../docs/demo.md) ; questions probables : [../docs/questions_jury.md](../docs/questions_jury.md) |

## Résultats

Mesurés sur le split de test de C3PI (photos jamais vues à l'entraînement), détails dans
[docs/vision.md](docs/vision.md) et le rapport.

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

```
module1_vision_mlops/
  training/            projet d'entraînement (paquet tujya_vision, son venv avec PyTorch)
    tujya_vision/      c3pi, data_cli, autolabel, prepare, train, evaluate, publish
    requirements.txt   ultralytics, mlflow, dvc, onnx
  serving/             chaîne de scan servie par l'API (ONNX Runtime, numpy, OpenCV)
    scanner.py         détecteur + crop + classifieur, importé par app/backend
    evaluate_scan.py   même code que l'API, mesuré sur un split
  tests/               tests du module (venv du backend)
  docs/                vision.md, dataset_c3pi.md, rapport (md + pdf), diagrams/, assets/
```

## Commandes

```bash
# Entraînement (venv dédié, GPU conseillé)
cd module1_vision_mlops/training
uv venv .venv -p 3.12 && uv pip install -p .venv -r requirements.txt
python -m tujya_vision.train classify --weights yolo11s-cls.pt --epochs 40   # MLflow tujya-vision
python -m tujya_vision.publish        # versions du Model Registry -> models/ (lu par l'API)

# Jeux d'entraînement depuis les images DVC (depuis la racine du dépôt, venv d'entraînement)
dvc repro prepare

# Tests du module (venv du backend : app/backend/.venv, voir app/README.md)
cd module1_vision_mlops && ../app/backend/.venv/Scripts/python -m pytest

# Chaîne de l'API sur le split de test (depuis la racine, venv du backend, modèles publiés)
python -m module1_vision_mlops.serving.evaluate_scan --split test
```

Les données (images C3PI, métadonnées, jeux préparés) sont sous `data/vision/` à la racine,
versionnées par DVC et jamais par git ; les modèles servis sont dans `models/` (ignoré par
git), remplis par `publish`.

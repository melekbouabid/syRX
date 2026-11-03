# MLflow, Model Registry et pipeline DVC des modèles de vision

Comment chaque modèle de vision est entraîné, tracé, enregistré et promu. Conventions du lab
MLflow : chaque run est rattaché au code, aux données et à l'environnement qui l'ont produit ;
le Model Registry utilise les alias `champion` / `challenger` ; une nouvelle version n'est
promue que si elle bat le champion (`promote_if_better`). Décisions : D91 et D92 dans
[docs/decisions.md](../decisions.md).

## Traçabilité d'un run : `src/utils/mlflow_utils.py`

`start_tracked_run(experiment, run_name, seed=..., dataset=..., dataset_path=...)` démarre un
run MLflow et y ajoute :

| Élément | Contenu |
|---|---|
| tag `git_commit` | commit du code (`git_dirty=true` si des fichiers suivis étaient modifiés) |
| tag `dvc_rev` | identifiant d'objet git de `dvc.lock` : la version de toutes les données DVC |
| tags `dataset`, `dataset_md5` | nom du jeu (`params.yaml`, `train.dataset`) et hash DVC du dossier |
| tag et paramètre `seed` | graine de l'entraînement |
| artefacts `reproducibility/` | `requirements.txt` figé depuis l'environnement, `dvc.lock`, `params.yaml` |

`git checkout <git_commit> && dvc checkout` restaure le code et les données d'un run.
Le même module fournit `log_model` (API MLflow 3, `name=`, un LoggedModel avec ses métriques),
`register` (nouvelle version, alias facultatif) et `promote_if_better`.

## Pipeline DVC (`dvc.yaml`, paramètres dans `params.yaml`)

```
prepare -> train_detector   -\
        -> train_classifier --> export_onnx -> evaluate
```

| Étape | Commande | Sorties |
|---|---|---|
| `train_detector`, `train_classifier` | `python -m src.models.train detect` / `classify` : entraînement YOLO avec `params.yaml` `train.*`, test, run MLflow | `models/train/<modèle>/` (poids, `results.csv`, `args.yaml`), métriques `models/train/<modèle>.json` (avec le `run_id`) |
| `export_onnx` | `python -m src.models.export_onnx` : export ONNX, contrôle ONNX Runtime = PyTorch, LoggedModel dans le run d'entraînement, version du Registry, `promote_if_better` | `models/onnx/`, métriques `models/export.json` |
| `evaluate` | `python -m src.models.evaluate` : détecteur puis classifieur ONNX sur les 600 photos de test | métriques `models/evaluate.json`, run `evaluate-end-to-end` |

Les deux étapes d'entraînement sont gelées (`frozen: true`) : leur sortie enregistrée est la
baseline (D91), et un `dvc repro` ne lance pas par accident plusieurs heures
d'entraînement. Pour réentraîner (GPU, Kaggle) : `dvc unfreeze train_classifier`, `dvc repro`.

```bash
# venv d'entraînement, serveur MLflow lancé (docker compose up -d mlflow)
.venv-train/Scripts/python -m dvc dag                         # graphe du pipeline
.venv-train/Scripts/python -m dvc repro export_onnx evaluate  # export, Registry, évaluation
.venv-train/Scripts/python -m dvc metrics show                # métriques des étapes
.venv-train/Scripts/python -m dvc params diff                 # paramètres changés
.venv-train/Scripts/python -m src.models.publish              # champions -> models/ (API)
```

## Model Registry et promotion

| Modèle enregistré | Métrique de promotion | Gain minimal |
|---|---|---:|
| `tujya-pill-detector` | `test_mAP50_95` (mAP@[.5:.95], split de test) | 0,005 |
| `tujya-pill-classifier` | `test_top1` (split de test) | 0,005 |

`promote_if_better(name, version, metric, min_gain)` :

- pas encore de champion : la version devient `champion` ;
- meilleure d'au moins `min_gain` (et strictement meilleure) : elle devient `champion`,
  l'ancien champion devient `challenger` ;
- sinon le champion reste ; la version devient `challenger` si elle bat le challenger actuel,
  sinon elle est archivée.

Chaque version porte un tag `status` (champion / challenger / archived) et un tag `promotion`
qui explique la décision ; la description rappelle l'architecture, le jeu, la métrique et le
run. Les alias remplacent les stages Staging / Production, dépréciés depuis MLflow 2.9.
Chaque version promue reçoit aussi un tag git annoté `model/<tâche>-<archi>-vX.Y.Z` dont la note
donne les métriques, le `run_id`, la version du Registry et le statut
([docs/git_workflow.md](../git_workflow.md)).

## Baseline (étape 2.1)

Les modèles servis depuis le 28/10 sont la version 1 des deux modèles enregistrés, alias
`champion`, ré-enregistrés par `python -m src.models.register_baseline` sans réentraînement
(D91) et réévalués sur le split de test :

| Modèle | Version | Métriques de test (réévaluation) | Tag git |
|---|---:|---|---|
| `tujya-pill-detector` (YOLO11n, 640 px) | 1 | mAP50 0,972, mAP@[.5:.95] 0,861, précision 0,950, rappel 0,939 | `model/detector-yolo11n-v1.0.0` |
| `tujya-pill-classifier` (YOLO11s-cls, 224 px) | 1 | top-1 93,7 %, top-5 98,5 % (588 crops, 148 classes) | `model/classifier-yolo11s-cls-v1.0.0` |

Les étapes 2.2 (détecteurs) et 2.3 (classifieurs) comparent d'autres architectures à cette
baseline : chaque candidat passe par `export_onnx`, donc par `promote_if_better`.

## Tests

`tests/utils/test_mlflow_utils.py` (venv `.venv`, CI `ci-tests.yml`) : tags et artefacts de
`start_tracked_run`, `log_model` et `register`, et `promote_if_better` (première version,
meilleure version promue, version moins bonne non promue, gain inférieur à `min_gain`, métrique
à minimiser, métrique absente), sur un serveur MLflow local temporaire (base SQLite).

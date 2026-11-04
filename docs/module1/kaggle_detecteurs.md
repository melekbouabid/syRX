# Comparaison des détecteurs sur Kaggle (étape 2.2)

Remarques du professeur : « pourquoi seulement YOLO ? », « où sont les hyperparamètres ? »,
« où sont les branches ? ». Cette étape compare cinq détecteurs de comprimés au détecteur
servi (baseline YOLO11n, étape 2.1), chacun avec sa propre recherche d'hyperparamètres
Optuna et sa branche git, sur le GPU gratuit de Kaggle. Décisions : D93 (jeu de données et
équité de la comparaison) et D94 (organisation, Kaggle, import MLflow) dans
[docs/decisions.md](../decisions.md).

**État : code, notebooks et import prêts et testés ; entraînements Kaggle à lancer**
(mode d'emploi ci-dessous). Les résultats et le champion seront ajoutés ici à l'import.

## Les modèles et leurs branches

| Branche | Modèle | Famille | Paramètres | Notebook | Recherche (`config/hpo/`) |
|---|---|---|---:|---|---|
| `experiment/detector-yolov8n` | YOLOv8n (Ultralytics) | une étape | 3,2 M | `notebooks/kaggle/detector_yolov8n.ipynb` | 12 essais, 10 époques sur 35 % du train ; final 60 époques |
| `experiment/detector-yolov8s` | YOLOv8s (Ultralytics) | une étape | 11 M | `detector_yolov8s.ipynb` | 10 essais, 10 époques sur 30 % ; final 50 époques |
| `experiment/detector-yolo11s` | YOLO11s (Ultralytics) | une étape | 9,4 M | `detector_yolo11s.ipynb` | 10 essais, 10 époques sur 30 % ; final 50 époques |
| `experiment/detector-yolov10n` | YOLOv10n (Ultralytics) | une étape, sans NMS | 2,3 M | `detector_yolov10n.ipynb` | 12 essais, 10 époques sur 35 % ; final 60 époques |
| `experiment/detector-fasterrcnn` | Faster R-CNN ResNet-50 FPN (torchvision) | deux étapes | 41 M | `detector_fasterrcnn.ipynb` | 8 essais, 3 époques sur 20 % ; final 12 époques |
| `main` (baseline, étape 2.1) | YOLO11n | une étape | 2,6 M | - | réévaluée sur le même split |

Tous partent de poids préentraînés sur COCO (*transfer learning*), une classe « pill ».
Toutes les branches `experiment/` partent de `feature/detector-kaggle-tooling` (code commun)
et n'ajoutent que ce qui est propre au modèle : sa configuration Optuna, son notebook et,
pour Faster R-CNN, le code torchvision et ses tests. Seule la branche du champion sera
fusionnée dans `main` ; les autres restent poussées comme trace des expériences.

## Hyperparamètres recherchés

Optuna, échantillonneur TPE avec graine (`seed: 0`), un run MLflow enfant par essai. Chaque
essai entraîne le modèle préentraîné quelques époques sur une partie du train et il est noté
sur la **validation** (`val_mAP50_95`) ; le meilleur essai est réentraîné sur tout le train
(arrêt précoce sur la validation) puis évalué **une seule fois** sur le test.

| Hyperparamètre | YOLO (4 modèles) | Faster R-CNN |
|---|---|---|
| `optimizer` | SGD, AdamW | SGD, AdamW |
| `lr0` (échelle log, plage selon l'optimiseur) | SGD 1e-3 à 2e-2 ; AdamW 1e-4 à 2e-3 | SGD 2e-3 à 4e-2 ; AdamW 1e-5 à 3e-4 |
| `imgsz` | 480, 640 | 480, 640 |
| `batch` | 8, 16, 32 | 2, 4, 8 |
| `mosaic`, `mixup` | 0 à 1, 0 à 0,3 | sans équivalent torchvision : non cherchés |
| `hsv_h`, `hsv_s`, `hsv_v` | 0 à 0,03 ; 0,2 à 0,9 ; 0,1 à 0,6 | mêmes plages |
| `fliplr`, `flipud` | 0 à 0,5 | 0 à 0,5 |

Les augmentations ne s'appliquent qu'au train (Ultralytics et
[fasterrcnn.py](../../src/models/detection/fasterrcnn.py)), jamais à la validation ni au test.

## Données et équité de la comparaison (D93)

- Jeu : **C3PI v2** (étape 1.4), photos « consumer » des 380 classes, boîtes produites par
  [autolabel.py](../../src/data/autolabel.py) comme pour v1, photos douteuses écartées ; étape
  DVC `prepare_detect_v2` ([prepare_detect_v2.py](../../src/data/prepare_detect_v2.py)),
  sortie `data/processed/c3pi/datasets_v2/detect` (images 640 px, format YOLO).
- Split de comparaison `test_holdout.txt` : les photos de test v2 dont la session de prise de
  vue ne contient aucune photo vue par la baseline (entraînée sur v1) en train ou en val.
  Tous les modèles, baseline comprise, sont mesurés dessus : métriques `test_v2_*`, métrique
  de promotion `test_v2_mAP50_95` (`params.yaml`, `compare.detector`). Le test v2 complet est
  aussi rapporté (`test_v2_all_*`).
- Un seul code de mesure pour tous ([metrics.py](../../src/models/detection/metrics.py)) :
  mAP@[.5:.95] et mAP50 au protocole COCO (101 points, au plus 100 détections par image),
  précision et rappel au seuil de meilleur F1 (IoU 0,5).
- Latence : ONNX Runtime, CPU, une image par appel, après échauffement
  ([latency.py](../../src/models/detection/latency.py)), mesurée **sur le PC** à l'import pour
  chaque modèle (`onnx_cpu_latency_ms`, même machine pour tous) ; celle du CPU Kaggle est
  gardée à part (`kaggle_onnx_cpu_latency_ms`).

| Split | Photos « consumer » v2 | Gardées (boîtes sûres) | Boîtes |
|---|---:|---:|---:|
| train | 9 148 | 7 571 | |
| val | 1 580 | 1 315 | |
| test (complet, `test_v2_all_*`) | 1 777 | 1 490 | 1 716 |
| test de comparaison (`test_holdout.txt`, `test_v2_*`) | | 995 | 1 133 |

Au total 10 376 photos gardées sur 12 505 (12 236 boîtes, 380 classes), 756 Mo ; 495 photos de
test appartiennent à une session vue par la baseline et sont exclues du split de comparaison.
Chiffres : `data/processed/c3pi/datasets_v2/detect/summary.json`.

Baseline réévaluée sur ce split (`python -m src.models.detection.evaluate_champion`, run
`evaluate-detector-v1-c3pi-v2`) : la référence que les candidats doivent battre d'au moins
0,005.

| Modèle | mAP@[.5:.95] comparaison | mAP50 | Précision | Rappel | mAP@[.5:.95] test v2 complet | Latence ONNX CPU (PC) |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11n baseline (entraînée sur v1, version 1, `champion`) | 0,7888 | 0,9513 | 0,931 | 0,940 | 0,8050 | 25,0 ms |

Sur le test v1 elle avait 0,861 : le split v2 (nouvelles classes, nouvelles sessions de prise
de vue) est plus difficile.

## Code (branche `feature/detector-kaggle-tooling`)

| Fichier | Rôle |
|---|---|
| [src/data/prepare_detect_v2.py](../../src/data/prepare_detect_v2.py) | jeu de détection v2 et split de comparaison (étape DVC `prepare_detect_v2`) |
| [src/models/detection/metrics.py](../../src/models/detection/metrics.py) | mAP@[.5:.95], mAP50, précision, rappel (COCO) |
| [src/models/detection/latency.py](../../src/models/detection/latency.py) | latence ONNX Runtime CPU (prétraitement de l'API) |
| [src/models/detection/labels.py](../../src/models/detection/labels.py) | étiquettes YOLO, boîtes en pixels, cibles torchvision, sous-ensembles |
| [src/models/detection/hpo.py](../../src/models/detection/hpo.py) | espaces Optuna lus dans `config/hpo/detector_<archi>.yaml` |
| [src/models/detection/runner.py](../../src/models/detection/runner.py) | une expérience : recherche, réentraînement, test, export ONNX, dossier d'export |
| [src/models/detection/yolo_backend.py](../../src/models/detection/yolo_backend.py) | entraînement, prédiction, export des YOLO (Ultralytics) |
| [src/models/detection/import_kaggle.py](../../src/models/detection/import_kaggle.py) | import d'un export dans MLflow : run parent + runs enfants, Registry, `promote_if_better` |
| [src/models/detection/evaluate_champion.py](../../src/models/detection/evaluate_champion.py) | réévaluation du champion sur le split de comparaison |
| [src/models/detection/package_kaggle.py](../../src/models/detection/package_kaggle.py) | dossier et zip du dataset Kaggle privé (`runs/kaggle/`, ignoré par git) |
| [src/models/detection/notebook.py](../../src/models/detection/notebook.py) | génère les notebooks Kaggle (un modèle de notebook pour tous) |
| [src/models/detection/compare.py](../../src/models/detection/compare.py) | tableau et figure de comparaison depuis MLflow |
| [tests/models/detection/](../../tests/models/detection/) | tests (store MLflow SQLite temporaire, famille de modèles factice, sans GPU) |

Le format d'export (`run.json`, schéma 1) contient : code (`git_commit`, branche), données
(`MANIFEST.json` du dataset : hash DVC, `dvc_rev`), environnement (GPU, versions), espace et
résultats de chaque essai (paramètres, métriques de validation, historique par époque),
entraînement final (historique, métriques de validation et de test, latence) ; à côté :
`weights/best.pt`, `model/detector.onnx`, `trials.csv`, `config.yaml`, `requirements.txt`,
`training/` (courbes), `optuna.db` (l'étude, reprise si la session redémarre).

## Mode d'emploi pas à pas

Rien de tout cela ne demande de mot de passe ni de jeton : les notebooks se lancent à la main
depuis le site Kaggle (un jeton API `kaggle.json`, si vous en installez un vous-même, permet
seulement d'automatiser les étapes 2 et 5).

### 1. Construire et empaqueter le jeu de données (PC)

```bash
# venv d'entraînement, depuis la racine (déjà fait sur la branche : dvc.lock à jour)
.venv-train/Scripts/python -m dvc repro prepare_detect_v2
.venv-train/Scripts/python -m src.models.detection.package_kaggle --owner <votre-nom-kaggle>
```

Sortie dans `runs/kaggle/` (ignoré par git) : le dossier `tujya-c3pi-detect-v2/` et
**`tujya-c3pi-detect-v2.zip` (0,76 Go, 10 376 images)**.

### 2. Créer le dataset Kaggle privé

Kaggle > **Datasets** > **New Dataset** : déposer `tujya-c3pi-detect-v2.zip`, titre
« Tujya C3PI v2 pill detection », identifiant `tujya-c3pi-detect-v2`, visibilité
**Private** (licence ODbL ; images du NLM pour la recherche, à ne pas redistribuer).
Avec un jeton installé par vos soins : `kaggle datasets create -p runs/kaggle/tujya-c3pi-detect-v2`
(privé par défaut).

### 3. Lancer chaque notebook (un par modèle)

Pour chaque modèle : Kaggle > **Code** > **New Notebook** > **File** > **Import Notebook** >
onglet GitHub, `melekbouabid/syRX`, branche `experiment/detector-<archi>`, fichier
`notebooks/kaggle/detector_<archi>.ipynb` (ou importer le fichier `.ipynb` téléchargé).
Puis, dans **Settings** : **Accelerator** GPU (T4 x1, ou P100 pour Faster R-CNN), **Internet**
activé (clone de la branche et `pip install` des versions figées). **Add Input** : votre dataset
`tujya-c3pi-detect-v2`. Enfin **Save Version** > **Save & Run All (Commit)** : le calcul
continue sans navigateur ouvert.

| Notebook | GPU conseillé | Durée estimée |
|---|---|---|
| `detector_yolov8n.ipynb` | T4 x1 | environ 3 h |
| `detector_yolov10n.ipynb` | T4 x1 | environ 3 h |
| `detector_yolov8s.ipynb` | T4 x1 | environ 5 h |
| `detector_yolo11s.ipynb` | T4 x1 | environ 5 h |
| `detector_fasterrcnn.ipynb` | P100 (ou T4 x1) | environ 7 h |

Total environ 23 h de GPU : le quota gratuit est d'environ 30 h par semaine et une session
s'arrête après 12 h (la recherche a une limite `timeout_hours` pour finir à temps). Lancer
d'abord un seul notebook pour vérifier le parcours.

### 4. Récupérer les sorties

Dans la version sauvegardée de chaque notebook, onglet **Output** : télécharger
`detector_<archi>_export.zip` (poids, ONNX, essais, `run.json`).

### 5. Réévaluer le champion actuel (une seule fois, PC)

```bash
docker compose up -d mlflow
.venv-train/Scripts/python -m src.models.detection.evaluate_champion
```

Ajoute `test_v2_*` et `onnx_cpu_latency_ms` au run de la version champion (et un run
`evaluate-detector-v1-c3pi-v2`) : sans cela l'import refuse de promouvoir un candidat.

### 6. Importer chaque export dans MLflow (PC)

```bash
.venv-train/Scripts/python -m src.models.detection.import_kaggle detector_yolov8n_export.zip
# ... un appel par modèle (plusieurs fichiers acceptés)
```

Chaque import crée, dans l'expérience `tujya-vision`, le run parent `detector-<archi>`
(tags `git_commit`, `git_branch`, `dvc_rev`, `dataset_md5`, `seed`, GPU ; paramètres retenus ;
métriques ; artefacts) et un run enfant `trial-NNN` par essai Optuna ; puis il enregistre
l'ONNX comme nouvelle version de `tujya-pill-detector` et appelle `promote_if_better` sur
`test_v2_mAP50_95` (gain minimal 0,005) : alias `champion`, `challenger` ou version archivée.
Le résumé est écrit dans `models/compare/detector_<archi>.json`.

Vérifications dans l'interface MLflow (http://127.0.0.1:5000) : expérience `tujya-vision`,
filtre `tags.stage = 'hpo'` puis **Compare** > **Parallel Coordinates** (hyperparamètres
contre `val_mAP50_95`) ; filtre `tags.stage IN ('compare', 'compare-eval')` pour comparer
les modèles ; onglet **Models** pour les alias.

### 7. Poids dans DVC, tableau et figure (sur la branche de chaque modèle)

```bash
git switch experiment/detector-<archi>
# dossier d'export dézippé, jamais dans git
.venv-train/Scripts/python -m dvc add data/models/detectors/detector-<archi>
git add data/models/detectors/detector-<archi>.dvc data/models/detectors/.gitignore
git commit -m "exp(detector): record the <archi> Kaggle run (test_v2_mAP50_95 0.xxxx)"
```

Puis, une fois tous les modèles importés (branche du champion) :
`python -m src.models.detection.compare` remplit le tableau ci-dessous et écrit la figure
`docs/module1/assets/detecteurs_comparaison.png` (mAP@[.5:.95] contre latence CPU).

### 8. Fin de l'étape

1. Tests et lint (CLAUDE.md « Working rules »), `dvc status` propre.
2. `python -m src.models.publish` copie le champion vers `models/` ; vérifier que l'API le
   sert (`/api/scan`, `models.json`). Faster R-CNN est exporté au format des YOLO
   (entrée 1x3xSxS, sortie 1x5xN), l'API le sert sans changement.
3. `git switch main && git merge --no-ff experiment/detector-<champion>`.
4. Tags annotés : un `model/detector-<archi>-v1.0.0` par modèle (note : métriques, run
   MLflow, version du Registry, statut champion / challenger / archivé), puis
   `v3.1.0-detecteurs`.
5. `git push origin main experiment/detector-* v3.1.0-detecteurs model/detector-*` et CI verte.

## Résultats

<!-- comparison:begin -->
En attente des entraînements Kaggle.
<!-- comparison:end -->

## Vérification locale (tests de fumée)

Avant Kaggle, chaque chaîne est vérifiée sur le PC (GPU RTX 2050) en mode « smoke » : le
notebook exécuté avec `TUJYA_SMOKE=1` (ou `python -m src.models.detection.runner --config
config/hpo/detector_<archi>.yaml --smoke`) prend 64 / 16 / 16 images, 2 essais Optuna d'une
époque, un réentraînement d'une époque ; l'export est importé avec
`python -m src.models.detection.import_kaggle <export> --smoke`. Ces runs portent le tag
`smoke=true` et un nom `smoke-...`, ne sont jamais enregistrés dans le Model Registry et
leurs métriques n'ont aucune valeur (une époque sur 64 images) : ils prouvent seulement que
le code, l'export et l'import fonctionnent.

## Limites

- Les boîtes de test sont produites automatiquement (pas d'annotation manuelle) : la
  comparaison mesure l'accord avec cet étiquetage, comme pour la baseline.
- Le split de comparaison est plus petit que le test v2 complet (sessions vues par la
  baseline retirées) ; le test complet est rapporté à côté.
- YOLOv10 : Ultralytics 8.4 exporte en ONNX la tête « un-à-plusieurs » (`end2end: False`,
  même format que YOLOv8, NMS dans l'API), alors que la prédiction PyTorch de l'évaluation
  utilise la tête « un-à-un » sans NMS ; l'écart sera mesuré à l'import (parité ONNX).

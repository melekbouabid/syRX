# Comparaison des détecteurs sur Kaggle (étape 2.2)

Remarques du professeur : « pourquoi seulement YOLO ? », « où sont les hyperparamètres ? »,
« où sont les branches ? ». Cette étape compare cinq détecteurs de comprimés au détecteur
servi (baseline YOLO11n, étape 2.1), chacun avec sa propre recherche d'hyperparamètres
Optuna et sa branche git, sur le GPU gratuit de Kaggle. Décisions : D93 (jeu de données et
équité de la comparaison) et D94 (organisation, Kaggle, import MLflow) dans
[docs/decisions.md](../decisions.md).

**État (07/11/2026) : terminé.** Les cinq détecteurs ont été entraînés sur Kaggle (Tesla T4,
8 à 12 essais Optuna chacun), importés dans MLflow (un run parent et un run enfant par essai)
et comparés à la baseline sur le même split : **YOLO11s est le champion** (version 9,
mAP@[.5:.95] 0,8885), **YOLOv10n le challenger** (version 10, 0,8860), les autres sont
archivés ; règle, ordre d'import et discussion dans la décision D98 et la section
[Résultats](#résultats).

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
`src/models/detection/fasterrcnn.py`, sur la branche `experiment/detector-fasterrcnn`), jamais
à la validation ni au test.

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
  ([latency.py](../../src/models/detection/latency.py)), mesurée **sur le PC** pour tous les
  modèles ensemble, en tours entrelacés avec pauses
  ([compare_latency.py](../../src/models/detection/compare_latency.py), `onnx_cpu_latency_ms`,
  même machine et mêmes images pour tous) ; celle du CPU Kaggle est gardée à part
  (`kaggle_onnx_cpu_latency_ms`).

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
| [src/models/detection/compare_latency.py](../../src/models/detection/compare_latency.py) | latence ONNX CPU, taille et paramètres de tous les détecteurs FP32, mesurés ensemble en tours entrelacés |
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
# tous les exports en un seul appel : importés du meilleur au moins bon (D98)
.venv-train/Scripts/python -m src.models.detection.import_kaggle --no-latency runs/kaggle/detector_*_export.zip
# puis la latence de tous les détecteurs FP32, mesurés ensemble (D96, D98)
.venv-train/Scripts/python -m src.models.detection.compare_latency
```

Ordre d'import : `promote_if_better` compare chaque candidat au champion **en place** avec un
gain minimal, donc une suite d'imports dépend de son ordre (un candidat importé juste après un
modèle à peine moins bon, déjà promu, ne le bat plus de 0,005). Plusieurs exports donnés en un
appel sont donc importés par valeur décroissante de `test_v2_mAP50_95` (lue dans leur
`run.json`) : le résultat ne dépend plus de l'ordre des arguments (`--keep-order` pour garder
l'ordre donné).

Latence : `--no-latency` saute la mesure isolée de l'import ; `compare_latency` chronomètre
ensuite tous les détecteurs FP32 du Registry (baseline comprise) avec le protocole de l'étape 2.4
(5 tours entrelacés de 30 passes, 20 s de pause avant chaque mesure, médianes ; le PC portable
se bride sous charge soutenue, D96) et enregistre `onnx_cpu_latency_ms`, `onnx_cpu_latency_p95_ms`,
la moyenne de chaque tour (`onnx_cpu_latency_round_ms`), `model_size_mb` et `onnx_parameters`
dans le run de chaque version (et dans le run de réévaluation de la baseline).

Chaque import crée, dans l'expérience `tujya-vision`, le run parent `detector-<archi>`
(tags `git_commit`, `git_branch`, `dvc_rev`, `dataset_md5`, `seed`, GPU ; paramètres retenus ;
métriques ; artefacts) et un run enfant `trial-NNN` par essai Optuna ; puis il enregistre
l'ONNX comme nouvelle version de `tujya-pill-detector` et appelle `promote_if_better` sur
`test_v2_mAP50_95` (gain minimal 0,005) : alias `champion`, `challenger` ou version archivée.
Le résumé est écrit dans `models/compare/detector_<archi>.json`.

Vérifications dans l'interface MLflow (http://127.0.0.1:5000) : expérience `tujya-vision`,
filtre `tags.stage = 'hpo'` puis **Compare** > **Parallel Coordinates** (hyperparamètres
contre `val_mAP50_95`) ; filtre `tags.stage LIKE 'compare%' and tags.smoke = 'false'` pour
comparer les modèles (la recherche MLflow n'accepte pas `IN` sur un tag) ; onglet **Models**
pour les alias.

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
`docs/module1/assets/detecteurs_comparaison.png` (mAP@[.5:.95] contre latence CPU et contre
taille du fichier ONNX).

### 8. Fin de l'étape

1. Tests et lint ([docs/project_rules.md](../project_rules.md) ; `pytest` à la racine, dans `backend/` et `tools/`, `ruff`, `flake8`, `black`, `mypy`), `dvc status` propre.
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
| Modèle | mAP@[.5:.95] val | mAP@[.5:.95] test | mAP50 test | Précision | Rappel | mAP@[.5:.95] test v2 complet | Latence ONNX CPU moy. (ms) | p95 (ms) | Taille ONNX (Mo) | Paramètres (M) | Version | Statut | Run MLflow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| yolo11s | 0.8677 | 0.8885 | 0.9739 | 0.952 | 0.936 | 0.8722 | 63.0 | 67.1 | 37.9 | 9.5 | v9 | champion | `894871cd` |
| yolov10n | 0.8569 | 0.8860 | 0.9760 | 0.949 | 0.946 | 0.8703 | 21.9 | 23.9 | 9.4 | 2.3 | v10 | challenger | `d28c1056` |
| yolov8s | 0.8626 | 0.8830 | 0.9747 | 0.950 | 0.941 | 0.8644 | 69.7 | 73.7 | 44.7 | 11.2 | v11 | archived | `e4d25272` |
| yolov8n | 0.8563 | 0.8774 | 0.9750 | 0.947 | 0.942 | 0.8543 | 24.2 | 25.5 | 12.8 | 3.2 | v2 | archived | `b0689536` |
| fasterrcnn | 0.8063 | 0.8218 | 0.9723 | 0.951 | 0.944 | 0.8097 | 420.9 | 435.5 | 165.6 | 41.3 | v12 | archived | `96b290a7` |
| yolo11n (baseline v1) |  | 0.7888 | 0.9513 | 0.931 | 0.940 | 0.8050 | 23.8 | 25.7 | 10.7 | 2.6 | v1 | archived | `6ae2678a` |
<!-- comparison:end -->

Valeurs du 07/11/2026 : mAP@[.5:.95] et mAP50 au protocole COCO sur le split de comparaison
(`test_holdout.txt`, 995 photos, 1 133 boîtes), précision et rappel au seuil de meilleur F1 ;
latence ONNX Runtime CPU sur le PC de développement, tous les modèles FP32 mesurés ensemble
([compare_latency.py](../../src/models/detection/compare_latency.py) : 5 tours entrelacés,
20 s de pause avant chaque mesure, au plus 1,5 s de charge par mesure, médianes ; YOLO :
640 px, Faster R-CNN : 480 px, la taille choisie par sa recherche) ; paramètres = poids du
graphe ONNX. Pour chaque modèle, le tour le plus lent est au plus 15 % au-dessus du plus
rapide (YOLO11n : 23,0 à 26,5 ms ; `onnx_cpu_latency_round_ms` dans MLflow) : aucune mesure
bridée (D98).

![mAP@[.5:.95] contre latence CPU et contre taille du fichier ONNX](assets/detecteurs_comparaison.png)

### Ce que montre la comparaison

- **Les quatre YOLO entraînés sur C3PI v2 battent la baseline** (YOLO11n entraînée sur v1,
  0,7888) de 0,089 à 0,100 de mAP@[.5:.95] : le jeu v2 et la recherche d'hyperparamètres
  apportent plus que le choix entre ces architectures.
- **Les trois meilleurs YOLO sont à 0,0055 les uns des autres** (YOLO11s 0,8885, YOLOv10n
  0,8860, YOLOv8s 0,8830). Avec un seul split de 995 photos, une seule graine et des boîtes
  produites automatiquement, l'écart entre YOLO11s et YOLOv10n (0,0025) n'est pas
  significatif ; le même ordre se retrouve sur le test v2 complet (0,8722 / 0,8703 / 0,8644),
  qui contient le split de comparaison et n'est donc pas une mesure indépendante, mais pas sur
  la validation (YOLOv8s y passe devant YOLOv10n).
- **Le mAP50 est presque le même pour tous** (0,972 à 0,976, baseline 0,951) : tous trouvent
  presque tous les comprimés ; la différence est dans la précision des boîtes aux seuils d'IoU
  élevés, que mesure mAP@[.5:.95].
- **Latence et taille** : YOLOv10n est le plus rapide (21,9 ms, 9,4 Mo, 2,3 M paramètres),
  YOLO11s coûte 2,9 fois plus de temps CPU (63,0 ms) et 4 fois plus de place (37,9 Mo) pour
  +0,0025 de mAP@[.5:.95]. YOLOv8s est plus lent, plus gros et moins précis que YOLO11s.
- **Faster R-CNN** (deux étapes, ResNet-50 FPN, l'autre famille demandée par le professeur) : un
  mAP50 du niveau des YOLO (0,972) mais un mAP@[.5:.95] plus bas de 0,067 que le champion, et
  19 fois plus lent que YOLOv10n (421 ms à 480 px). Ses boîtes sont moins serrées : la
  recherche (3 époques sur 20 % du train) a retenu 480 px, et le réentraînement n'a eu que
  12 époques dans le temps de GPU de Kaggle ; la régression des boîtes se fait sur des
  caractéristiques ramenées à 7 x 7 par RoIAlign avec une perte smooth L1, alors que les YOLO
  optimisent directement le recouvrement (CIoU et *distribution focal loss*). Sa lenteur est
  celle des deux étapes : réseau de propositions, puis tête de classification et de régression
  sur 1 000 propositions par image, sur une colonne vertébrale de 41 M paramètres.
- La parité PyTorch / ONNX (100 images de test, ONNX décodé comme dans l'API) reste à
  0,003 près pour les cinq modèles (`parity_*` dans MLflow), YOLOv10n compris (tête
  « un-à-plusieurs » exportée, voir Limites).

### Hyperparamètres retenus par la recherche Optuna

| Modèle | Essais (meilleur) | `val_mAP50_95` de l'essai | Optimiseur | `lr0` | `imgsz` | `batch` | `mosaic` | `mixup` | Époques finales | GPU Kaggle (min) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n | 12 (n° 11) | 0,7900 | AdamW | 5,1e-4 | 640 | 16 | 0,84 | 0,29 | 60 | 162 |
| YOLOv8s | 10 (n° 6) | 0,7937 | AdamW | 2,1e-4 | 640 | 32 | 0,02 | 0,25 | 50 | 145 |
| YOLO11s | 10 (n° 6) | 0,8043 | AdamW | 2,1e-4 | 640 | 32 | 0,02 | 0,25 | 50 | 202 |
| YOLOv10n | 12 (n° 11) | 0,7855 | AdamW | 1,2e-4 | 640 | 32 | 0,89 | 0,21 | 60 | 185 |
| Faster R-CNN | 8 (n° 0) | 0,7454 | AdamW | 6,9e-5 | 480 | 4 | - | - | 12 | 172 |

Toutes les valeurs (HSV, flips) et l'historique de chaque essai sont dans les runs enfants
(`tags.stage = 'hpo'`). YOLOv8s et YOLO11s ont le même espace de recherche et la même graine :
TPE leur a proposé les mêmes essais et le meilleur est le même (n° 6), donc ces deux
architectures sont comparées à hyperparamètres égaux.

### Promotion et ordre d'import (D98)

Champion en place avant les quatre imports : YOLOv8n (version 2, 0,8774). Règle : un candidat
doit le battre d'au moins 0,005 sur `test_v2_mAP50_95` ; parmi ceux qui le font (YOLO11s
+0,0111, YOLOv10n +0,0086, YOLOv8s +0,0056 ; Faster R-CNN -0,0556), le meilleur devient
champion, et le deuxième modèle de la comparaison challenger. `promote_if_better` appliqué
import par import dépend de l'ordre : sur les 24 ordres possibles des quatre imports (simulés
avec la vraie fonction sur un store MLflow temporaire), 12 donnent YOLO11s / YOLOv10n, 4
YOLO11s / YOLOv8s et 8 YOLOv10n / YOLO11s (YOLOv10n importé avant YOLO11s est promu, puis
YOLO11s ne le bat que de 0,0025). Les exports ont donc été importés en un appel, du meilleur
au moins bon (`import_kaggle` les trie, quel que soit l'ordre des arguments) : YOLO11s,
YOLOv10n, YOLOv8s, Faster R-CNN. Résultat, sans correction manuelle des alias :

| Version | Modèle | `test_v2_mAP50_95` | Alias / statut | Décision (`promotion`) | Run parent | Runs enfants |
|---|---|---:|---|---|---|---:|
| 9 | YOLO11s | 0,8885 | `champion` | promu : +0,0111 contre la version 2 | `894871cd18f840fdba734d7ce985cd40` | 10 |
| 10 | YOLOv10n | 0,8860 | `challenger` | non promu : -0,0025 contre la version 9 | `d28c1056db9b455fb35d7ac83e65bc94` | 12 |
| 11 | YOLOv8s | 0,8830 | archivé | non promu : -0,0055 | `e4d2527274b3477a89ffd4f93d3e145e` | 10 |
| 12 | Faster R-CNN | 0,8218 | archivé | non promu : -0,0667 | `96b290a7da2742ee955f76ca2aff4010` | 8 |
| 2 | YOLOv8n | 0,8774 | archivé (champion jusqu'au 07/11) | importé le 06/11 | `b06895360a7e441899172c1c04bd7721` | 12 |
| 1 | YOLO11n (baseline) | 0,7888 | archivé | baseline de l'étape 2.1 | `6766123e6ecc43b2bda50f01d9419e32` | - |

Poids et exports Kaggle : `data/models/detectors/detector-<archi>` (DVC), un pointeur `.dvc`
par branche `experiment/detector-<archi>` ; seule `experiment/detector-yolo11s` est fusionnée
dans `main`.

## Vérification locale (tests de fumée)

Avant Kaggle, chaque chaîne est vérifiée sur le PC (GPU RTX 2050) en mode « smoke » : le
notebook exécuté avec `TUJYA_SMOKE=1` (ou `python -m src.models.detection.runner --config
config/hpo/detector_<archi>.yaml --smoke`) prend 64 / 16 / 16 images, 2 essais Optuna d'une
époque, un réentraînement d'une époque ; l'export est importé avec
`python -m src.models.detection.import_kaggle <export> --smoke`. Ces runs portent le tag
`smoke=true` et un nom `smoke-...`, ne sont jamais enregistrés dans le Model Registry et
leurs métriques n'ont aucune valeur (une époque sur 64 images) : ils prouvent seulement que
le code, l'export et l'import fonctionnent.

Résultats du 04/11 (RTX 2050 4 Go, notebooks exécutés localement avec `TUJYA_SMOKE=1`, puis
importés avec `--smoke` ; aucune version ajoutée au Registry, le champion reste la version 1) :

| Notebook (commit) | Durée | Essais | Run parent MLflow | mAP@[.5:.95] PyTorch / ONNX (parité, 16 images) | Latence ONNX CPU |
|---|---:|---:|---|---:|---:|
| `detector_yolov8n.ipynb` (`c6dae36`) | 1,7 min | 2 (terminés) | `smoke-detector-yolov8n` (`748e2533`) | 0,420 / 0,421 | 8,2 ms (320 px) |
| `detector_fasterrcnn.ipynb` (`3e37d0f`) | 5,5 min | 2 (terminés) | `smoke-detector-fasterrcnn` (`3be3675c`) | 0,524 / 0,538 | 2 084 ms (320 px) |

Ce qu'ils vérifient : clonage ou dépôt local, lecture du jeu, recherche Optuna (runs enfants),
réentraînement, évaluation sur les deux splits de test, export ONNX lu au format de l'API
(parité PyTorch / ONNX), latence dans un processus séparé, zip, import MLflow avec les tags
de provenance. Les trois autres notebooks sont générés par le même modèle et testés
(`tests/models/detection/test_tools.py` : JSON valide, code Python compilable, identiques au
générateur) ; seules leurs configurations Optuna diffèrent (validées par `test_hpo.py`).

## Limites

- Les boîtes de test sont produites automatiquement (pas d'annotation manuelle) : la
  comparaison mesure l'accord avec cet étiquetage, comme pour la baseline.
- Le split de comparaison est plus petit que le test v2 complet (sessions vues par la
  baseline retirées) ; le test complet est rapporté à côté.
- Faster R-CNN est lent sur CPU en ONNX (421 ms par image à 480 px sur le PC, 1 293 ms sur le
  CPU de Kaggle : NMS et RoIAlign de torchvision exportés pour 1 000 propositions et un seuil
  de score de 0,001) ; la latence fait partie de la comparaison, c'est le prix du détecteur à
  deux étapes.
- Un seul entraînement par modèle (une graine) : les écarts de moins de 0,005 entre YOLO
  récents ne sont pas significatifs (D98).
- Latence : le PC portable ralentit d'environ 15 fois après deux secondes d'inférence
  continue ; chaque mesure est limitée à 1,5 s de charge (`latency_budget_s`), donc les gros
  modèles sont chronométrés sur moins de passes (YOLO11s 16, YOLOv8s 14, Faster R-CNN 2 par
  tour, contre 30 pour les petits YOLO), toujours sur 5 tours.
- YOLOv10 : Ultralytics 8.4 exporte en ONNX la tête « un-à-plusieurs » (`end2end: False`,
  même format que YOLOv8, NMS dans l'API), alors que la prédiction PyTorch de l'évaluation
  utilise la tête « un-à-un » sans NMS ; l'écart est mesuré par les métriques `parity_torch_*` / `parity_onnx_*` (100 images de test,
  ONNX décodé comme dans l'API).

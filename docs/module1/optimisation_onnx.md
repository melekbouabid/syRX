# Optimisation des champions : ONNX FP32, FP16 et INT8 (étape 2.4)

Barème du module : optimisation des poids, quantification (FP16 / INT8) et ONNX Runtime.
Les deux champions du Model Registry sont déclinés en quatre variantes, mesurées par le même
code, et la variante retenue est servie par l'API. Décisions : D95 (alias `served`, outillage
commun) et D96 (tête YOLO en float, méthode de latence) dans [docs/decisions.md](../decisions.md).

## Commandes

```bash
# venv d'entraînement, serveur MLflow lancé (docker compose up -d mlflow)
.venv-train/Scripts/python -m src.models.optimize            # les deux champions
.venv-train/Scripts/python -m src.models.optimize --task detect --no-register   # mesure seule
.venv-train/Scripts/python -m src.models.publish             # copie les variantes servies vers models/
.venv-train/Scripts/python -m src.models.publish --exact     # les champions FP32 eux-mêmes
```

Un nouveau champion (étapes 2.2 et 2.3) n'a besoin que de ces deux commandes : tant que
`optimize` n'a pas été relancé, `publish` sert le nouveau champion en FP32 (une variante d'un
ancien champion n'est jamais servie).

## Les variantes

| Variante | Fabrication ([src/models/quantize.py](../../src/models/quantize.py)) | Calibration |
|---|---|---|
| FP32 | export ONNX du Registry (référence) | - |
| FP16 | poids et activations en float16, entrées et sorties en float32 (`keep_io_types`) : l'API l'utilise sans changement | - |
| INT8 statique | QDQ, poids int8 par canal, activations int8 ; pour le détecteur, la tête YOLO (décodage des boîtes) reste en float (D96) | MinMax, 200 photos du **val** v2 (détecteur), 300 crops du **val** v1 (classifieur) ; jamais le test |
| INT8 dynamique | poids uint8, activations quantifiées à l'exécution | - |

Les métadonnées Ultralytics (`names`, `imgsz`) sont recopiées sur chaque variante : l'API lit
les mêmes informations quelle que soit la précision.

## Mesure ([src/models/optimize.py](../../src/models/optimize.py))

- Détecteur (YOLOv8n, version 2, champion depuis l'import Kaggle de l'étape 2.2) : mAP@[.5:.95]
  sur le split de comparaison de l'étape 2.2 (`test_holdout.txt` de C3PI v2, 995 photos, 1 133
  boîtes), par le même code ([metrics.py](../../src/models/detection/metrics.py), décodage comme
  l'API). Pas de mesure sur le test v1 : des photos du train v2 peuvent y figurer.
- Classifieur (YOLO11s-cls, version 1) : top-1 et top-5 sur les 588 crops de test de C3PI v1.
- Latence : ONNX Runtime CPU, une image par appel, dans un processus neuf
  ([latency.py](../../src/models/detection/latency.py)), 20 images de test, 5 tours entrelacés
  de 30 passes avec 20 s de pause avant chaque mesure, médiane des tours (le PC portable se bride
  sous charge soutenue, D96). Processeur : celui du PC de développement (tag `latency_cpu`).
- Taille : fichier ONNX autonome.
- MLflow : un run par variante dans `tujya-vision` (tag `stage=optimize`, `precision`,
  `variant_of`, provenance `git_commit` / `dvc_rev`) ; chaque variante FP16 / INT8 est une
  version du même modèle enregistré.

## Résultats (06/11/2026)

| Modèle | Variante | Précision | Écart vs FP32 | Latence moyenne (ms) | Latence p95 (ms) | Taille (Mo) | Version Registry | Run MLflow |
|---|---|---:|---:|---:|---:|---:|---|---|
| tujya-pill-detector (YOLOv8n 640) | FP32 | 0,8773 (mAP@[.5:.95]) | - | 24,5 | 26,7 | 12,8 | v2 (`champion`) | `1703391f` |
| | **FP16 (servie)** | 0,8757 | -0,0016 | **22,7** | 24,7 | 6,4 | v6 (`served`) | `170853fc` |
| | INT8 statique | 0,8472 | -0,0301 | 48,8 | 52,5 | 6,2 | v7 | `6a0a46df` |
| | INT8 dynamique | 0,8353 | -0,0420 | 44,4 | 48,0 | 3,5 | v8 | `828809e6` |
| tujya-pill-classifier (YOLO11s-cls 224) | **FP32 (servie)** | 0,9371 (top-1) | - | **4,1** | 4,9 | 22,6 | v1 (`champion`, `served`) | `dd39c8db` |
| | FP16 | 0,9354 | -0,0017 | 4,4 | 4,9 | 11,3 | v5 | `18834c1f` |
| | INT8 statique | 0,8929 | -0,0442 | 11,4 | 13,2 | 5,9 | v6 | `4986d4cb` |
| | INT8 dynamique | 0,6718 | -0,2653 | 8,4 | 9,3 | 5,8 | v7 | `02d0ac15` |

Top-5 du classifieur : FP32 0,9847. Le FP32 du détecteur (0,8773) diffère de 0,0001 de la
mesure de l'import Kaggle (0,8774, ONNX Runtime de Kaggle).

## Choix de la variante servie

Règle (`choose_variant`, `params.yaml` `optimize`) : parmi les variantes qui perdent au plus
0,01 de précision (un point de mAP ou de top-1) et sont au moins 5 % plus rapides que FP32,
la plus rapide ; sinon FP32.

- **Détecteur : FP16** : -0,0016 de mAP@[.5:.95], 7 % plus rapide, fichier deux fois plus
  petit.
- **Classifieur : FP32** : le FP16 perd 0,0017 de top-1 sans être plus rapide sur ce CPU ;
  aucune variante ne remplit les deux conditions.
- **INT8 écarté pour les deux** : sur ce processeur, ONNX Runtime exécute les modèles QDQ
  plus lentement que le float (environ 2x pour le détecteur, 3x pour le classifieur), et la
  précision baisse de 3 à 4 points (26 points pour le classifieur en dynamique, dont les
  activations ne sont pas calibrées). L'INT8 reste utile pour la taille (3,5 à 6 Mo) et sur
  un processeur doté d'instructions entières rapides (VNNI) ; il n'est pas retenu ici.

Vérification dans l'API (`PillScanner` de [backend/app/inference.py](../../backend/app/inference.py),
modèles publiés, seuil de détection 0,25) sur 100 photos du split de comparaison : 114 boîtes
sur 117 trouvées (IoU >= 0,5), 119 prédites : rappel 0,974, précision 0,958. `models/models.json`
indique la version, la précision et la version champion d'origine (`variant_of`).

## Registry après l'étape

| Modèle | Versions | Alias |
|---|---|---|
| tujya-pill-detector | v1 YOLO11n FP32 (baseline), v2 YOLOv8n FP32, v6-v8 variantes de v2 (v3-v5 : première mesure, `status=superseded`, D96) | `champion` v2, `challenger` v1, `served` v6 |
| tujya-pill-classifier | v1 YOLO11s-cls FP32, v5-v7 variantes de v1 (v2-v4 `superseded`) | `champion` v1, `served` v1 |

L'alias `champion` reste sur un modèle FP32 : les détecteurs Kaggle encore à importer sont
comparés FP32 contre FP32 par `promote_if_better` (D95).

## Limites

- Latences mesurées sur un seul PC portable (CPU bridé sous charge) : l'ordre des variantes est
  fiable grâce aux tours entrelacés, les valeurs absolues dépendent de la machine.
- Pas de quantification avec entraînement (QAT) : hors du temps de GPU disponible.
- L'étape 2.2 (comparaison des détecteurs) peut encore changer le champion ; la commande
  `optimize` sera alors relancée sur le nouveau champion.

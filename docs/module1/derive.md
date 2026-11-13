# Dérive des données des photos scannées (étape 2.8)

Demande du guide de mi-parcours : surveiller en production si les données reçues ressemblent
encore aux données d'entraînement. Les modèles de Tujya ont appris sur les photos de C3PI v2
(photos « grand public » d'un comprimé sur fond uni, éclairage correct). Une photo très
sombre, floue, d'une autre taille ou qui ne montre pas un comprimé est notée avec moins de
fiabilité : la confiance affichée ne veut alors plus dire la même chose. Le module compare donc,
en continu, les photos reçues par l'API avec des photos de référence de C3PI v2 (du même type que
celles de l'entraînement, mais jamais apprises), et l'interface affiche
une alerte quand elles s'en écartent. Décision : D122 dans [docs/decisions.md](../decisions.md).

## Les huit statistiques d'une photo

Calculées par le même code ([backend/app/drift.py](../../backend/app/drift.py), `image_stats` /
`scan_stats`) pour la référence et pour l'API :

| Statistique | Définition | Ce qu'elle détecte |
|---|---|---|
| `brightness` | niveau de gris moyen (0-255) | pièce sombre, surexposition |
| `contrast` | écart type des niveaux de gris | photo terne, fond chargé |
| `sharpness` | log10 de la variance du laplacien, image ramenée à 512 px | flou de bougé, mise au point ratée |
| `megapixels` | largeur x hauteur / 1e6 | autre appareil, capture d'écran |
| `aspect_ratio` | grand côté / petit côté | photo recadrée, format portrait de téléphone |
| `boxes` | comprimés trouvés par le détecteur | photo sans comprimé, plusieurs comprimés |
| `detector_confidence` | confiance moyenne des comprimés trouvés | objets qui ne ressemblent pas aux comprimés appris |
| `classifier_confidence` | probabilité top-1 moyenne des comprimés trouvés | comprimé hors des 380 classes, crop de mauvaise qualité |

Les deux dernières n'existent que si un comprimé est trouvé (sinon la photo ne compte pas pour
elles). **Aucune image, aucun nom de fichier, rien sur l'utilisateur n'est gardé** : seulement
ces huit nombres, en mémoire, dans une fenêtre glissante des `window` dernières photos (200 par
défaut) ; un redémarrage de l'API la vide.

## Référence (étape DVC `drift_reference`)

[src/data/drift_reference.py](../../src/data/drift_reference.py) prend 600 photos réparties sur le
**split de validation** du jeu de détection C3PI v2 (photos que les modèles n'ont jamais apprises,
choisies comme celles du test ; pourquoi pas le train : section suivante et D122), les passe dans
le scan servi (mêmes modèles ONNX que l'API) et écrit
`data/processed/c3pi/drift/reference.json` (un échantillon par statistique et les seuils de
`params.yaml` `drift`) et `summary.json` (moyennes et quantiles, métriques DVC). Un run MLflow
dans l'expérience **`tujya-monitoring`** garde les seuils (paramètres), les moyennes et quantiles
(métriques), les deux fichiers et un histogramme par statistique (artefacts), avec les tags de
provenance (`git_commit`, `dvc_rev`, `dataset_md5`).

```bash
# venv du projet ou d'entraînement, modèles publiés dans models/, MLflow démarré
.venv-train/Scripts/python -m dvc repro drift_reference
```

La référence dépend des modèles servis (les deux confiances) : publier un nouveau modèle fait
apparaître l'étape dans `dvc status`, et on la relance.

Référence actuelle (600 photos du val v2, modèles servis YOLO11s v9 + EfficientNetV2-S v8 ;
`summary.json`) :

| Statistique | Photos | Moyenne | 5e centile | Médiane | 95e centile |
|---|---:|---:|---:|---:|---:|
| `brightness` | 600 | 136,1 | 102,2 | 132,3 | 186,8 |
| `contrast` | 600 | 19,3 | 7,3 | 17,2 | 35,1 |
| `sharpness` | 600 | 2,02 | 1,15 | 1,96 | 3,14 |
| `megapixels` | 600 | 6,39 | 0,20 | 4,92 | 14,63 |
| `aspect_ratio` | 600 | 1,371 | 1,333 | 1,333 | 1,778 |
| `boxes` | 600 | 1,17 | 1 | 1 | 2 |
| `detector_confidence` | 597 | 0,877 | 0,550 | 0,930 | 0,957 |
| `classifier_confidence` | 597 | 0,853 | 0,464 | 0,909 | 0,960 |

Trois photos sans comprimé détecté ne comptent pas pour les deux confiances.

## Tests statistiques

Pour chaque statistique, la fenêtre de l'API est comparée à l'échantillon de référence par
deux mesures complémentaires (scipy) :

- **test de Kolmogorov-Smirnov à deux échantillons** (`scipy.stats.ks_2samp`) : dit si la
  différence est réelle (p-value) ;
- **Population Stability Index (PSI)**, classes = déciles de la référence : dit si elle est
  grande (0 = identique ; par convention moins de 0,1 stable, 0,1 à 0,25 modéré, plus de 0,25
  fort).

Une statistique est en **dérive** quand la p-value KS est sous `ks_alpha` (0,01) **et** que son
PSI atteint `psi_drift` (0,25) ; en **avertissement** quand le PSI n'atteint que `psi_warning`
(0,1). Les deux conditions ensemble évitent les deux pièges : avec quelques centaines de photos,
le KS seul signale des écarts trop petits pour compter ; le PSI seul bouge d'environ 0,05 entre
deux échantillons de la même distribution. En dessous de `min_samples` (30) photos dans la
fenêtre, le rapport dit `insufficient` et ne teste rien. Tous les seuils sont dans
[params.yaml](../../params.yaml) (`drift`) et sont recopiés dans le fichier de référence que lit
l'API.

**Pourquoi scipy et pas Evidently AI.** Evidently (0.7.23, vérifié sur PyPI le 12/11/2026) fait
les mêmes tests (KS, PSI) mais déclare 26 dépendances directes, dont pandas, plotly,
statsmodels, scikit-learn, nltk, litestar, un second serveur uvicorn et un module de
télémétrie (`iterative-telemetry`) : trop pour deux fonctions dans l'image de l'API, et une
télémétrie n'a rien à faire dans une application qui ne doit rien envoyer hors du PC. scipy est
déjà dans l'image (dépendance de LightGBM, désormais déclarée dans `backend/requirements.txt`) ;
le calcul du PSI tient en quinze lignes testées.

## API et interface

- `GET /monitoring/drift` (sans compte, comme `/health` : le rapport ne contient que des
  nombres agrégés) : statut global (`unavailable` sans fichier de référence, `insufficient`,
  `ok`, `warning`, `drift`), taille de la fenêtre, statistiques en dérive, et pour chacune
  moyennes, KS, p-value, PSI et statut. Schéma Pydantic `DriftReport` dans Swagger (`/docs`).
- `POST /monitoring/drift/reset` (administrateur) vide la fenêtre, par exemple avant une
  démonstration.
- Chaque photo envoyée à `POST /predict`, `POST /predict/visualize` ou `POST /api/scan` ajoute
  ses huit nombres à la fenêtre (quelques millisecondes, dans le thread du scan).
- Interface : page **Monitoring** (menu de gauche) avec le tableau des huit statistiques, et un
  **bandeau rouge sur la page Scanner** quand le statut est `drift` (jamais pour un simple
  manque de photos), revérifié après chaque scan, avec le conseil de reprendre la photo.

## Démonstration reproductible

[src/data/drift_demo.py](../../src/data/drift_demo.py) joue cinq scénarios de 200 photos (la
taille de la fenêtre), chacun passé dans le scan servi et noté comme une fenêtre de l'API :

| Scénario | Photos | Statut | Statistiques en dérive (PSI) |
|---|---|---|---|
| `test-v2` | test v2, jamais apprises | **ok** | aucune (KS : toutes les p-values au-dessus de 0,01) |
| `train-v2` | train v2, apprises par les modèles | **drift** | confiance du classifieur (0,931 contre 0,853, PSI 1,39), mégapixels (8,6 contre 6,4, PSI 0,93), netteté (PSI 0,44) |
| `dark` | les photos du test, niveaux de gris x 0,35 | **drift** | luminosité (45,6 contre 136,1, PSI 8,3), contraste (6,5), netteté (2,30), confiance du détecteur (0,803, PSI 1,40) ; classifieur en avertissement (0,791) |
| `blurred` | les photos du test, flou gaussien de 2 % du grand côté | **drift** | netteté (0,22 contre 2,02, PSI 8,3), confiance du classifieur (0,244, PSI 6,98), confiance du détecteur (0,692), comprimés trouvés (1,05) |
| `out-of-distribution` | 20 images hors sujet et 180 photos de boîtes de médicaments (étape 2.7) | **drift** | sept statistiques sur huit, dont mégapixels (0,22), contraste (59,3), confiance du classifieur (0,218) |

```bash
# venv du projet, modèles publiés dans models/, référence de `dvc repro drift_reference`
PYTHONPATH="backend;." python -m src.data.drift_demo            # dans le processus, 5 scénarios
PYTHONPATH="backend;." python -m src.data.drift_demo --mlflow   # + un run par scénario (tujya-monitoring)
# contre l'API qui tourne : remplit SA fenêtre, l'alerte s'affiche dans l'application
PYTHONPATH="backend;." python -m src.data.drift_demo --api http://127.0.0.1:8000 --only dark
```

Les cinq runs `drift-demo-<scénario>` de l'expérience `tujya-monitoring` (13/11/2026) gardent le
KS, la p-value et le PSI de chaque statistique et le rapport complet (`drift_report.json`).
Lecture :

- **pas de fausse alerte sur des photos nouvelles du même type** : le test v2 ressort `ok` ;
- **une photo dégradée se voit deux fois** : sur sa propre statistique (luminosité, netteté) et
  sur la confiance des modèles qui baisse (le flou fait tomber la confiance du classifieur de
  0,85 à 0,24 : les prédictions d'une telle fenêtre ne valent plus grand-chose) ;
- **le scénario `train-v2` explique le choix de la référence** (D122) : sur les photos qu'ils ont
  apprises, les modèles sont plus sûrs d'eux (0,93) que sur toute photo nouvelle ; une référence
  calculée sur le train signalerait donc à tort une dérive dès la mise en service.

Captures (après `--api ... --only dark`) : la page Monitoring en dérive
([30](../screenshots/30-monitoring-drift.jpg)) et le bandeau de la page Scanner
([31](../screenshots/31-scan-drift-banner.jpg)).

Tests : [backend/tests/test_drift.py](../../backend/tests/test_drift.py) (statistiques, KS, PSI,
fenêtre, statuts, route, enregistrement par `/predict` sans image) et
[tests/data/test_drift.py](../../tests/data/test_drift.py) (référence, transformations de la
démonstration, et sur de vraies photos du test v2 : 60 photos `ok`, les mêmes assombries ou
floues en `drift` ; sauté sans les données DVC, donc sur GitHub Actions).

## Limites

- La fenêtre vit en mémoire d'un processus : elle se vide au redémarrage et n'est pas partagée
  entre plusieurs processus de l'API (un seul ici). Une version de production l'écrirait dans
  une base (sans l'image).
- Les tests regardent chaque statistique séparément (pas de test multivarié) ; une dérive qui
  ne change aucune des huit (par exemple un nouveau comprimé de même aspect) n'est pas vue :
  la confiance du classifieur est le seul indice.
- Le PSI sur les déciles de la référence se comporte mal pour une variable presque constante
  ou discrète : les déciles du rapport d'aspect sont tous égaux (4:3), son PSI reste à 0 même
  quand le KS vaut 0,96 (images hors distribution carrées) ; à l'inverse le PSI des mégapixels
  (quelques résolutions d'appareils) vaut 0,53 sur le test v2 sans aucun écart réel (p-value KS
  0,68). La règle « KS **et** PSI » évite la fausse alerte, au prix de ne pas voir un changement
  de format seul ; une version suivante pourrait utiliser des classes fixes pour ces deux
  variables.
- Une dérive n'est pas une erreur : elle dit que la confiance affichée est moins fiable et que
  la photo devrait être reprise ; elle ne bloque jamais un scan.

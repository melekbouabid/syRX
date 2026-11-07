# Notebooks

Notebooks d'entraînement et d'exploration, rangés comme le demande le guide des modules.

- `kaggle/` : notebooks lancés sur Kaggle (GPU gratuit) pour comparer les détecteurs et les
  classifieurs (étapes 2.2 et 2.3) et pour le fine-tuning du module 2 (étape 3.5). Ils
  appellent le code de [src/](../src/) ; les résultats vont dans MLflow et les poids dans DVC,
  jamais dans git.
- Les notebooks sont commités **sans leurs sorties** (hook `nbstripout`,
  [.pre-commit-config.yaml](../.pre-commit-config.yaml)).
- Étape 2.2 (détecteurs) : un notebook par modèle, `kaggle/detector_<archi>.ipynb`, sur sa
  branche `experiment/detector-<archi>`, généré par `python -m src.models.detection.notebook`
  (paquets installés sur Kaggle : [kaggle/requirements.txt](kaggle/requirements.txt)).
  Mode d'emploi : [docs/module1/kaggle_detecteurs.md](../docs/module1/kaggle_detecteurs.md).
- Étape 2.3 (classifieurs) : un notebook par modèle, `kaggle/classifier_<archi>.ipynb`, sur sa
  branche `experiment/classifier-<archi>`, généré par
  `python -m src.models.classification.notebook`. Mode d'emploi, ordre de lancement et budget
  GPU : [docs/module1/kaggle_classifieurs.md](../docs/module1/kaggle_classifieurs.md).

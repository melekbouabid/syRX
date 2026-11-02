# Notebooks

Notebooks d'entraînement et d'exploration, rangés comme le demande le guide des modules.

- `kaggle/` : notebooks lancés sur Kaggle (GPU gratuit) pour comparer les détecteurs et les
  classifieurs (étapes 2.2 et 2.3) et pour le fine-tuning du module 2 (étape 3.5). Ils
  appellent le code de [src/](../src/) ; les résultats vont dans MLflow et les poids dans DVC,
  jamais dans git.
- Les notebooks sont commités **sans leurs sorties** (hook `nbstripout`,
  [.pre-commit-config.yaml](../.pre-commit-config.yaml)).

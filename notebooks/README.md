# Notebooks

- `module1/kaggle/`: notebooks run on Kaggle (free GPU) to compare the detectors and the
  classifiers. Each one is generated from one template and lives on its experiment branch
  (`experiment/detector-<arch>`, `experiment/classifier-<arch>`); the champions' notebooks are
  kept here. They call the code of [src/](../src/); results go to MLflow and weights to DVC,
  never to git. Packages installed on Kaggle: [module1/kaggle/requirements.txt](module1/kaggle/requirements.txt).
- Generate a notebook: `python -m src.models.detection.notebook <arch>` or
  `python -m src.models.classification.notebook <arch>`; a test checks that the committed
  notebooks are exactly the generated ones.
- Notebooks are committed **without outputs** (`nbstripout` hook,
  [.pre-commit-config.yaml](../.pre-commit-config.yaml)).

Step-by-step guide: [docs/module1/README.md](../docs/module1/README.md).

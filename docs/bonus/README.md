# Bonus (hors des deux modules notés)

Travaux faits en plus des exigences, une fois les deux modules terminés.

## Outil de toxicité (Tox21)

Pour une ou deux molécules (nom de principe actif ou SMILES), probabilité d'activité dans les
12 tests in vitro de Tox21, **« prédiction d'un modèle sur des tests in vitro, pas une
évaluation clinique »**. Chaque molécule est prédite séparément ; rien n'est jamais dit sur
leur association, et jamais « non toxique ».

- Modèle : empreintes Morgan (RDKit, rayon 2, 2 048 bits), un LightGBM par test, **split par
  scaffold** (6 258 / 782 / 783 molécules) ; ROC-AUC moyen **0,725** en test (0,647 à 0,822
  selon le test), 0,766 en validation ; MLflow `tujya-tox`, modèle enregistré `tujya-tox21`.
- Utilisé par l'outil `predict_toxicity` de l'agent et du serveur MCP (module 2), la route
  `POST /api/toxicity` et la page « Toxicity » du site
  ([capture](../docs/screenshots/16-toxicity.jpg)).
- Code : [toxicity/model.py](toxicity/model.py) (prédiction, domaine d'applicabilité,
  formulations), [toxicity/train_tox21.py](toxicity/train_tox21.py) (téléchargement,
  entraînement, publication), [toxicity/fetch_ingredient_smiles.py](toxicity/fetch_ingredient_smiles.py)
  (structures des 82 principes actifs depuis PubChem). Tests : [tests/test_toxicity.py](tests/test_toxicity.py).
- Documentation : [docs/toxicity.md](docs/toxicity.md) ; décisions D73 et D74 dans
  [../docs/decisions.md](../docs/decisions.md).

## Étude de synergie (retirée)

Une étude de prédiction de la synergie (DrugComb, score ZIP) a été menée puis retirée du
dépôt : elle n'a pas atteint la règle fixée d'avance (RMSE −7,3 % contre la moyenne, sous les
10 % demandés), donc aucun outil `predict_synergy` n'existe. Décisions D75 (l'étude) et D85
(le retrait) ; le code reste dans l'historique git.

## Commandes

```bash
# Depuis la racine du dépôt, venv du backend (app/backend/.venv) ; l'entraînement a aussi
# besoin de MLflow : uv pip install -p app/backend/.venv mlflow==3.17.2
python -m bonus.toxicity.train_tox21 download
python -m bonus.toxicity.train_tox21 train        # serveur MLflow sur 127.0.0.1:5000
python -m bonus.toxicity.train_tox21 publish      # -> models/tox21/, lu par l'API

# Tests
cd bonus && ../app/backend/.venv/Scripts/python -m pytest
```

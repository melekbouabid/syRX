# Workflow Git

Comment le dépôt est organisé dans le temps : les branches, les messages de commit, les tags
des étapes et des modèles, et les hooks qui font respecter ces règles. Le travail suit des
étapes ; chaque étape a sa branche, ses tests, sa validation et une CI au vert avant d'être
fusionnée.

## Branches

| Branche | Usage | Exemple |
|---|---|---|
| `main` | Toujours stable : la CI y est verte et chaque fusion correspond à une étape validée. On n'y travaille jamais directement. | |
| `feature/<sujet>` | Nouvelle fonctionnalité de l'application, de l'API ou du RAG. | `feature/reranker` |
| `experiment/<modèle>` | Une expérience d'entraînement = une branche : un modèle, des hyperparamètres, un run MLflow. Elle est fusionnée si l'expérience est retenue (champion ou challenger), sinon gardée pour la trace. | `experiment/yolo11s-640` |
| `fix/<sujet>` | Correction d'un bug. | `fix/scan-empty-image` |
| `refactor/<sujet>` | Changement de structure sans changement de comportement. | `refactor/module-layout` |
| `chore/<sujet>` | Outillage, configuration, dépendances. | `chore/conventions-hooks` |

Cycle d'une étape :

```bash
git switch main && git pull
git switch -c feature/reranker
# ... petits commits ...
git switch main
git merge --no-ff feature/reranker      # garde la trace de la branche dans l'historique
git tag -a v4.1.0-reranker -m "..."     # voir « Tags » plus bas
git push origin main feature/reranker v4.1.0-reranker
```

On ne fait jamais de `push --force` sur `main` ni sur une branche déjà poussée.

## Messages de commit

Format [Conventional Commits](https://www.conventionalcommits.org/fr/) :

```text
type(scope): description
type: description
```

- **type**, en minuscules : `feat` (fonctionnalité), `fix` (correction), `exp` (expérience
  d'entraînement), `data` (données, DVC), `docs`, `refactor`, `test`, `ci`, `perf`, `style`
  (mise en forme seule), `chore` (outillage) ;
- **scope** (facultatif) : la partie touchée, en lettres, chiffres, `_`, `-` ou `/` (`api`,
  `frontend`, `detector`, `rag`...) ;
- **description** : au moins 5 caractères, à l'impératif, en anglais comme le code.
- Un `!` après le type ou le scope signale un changement incompatible :
  `feat(api)!: rename the scan endpoint`.

| Message | Résultat |
|---|---|
| `docs: add a note to the README` | accepté |
| `feat(models): add config loader` | accepté |
| `exp(detector): yolo11s at 640 px with mosaic augmentation` | accepté |
| `added a note` | refusé (pas de type) |
| `fix: typo` | refusé (description de moins de 5 caractères) |
| `Docs: update readme` | refusé (type en majuscules) |

Les messages écrits par Git lui-même sont acceptés : `Merge ...`, `Revert "..."`, et les
commits `fixup!` / `squash!` de la section suivante.

### Nettoyer les commits « WIP » avant la fusion

Sur une branche **pas encore poussée**, les commits de travail sont regroupés avant la fusion :

```bash
git commit --fixup <sha>                # corrige un commit précédent de la branche
git rebase -i --autosquash main         # replie les fixup! dans leur commit d'origine
```

Une fois la branche poussée, on ne réécrit plus son historique : on ajoute un commit.

## Tags

Tous les tags sont **annotés** (`git tag -a`) : ils ont un auteur, une date et une note.

**Étapes du projet** : `vX.Y.Z-<étape>`, posé sur le commit de fusion dans `main`.

- `X` change à chaque phase du plan (dépôt, structure, Module 1, Module 2, finalisation) ;
- `Y` à chaque étape d'une phase ; `Z` pour une correction de l'étape.
- La note dit ce que l'étape a apporté. Exemples : `v1.1.0-conventions`, `v3.1.0-detecteurs`.

**Versions des modèles** : `model/<tâche>-<architecture>-vX.Y.Z`, posé sur le commit qui a
produit le modèle (fin de la branche `experiment/...`). Exemple :
`model/detector-yolo11s-v1.2.0`. La note du tag contient :

```text
metrics:   mAP50=<valeur> mAP50-95=<valeur> (split test)
mlflow:    run_id=<id du run>  experiment=tujya-vision
registry:  tujya-pill-detector version <n>
status:    champion
```

Le statut est `champion`, `challenger` ou `archivé`. Un tag ne bouge jamais : quand le statut
change, la référence à jour est l'alias MLflow ci-dessous.

## Modèles : MLflow et alias

Règle suivie par les entraînements (mise en place avec la baseline MLflow du Module 1) :
chaque entraînement est un run MLflow (paramètres, métriques, artefacts, tags `git_commit`,
`dvc_rev`, seed, `requirements.txt` et `dvc.lock`) et une version dans le Model Registry
(`tujya-pill-detector`, `tujya-pill-classifier`).

- L'alias **`champion`** désigne la version servie par l'API ; **`challenger`** la deuxième
  meilleure, candidate au remplacement.
- La promotion est automatique (`promote_if_better`) : une nouvelle version devient
  `champion` seulement si elle bat l'actuelle d'au moins un gain minimal sur la métrique
  choisie ; l'ancien champion devient `challenger`.

Le tag git dit « quel code a produit cette version », l'alias MLflow dit « quelle version est
en service ».

## Données

Les données et les poids sont versionnés par DVC, jamais par git : seuls les fichiers
pointeurs `*.dvc` et `dvc.lock` sont commités. `dvc status` doit être propre à la fin de
chaque étape.

## Hooks

Configuration : [.pre-commit-config.yaml](../.pre-commit-config.yaml). Installation, une fois
par clone :

```bash
pip install pre-commit
pre-commit install --hook-type pre-commit --hook-type commit-msg
npm --prefix frontend/eslint ci     # pour le hook eslint
pre-commit run --all-files              # vérifier tout le dépôt
```

| Hook | Ce qu'il vérifie |
|---|---|
| `ruff`, `ruff-format` | Lint (corrigé automatiquement) et formatage Python ([ruff.toml](../ruff.toml)). |
| `flake8`, `black` | Les outils du guide, en vérification seulement ([.flake8](../.flake8), `[tool.black]` du [pyproject.toml](../pyproject.toml)) : même longueur de ligne que ruff, et `ruff format` donne la mise en page de black. |
| `mypy` | Types du pipeline de données et du corpus du Module 2. |
| `eslint` | Lint du frontend React ([frontend/eslint/](../frontend/eslint/eslint.config.js)). |
| `check-added-large-files` | Refuse un fichier de plus de 500 Ko : les données vont dans DVC. |
| `nbstripout` | Enlève les sorties des notebooks avant le commit. |
| `detect-private-key` | Refuse une clé privée. |
| `check-secrets` | Hook du dépôt ([tools/hooks/check_secrets.py](../tools/hooks/check_secrets.py)) : refuse une clé d'API `sk-...`, une clé AWS `AKIA...` ou un mot de passe en dur de 8 caractères ou plus, avec le fichier et le numéro de ligne. Un faux positif connu porte le commentaire `pragma: allowlist secret`. |
| `commit-msg` | Hook du dépôt ([tools/hooks/check_commit_msg.py](../tools/hooks/check_commit_msg.py)) : la convention de message ci-dessus. |
| `no-secrets-or-data` | Refuse `.env` et les fichiers de `data/` (sauf les pointeurs DVC). |

Exemple de refus :

```text
$ git commit -m "added a note"
commit message follows Conventional Commits, type(scope) description....Failed
- hook id: commit-msg
- exit code: 1

Invalid commit message: 'added a note'
Expected: type(scope): description   or   type: description
```

Les tests des deux hooks du dépôt sont dans [tools/tests/](../tools/tests/test_hooks.py)
(`cd tools && python -m pytest`).

**Les hooks locaux ne suffisent pas** : `git commit --no-verify` les contourne. La CI
(GitHub Actions, job `hooks` de [ci-tests.yml](../.github/workflows/ci-tests.yml)) refait donc côté serveur la vérification des messages de tous
les commits poussés, en plus du lint et des tests de chaque partie du dépôt.

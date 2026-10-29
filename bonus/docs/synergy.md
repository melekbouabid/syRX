# Bonus : synergie des associations de médicaments (DrugComb), sans outil

Question : à partir des données DrugComb déjà collectées par le pipeline SynergyRx
([data_pipeline/](../../data_pipeline/)), un modèle peut-il prédire la synergie d'une
association **comprenant un médicament qu'il n'a jamais vu**, assez bien pour justifier un
outil `predict_synergy` ? La règle a été fixée **avant l'entraînement** (D75) : n'ajouter
l'outil que si le RMSE de test du modèle est au moins 10 % sous celui de la moyenne **et**
si son r de Pearson est d'au moins 0,3, sur un jeu de test qui laisse des médicaments de côté.

**Résultat : non.** Le modèle bat la moyenne (RMSE −7,3 %, r 0,377) mais pas de la marge
exigée : aucun outil n'a été ajouté, et le modèle n'a pas été réglé davantage contre le jeu
de test. La synergie sur lignées de cellules cancéreuses est une donnée de recherche, pas une
information sur les médicaments d'un patient ; Tujya n'en dit rien dans l'application.

## Données (2026-10-28)

`data/parquet/` du pipeline SynergyRx, désormais versionné par DVC (`data/parquet.dvc`,
6 fichiers, 38 Mo, envoyés sur le remote local).

| | Lignes / expériences |
|---|---:|
| Lignes de `synergy_pairs.parquet` | 689 081 (DrugComb 379 670, TDC 286 674, O'Neil 22 737) |
| Expériences distinctes (paire non ordonnée + lignée) | 379 670 |
| Expériences par source | DrugComb + TDC 281 760 ; DrugComb seul 75 173 ; DrugComb + O'Neil 17 823 ; les trois 4 914 |
| Accord avec DrugComb | TDC : r = 1,00 sur Loewe et ZIP (286 674 en commun : une copie) ; O'Neil : r = 0,415 sur Loewe (22 737 en commun ; pas de ZIP) |
| **Gardé : une source par expérience = DrugComb** | 379 670 (chaque expérience est dans DrugComb ; aucune ligne répétée à l'intérieur) |
| Les deux médicaments avec une structure valide (289 médicaments en ont une) | **323 339 expériences, 287 médicaments, 110 lignées** |

**Cible : ZIP** (le score de DrugComb, qui combine les références de Loewe et de Bliss ; sur
les expériences gardées, moyenne −0,77, écart-type 5,55 ; Loewe est asymétrique, moyenne
−9,05, écart-type 14,76). Licences : DrugComb CC BY 4.0 ; les lignes O'Neil ne sont pas
utilisées.

## Modèle

[`synergy/train_synergy.py`](../synergy/train_synergy.py) (venv du backend avec MLflow ; 46 s
sur le CPU du portable) :

- Variables des médicaments : somme des empreintes de Morgan des deux médicaments (rayon 2,
  1 024 bits ; l'ordre de la paire ne compte pas), avec la standardisation de l'outil de
  toxicité (plus grand fragment).
- Variables des lignées : 30 composantes principales de la matrice d'expression standardisée
  (86 lignées ; manquantes pour les autres, ce que LightGBM gère) et le tissu.
- **Split en laissant des médicaments de côté** : 15 % des 287 médicaments tirés au hasard
  réservés au test (chaque expérience contenant au moins l'un d'eux), 10 % de plus pour la
  validation (arrêt anticipé) : 178 307 entraînement / 51 435 validation / 93 597 test
  (86 135 avec un médicament inconnu, 7 462 avec deux).
- Régression LightGBM (63 feuilles, taux d'apprentissage 0,05, fraction de variables 0,5),
  arrêt anticipé à 79 itérations.

| Jeu de test (médicaments laissés de côté) | RMSE de la moyenne | RMSE de la moyenne par lignée (r) | **RMSE de LightGBM (r)** |
|---|---:|---:|---:|
| Tout le test (93 597) | 5,597 | 5,472 (0,215) | **5,189 (0,377)** |
| Un médicament inconnu (86 135) | 5,582 | 5,448 (0,221) | 5,165 (0,381) |
| Deux médicaments inconnus (7 462) | 5,777 | 5,734 (0,142) | 5,455 (0,328) |
| Validation (51 435) | 5,479 | 5,286 (0,263) | 5,075 (0,376) |

Gain de RMSE sur la moyenne pour le test : **7,3 %** (règle : ≥ 10 %) ; Pearson 0,377
(règle : ≥ 0,3). **Règle non atteinte -> pas d'outil `predict_synergy`.**

MLflow : expérience **`tujya-synergy`**, run `lgbm-leave-drug-out-zip`
(`77af603c48c54b06bcdb016286f7de43`) : paramètres, effectifs, toutes les métriques
ci-dessus, étiquette `clearly_better_than_mean_baseline = False`, `results.json`. Aucun
modèle enregistré.

## Pourquoi c'est difficile, et la suite

- Laisser des médicaments de côté est le cadre honnête pour un outil (un utilisateur peut
  demander n'importe quel médicament), et c'est bien plus difficile qu'un split aléatoire où
  les deux médicaments ont été vus à l'entraînement ; les modèles de synergie publiés
  rapportent en général des splits aléatoires ou par paire, qui ne sont pas comparables.
- 323 339 expériences mais seulement 287 médicaments : pour généraliser à un nouveau
  médicament, c'est le nombre de médicaments distincts qui compte, et il est faible.
- Suite si le bonus est poursuivi : variables de cibles ou de réponse en monothérapie au lieu
  des seules empreintes, normalisation par étude, une autre cible (Loewe, ou une classe
  synergique / antagoniste), et un nouveau jeu de médicaments réservés pour les évaluer,
  puisque ce jeu de test a maintenant été regardé.

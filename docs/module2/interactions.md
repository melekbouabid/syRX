# Principes actifs des comprimés et interactions

Comment chaque classe d'identification reçoit son ou ses principes actifs, et comment sont
construites les interactions entre ces principes actifs. Exécuté le 2026-10-28. Code :
[`src/data/corpus/ingredients.py`](../../src/data/corpus/ingredients.py),
[`src/data/corpus/interactions.py`](../../src/data/corpus/interactions.py),
[`src/data/corpus/__main__.py`](../../src/data/corpus/__main__.py).

```
config/pill_classes.csv (liste des NDC11 par classe)
  -> RxNav ndcstatus      NDC11 -> RxCUI du produit (NDC obsolètes compris, par leur historique)
  -> RxNav historystatus  RxCUI -> principes actifs de base (RxNorm IN), dosages retirés
  -> data/processed/pharma/pill_ingredients.csv  + colonnes de principes actifs dans config/pill_classes.csv
  -> DDInter 2.0 (14 fichiers ATC, `python -m src.data.corpus ddinter`)
  -> config/ingredient_aliases.csv (3 noms DDInter différents de RxNorm)
  -> data/processed/pharma/interactions_between_classes.csv, data/processed/pharma/interactions_with_any_drug.csv
```

Commandes (venv du projet `.venv`, depuis la racine du dépôt) :

```bash
python -m src.data.corpus --set http.require_contact_email=false ddinter
python -m src.data.corpus --set http.require_contact_email=false ingredients
python -m src.data.corpus interactions
```

`http.require_contact_email=false` a servi parce qu'aucun `.env` n'existait sur cette machine
(même situation que D1) ; avec `CONTACT_EMAIL` dans `.env`, l'option est inutile.

## Principes actifs (source : RxNav / RxNorm, NLM)

- API : https://lhncbc.nlm.nih.gov/RxNav/APIs/ (`ndcstatus`, `rxcui/{id}/historystatus`,
  `approximateTerm` en secours). Débit limité à 10 requêtes/s (la NLM autorise 20/s par IP).
- Le principe actif de base (`baseName`, par exemple « cyclobenzaprine ») est gardé, pas le
  sel (« cyclobenzaprine hydrochloride »), car DDInter est indexé par le nom de base.
- Les dosages figurent dans la réponse de RxNav mais ne sont **jamais stockés** (aucune
  dose dans Tujya).

Résultat : **149 classes sur 150 associées par leurs NDC, 82 principes actifs distincts** ;
aucune classe n'a eu besoin du secours par le nom. 11 classes sont des associations (par
exemple losartan + hydrochlorothiazide, glyburide + metformine, acide folique + vitamines B12
et B6). Le divalproex et l'acide valproïque correspondent tous deux au « valproate » de RxNorm.

Non associée : la classe 104, « FORMULA B PLUS TAB » (vitamines et minéraux, NDC
00904-7929-60). RxNav ne connaît ce NDC que comme « ALIEN » (source hors RxNorm), sans liste
de principes actifs. La classe reste dans le modèle de vision ; l'application doit dire
qu'aucune information sur les principes actifs n'est disponible.

Colonnes de `data/processed/pharma/pill_ingredients.csv` : class_id, class_key, ndc11, ndc_status
(ACTIVE / OBSOLETE / ALIEN / UNKNOWN), product_rxcui, product_name (nom RxNorm),
ingredient_rxcui, ingredient (RxNorm IN en minuscules), method (`ndc` ou `name`).

## Interactions (source : DDInter 2.0)

- Licence CC BY-NC-SA 4.0 (non commerciale, partage à l'identique). Activée pour ce projet
  académique par l'étape `ddinter`, qui lance le connecteur existant du pipeline avec
  `enabled=true`, `terms_acknowledged=true` et les 14 fichiers. Le
  [`config/data_pipeline/config.yaml`](../../config/data_pipeline/config.yaml) partagé garde DDInter
  désactivé par défaut pour l'exécution `tujya-data update` (deux tests existants fixent
  ce défaut). Pas de redistribution : les tables restent sous `data/`, ignoré par git.
- Citation : Tian Y, et al. DDInter 2.0: an enhanced drug interaction resource with expanded
  data coverage, new interaction types, and improved user interface. Nucleic Acids Res.
  2025;53(D1):D1356-62. doi:10.1093/nar/gkae726
- Par défaut, le connecteur ne téléchargeait que 8 des 14 fichiers ATC (A, B, D, H, L, P, R,
  V). Les six autres (C cardiovasculaire, G, J, M, N système nerveux, S) sont publiés à la
  même adresse et à la même date ; l'étape `ddinter` demande les 14 (234 981 lignes uniques
  au lieu de 160 235).
- Rapprochement des noms : sans tenir compte de la casse, après trois alias de
  `config/ingredient_aliases.csv` (valproate = Valproic acid, vitamin b12 = Cyanocobalamin,
  vitamin b6 = Pyridoxine). **Les 82 principes actifs sont tous trouvés dans DDInter.**

| Table | Paires | Major | Moderate | Minor | Unknown |
|---|---:|---:|---:|---:|---:|
| `interactions_between_classes.csv` (les deux principes actifs appartiennent aux 150 classes) | 2 350 | 125 | 826 | 72 | 1 327 |
| `interactions_with_any_drug.csv` (au moins l'un des deux) | 40 508 | 4 124 | 20 352 | 1 446 | 14 586 |

Colonnes : ingredient_a, ingredient_b (ordre alphabétique, chaque paire n'apparaît qu'une
fois), pair (« a + b »), severity (niveau DDInter : Major, Moderate, Minor, Unknown),
description, conflict (DDInter liste la paire deux fois avec des niveaux différents ; le plus
grave est gardé ; 0 cas ici), ddinter_id_a, ddinter_id_b, source, source_url, license.

**À propos de `description`.** Les fichiers en bloc de DDInter ne contiennent que les deux
médicaments et le niveau ; le mécanisme et la conduite à tenir ne sont que sur les pages web
de DDInter, que ce projet ne moissonne pas. La description reprend donc seulement ce que dit
DDInter (« DDInter 2.0 lists an interaction between fluconazole and warfarin with severity
Major. »). Les explications viennent du corpus RAG (notices openFDA), toujours avec leur
citation. « Unknown » signifie que DDInter enregistre une interaction sans en noter la
gravité ; elle ne doit jamais être présentée comme « pas d'interaction ». Une paire absente
de la table donne « aucune interaction connue dans cette base de données », jamais « sûr ».

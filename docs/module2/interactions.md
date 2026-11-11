# Principes actifs des comprimés et interactions

Comment chaque classe d'identification reçoit son ou ses principes actifs, et comment sont
construites les interactions entre ces principes actifs. Exécuté le 2026-10-28 pour les 150
classes de v1, puis le 2026-11-11 pour les **380 classes de C3PI v2** que sert le classifieur
de l'étape 2.3 (D119 dans [docs/decisions.md](../decisions.md)). Code :
[`src/data/corpus/ingredients.py`](../../src/data/corpus/ingredients.py),
[`src/data/corpus/interactions.py`](../../src/data/corpus/interactions.py),
[`src/data/corpus/__main__.py`](../../src/data/corpus/__main__.py).

```
config/splits/c3pi_v2/classes.csv (380 classes v2)
  -> config/pill_classes.csv (`python -m src.data.corpus classes` : identifiants v2, NDC11 par classe)
  -> RxNav ndcstatus      NDC11 -> RxCUI du produit (NDC obsolètes compris, par leur historique)
  -> RxNav historystatus  RxCUI -> principes actifs de base (RxNorm IN), dosages retirés
  -> data/processed/pharma/pill_ingredients.csv  + colonnes de principes actifs dans config/pill_classes.csv
  -> DDInter 2.0 (14 fichiers ATC, `python -m src.data.corpus ddinter`)
  -> config/ingredient_aliases.csv (11 noms DDInter différents de RxNorm)
  -> data/processed/pharma/interactions_between_classes.csv, data/processed/pharma/interactions_with_any_drug.csv
```

Commandes (venv du projet `.venv`, depuis la racine du dépôt) :

```bash
python -m src.data.corpus ddinter
python -m src.data.corpus classes        # 380 classes v2 (D119)
python -m src.data.corpus ingredients
python -m src.data.corpus interactions
.venv-train/Scripts/python -m dvc add data/processed/pharma
cd backend && ../.venv/Scripts/python -m app.pharma_loader   # recharge les tables de l'API
```

En octobre, `--set http.require_contact_email=false` a servi parce qu'aucun `.env` n'existait sur
cette machine (même situation que D1) ; le 11/11, `CONTACT_EMAIL` est lu dans `.env` par le
client poli du pipeline (jamais écrit dans un fichier suivi). La table des 150 classes de v1
reste dans `config/pill_classes_v1.csv` : les étapes qui décrivent le jeu v1 (sélection,
split Spark, crops de comparaison de l'étape 2.3) la lisent toujours.

## Principes actifs (source : RxNav / RxNorm, NLM)

- API : https://lhncbc.nlm.nih.gov/RxNav/APIs/ (`ndcstatus`, `rxcui/{id}/historystatus`,
  `approximateTerm` en secours). Débit limité à 10 requêtes/s (la NLM autorise 20/s par IP).
- Le principe actif de base (`baseName`, par exemple « cyclobenzaprine ») est gardé, pas le
  sel (« cyclobenzaprine hydrochloride »), car DDInter est indexé par le nom de base.
- Les dosages figurent dans la réponse de RxNav mais ne sont **jamais stockés** (aucune
  dose dans Tujya).

Résultat sur les **380 classes de v2 (11/11) : 379 associées par leurs NDC, 176 principes actifs
distincts, 30 associations** ; les 149 classes déjà associées en v1 retrouvent exactement les
mêmes principes actifs. La seule classe sans principe actif est la même qu'en v1 (ci-dessous,
classe 90 en v2).

Résultat de v1 (28/10) : **149 classes sur 150 associées par leurs NDC, 82 principes actifs distincts** ;
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
- Rapprochement des noms : sans tenir compte de la casse, après les alias de
  `config/ingredient_aliases.csv` (valproate = Valproic acid, vitamin b12 = Cyanocobalamin,
  vitamin b6 = Pyridoxine ; ajoutés pour v2 : nom USAN contre DCI, par exemple alendronate =
  Alendronic acid, norethindrone = Norethisterone, ethinyl estradiol = Ethinylestradiol).
  v1 : les 82 principes actifs sont tous trouvés dans DDInter. **v2 : 172 sur 176** ; absents :
  `inert ingredients` (excipient listé par RxNorm), `sodium fluoride`, `iron carbonyl` et
  `polysaccharide iron complex` (DDInter n'a qu'une entrée « Iron » et un nom DDInter ne peut
  désigner qu'un seul principe actif : pas de choix arbitraire, ces comprimés donnent « aucune
  interaction connue dans cette base de données »).

| Table | Paires | Major | Moderate | Minor | Unknown |
|---|---:|---:|---:|---:|---:|
| `interactions_between_classes.csv` (les deux principes actifs appartiennent aux 380 classes) | 8 611 | 689 | 3 338 | 308 | 4 276 |
| `interactions_with_any_drug.csv` (au moins l'un des deux) | 75 653 | 8 707 | 40 207 | 2 993 | 23 746 |
| v1 (150 classes) : entre classes | 2 350 | 125 | 826 | 72 | 1 327 |
| v1 (150 classes) : avec tout médicament | 40 508 | 4 124 | 20 352 | 1 446 | 14 586 |

Le corpus RAG (`data/processed/rag/corpus`, étape DVC `rag_corpus`, gelée) reste celui des 150
classes de v1 : ses paires DDInter et ses notices ne couvrent que les 82 principes actifs de v1
(D119). L'outil `check_interaction` et l'écran Vérifier lisent les tables ci-dessus, donc les
380 classes ; les notices et résumés des nouveaux principes actifs viendront avec la collecte
RAG v2 (D90).

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

# Fine-tuning du modèle d'embedding sur Kaggle (étape 3.5)

Option des consignes du module 2 (« fine-tuning ou entraînement de small modèles »), choisie
pour renforcer le module (remarque 15 du professeur). Le modèle affiné est celui qui sert
aujourd'hui l'agent : **bge-m3** (BAAI, 568 M de paramètres, XLM-RoBERTa large, multilingue),
qui transforme les questions (français ou anglais) et les passages du corpus (anglais) en
vecteurs pour la recherche dans Qdrant. L'entraînement se fait sur le GPU gratuit de Kaggle
(T4, 16 Go) ; le PC (RTX 2050, 4 Go) ne sert qu'aux tests de fumée. Décision : D121 dans
[docs/decisions.md](../decisions.md).

**État (14/11/2026) : FAIT, modèle non servi.** Run Kaggle (T4, 152 min) importé dans MLflow,
poids versionnés par DVC, modèle comparé dans la grille de l'étape 3.3 (section « Résultats ») :
meilleur MRR (+0,083, significatif) mais couverture du top 10 plus faible et gain concentré sur
les documents vus à l'entraînement ; il n'est pas servi (D123) et cette branche reste la trace
de l'expérience.

## Pourquoi l'embedding (et pas le reranker ni un LLM)

- C'est le maillon qui décide de ce que le LLM lit : un passage absent des k premiers résultats
  ne peut pas être cité. Le benchmark de l'étape 3.2 montre encore des pertes sur les questions
  en français (le corpus est en anglais).
- Le reranker (bge-reranker-base) coûte 13 à 18 s par question sur le CPU de ce PC (étape 3.2) :
  l'améliorer ne change rien à la configuration servie, qui s'en passe.
- Un LoRA de qwen2.5:3b demanderait des réponses de référence écrites à la main et un nouvel
  export GGUF pour Ollama ; le gain se mesurerait mal avec 96 questions.
- L'embedding a une métrique directe et reproductible (Recall@k, MRR sur des paires dont on
  connaît le passage), et le même benchmark de récupération (étape 3.3) compare avant / après.

## Données : paires question / passage sans fuite

Le corpus n'a que des passages ; [src/models/embedding/pairs.py](../../src/models/embedding/pairs.py)
écrit une question pour chacun, sans LLM (reproductible) :

| Passage | Question | Exemple |
|---|---|---|
| Chunk de notice openFDA (section S du médicament M) | une question en anglais **et** une en français par section, formulée autrement que celles du benchmark | « How does metformin work in the body? » / « Comment agit metformine dans le corps ? » |
| Paire DDInter A + B | une question, en anglais ou en français (alternés) | « Is there an interaction between allopurinol and amiloride? » |
| Résumé PubMed | le titre de l'article (premier chunk) | « Lactic acidosis during metformin therapy... » |

Chaque paire a un **négatif difficile** : un autre chunk du même principe actif (autre section
de la notice, autre paire, autre article), pour apprendre à distinguer des passages proches.

**Aucune fuite du benchmark** (`src/benchmark/rag_questions.jsonl`, 96 questions) :

- aucun chunk attendu par une question du benchmark (332) n'est utilisé, ni comme positif ni
  comme négatif ;
- les articles PubMed et les paires DDInter attendus sont exclus en entier (34 documents) ; une
  notice ne perd que les sections attendues (les chunks attendus d'une section sont tous ceux de
  la section) ;
- aucune question d'entraînement n'est égale à une question du benchmark (après normalisation) ;
- les questions des sections sont formulées autrement que celles du benchmark
  (« When should X not be used? » et non « What are the contraindications of X? »).

`check_no_leak` vérifie ces règles à chaque construction (et un test les vérifie sur un mini
corpus). Les documents sont répartis 90 / 10 (hash de l'identifiant) : la validation porte sur
des documents jamais vus à l'entraînement.

Résultat (étape DVC `finetune_pairs`, sortie `data/processed/rag/finetune`, jamais dans git) :

| | Paires | Documents | Notices | DDInter | PubMed | Anglais | Français |
|---|---:|---:|---:|---:|---:|---:|---:|
| Entraînement | 8 077 | 3 565 | 4 584 | 2 094 | 1 399 | 4 759 | 3 318 |
| Validation | 1 007 | 393 | 624 | 231 | 152 | 587 | 420 |

8 312 chunks utilisables sur 8 644 (17 chunks rétractés et 332 chunks du benchmark retirés).

## Entraînement ([src/models/embedding/train.py](../../src/models/embedding/train.py), [config/finetune/bge-m3.yaml](../../config/finetune/bge-m3.yaml))

- sentence-transformers 5.1.2, perte **CachedMultipleNegativesRankingLoss** : chaque question
  doit classer son passage avant son négatif difficile et avant tous les autres passages du lot
  (63 négatifs du lot + 64 négatifs difficiles) ; le cache (GradCache) garde un lot de 64 en
  calculant les gradients 8 séquences à la fois, dans la mémoire du T4 ;
- 2 époques, taux d'apprentissage 1e-5 (petit : le modèle est déjà bon et doit garder son
  multilinguisme), préchauffage 10 %, FP16, longueur 384 jetons, graine 0, lots sans doublon de
  question ;
- **mémoire** : le fine-tuning complet de bge-m3 demande environ 9 Go (poids, gradients et états
  AdamW en FP32) plus les activations d'un mini-lot. Si le T4 manque de mémoire, l'entraînement
  repart automatiquement avec les 16 premières couches (sur 24) et les embeddings gelés : seules
  les 8 couches hautes apprennent. `run.json` et le tag MLflow `frozen_layers` disent quel mode
  a tourné ;
- avant et après : recherche des 1 007 questions de validation dans **tout** le corpus
  (8 644 passages), Recall@1/5/10, MRR@10, nDCG@10 ([evaluate.py](../../src/models/embedding/evaluate.py) ;
  les chunks d'une même section partagent leur question et comptent tous comme pertinents) ;
- export : modèle en FP16 (safetensors, environ 1,1 Go), historique, `run.json` (configuration,
  environnement, commit, métriques avant / après, durées).

## Mode d'emploi pas à pas

Rien de tout cela ne demande de mot de passe ni de jeton.

### 1. Construire et empaqueter les paires (PC, déjà fait)

```bash
# venv du projet, depuis la racine, branche experiment/fine-tuning-embedding
.venv-train/Scripts/python -m dvc repro finetune_pairs
.venv/Scripts/python -m src.models.embedding.package_kaggle
```

Sortie : **`runs/kaggle/tujya-rag-finetune-pairs.zip` (4,7 Mo)** : `train.jsonl`, `val.jsonl`,
`corpus.jsonl` (les 8 644 passages tels qu'ils sont indexés) et `MANIFEST.json` (comptes, md5
de chaque fichier, commit).

### 2. Créer le dataset Kaggle privé

Kaggle > **Datasets** > **New Dataset** : déposer `tujya-rag-finetune-pairs.zip`, titre « Tujya
RAG fine-tuning pairs », identifiant `tujya-rag-finetune-pairs`, visibilité **Private**.

### 3. Lancer le notebook

Kaggle > **Code** > **New Notebook** > **File** > **Import Notebook** > onglet GitHub,
`melekbouabid/syRX`, branche `experiment/fine-tuning-embedding`, fichier
`notebooks/kaggle/finetune_bge-m3.ipynb`. **Settings** : **Accelerator** GPU T4 x1,
**Internet** activé (clone, `pip install` des versions figées, téléchargement de bge-m3 depuis
Hugging Face, sans compte). **Add Input** : le dataset `tujya-rag-finetune-pairs`. Puis
**Save Version** > **Save & Run All (Commit)**.

**Durée GPU estimée : environ 1 h (2 h au plus).** Calcul : 8 077 paires en lots de 64 =
127 pas par époque, 254 pas pour 2 époques ; chaque pas lit 64 questions courtes et 128
passages (au plus 384 jetons) trois fois (passe sans gradient, passe avec gradient, rétro-
propagation, cache GradCache), soit environ 8 s sur un T4 en FP16 pour XLM-RoBERTa large :
environ 35 min. S'y ajoutent l'installation et le téléchargement du modèle (2,2 Go, environ
5 min), deux encodages du corpus (8 644 passages, environ 3 min chacun) et l'export (2 min).
Si le mode « couches hautes » se déclenche, l'entraînement est plus rapide (moins de
gradients). C'est bien en dessous du quota gratuit (environ 30 h par semaine) et de la limite
de 12 h par session.

### 4. Récupérer et importer l'export (PC)

Onglet **Output** de la version sauvegardée : télécharger `finetune_bge-m3_export.zip` dans
`runs/kaggle/`, puis :

```bash
docker compose up -d mlflow
# Windows : PYTHONIOENCODING=utf-8 évite une erreur d'affichage de MLflow (emoji) dans la console
PYTHONIOENCODING=utf-8 .venv-train/Scripts/python -m src.models.embedding.import_kaggle runs/kaggle/finetune_bge-m3_export.zip
git switch experiment/fine-tuning-embedding
.venv-train/Scripts/python -m dvc add data/models/embeddings/finetune-bge-m3
git add data/models/embeddings/finetune-bge-m3.dvc
git commit -m "exp(embedding): record the bge-m3 Kaggle fine-tuning (val recall@5 0.xxxx)"
```

L'import crée un run dans l'expérience MLflow **`tujya-rag-finetuning`** : configuration en
paramètres, métriques `base_*` et `tuned_*` (avant / après), courbe de perte (`train_loss` par
pas et `loss.png`), `run.json`, tags de provenance (`git_commit` du PC, `kaggle_git_commit` et
`kaggle_git_branch` du code qui a tourné, `dvc_rev`, `dataset_md5` des paires, GPU,
`frozen_layers`, `smoke`). Le modèle est dézippé dans `data/models/embeddings/finetune-bge-m3/`
(DVC, jamais git) et un résumé est écrit dans `models/compare/embedding_finetune-bge-m3.json`.

### 5. Comparer dans le benchmark (étape 3.3)

Le modèle affiné entre dans la grille de l'étape 3.3 comme une configuration d'embedding de plus
(groupe `bge-m3-finetuned` de [config/rag_grid.yaml](../../config/rag_grid.yaml), lu en local par
sentence-transformers) et il est mesuré sur les 85 questions du benchmark qui ont des chunks
attendus, qu'il n'a jamais vues :

```bash
# GPU libre : arrêter d'abord les modèles Ollama
.venv-train/Scripts/python -m src.benchmark.encode_st --model BAAI/bge-m3 --slug bge-m3-st
.venv-train/Scripts/python -m src.benchmark.encode_st --model data/models/embeddings/finetune-bge-m3/model --slug bge-m3-finetuned
python -m src.benchmark.bench_grid run --group bge-m3-finetuned     # venv du projet
```

## Résultats (14/11/2026)

Run Kaggle : Tesla T4, 2 époques sur 8 077 paires (128 pas), 152 min, aucune couche gelée ;
run MLflow `03c72c9c` (`tujya-rag-finetuning`) ; poids : `data/models/embeddings/finetune-bge-m3.dvc`
(1,16 Go, DVC). Paires de validation (gabarits, 1 007) : Recall@5 0,930 -> 0,999, MRR@10 0,784 ->
0,960.

Benchmark de l'application (85 questions, filtre par principe actif, top 10 ; même chemin
sentence-transformers avant et après : le bge-m3 d'origine y redonne le MRR d'Ollama, 0,833
contre 0,832) :

| Configuration | R@1 | R@5 | R@10 | MRR | nDCG@10 | Coverage@10 | Run |
|---|---:|---:|---:|---:|---:|---:|---|
| bge-m3 d'origine, dense | 0,776 | 0,906 | 0,953 | 0,833 | 0,781 | 0,810 | `e5300cff` |
| **bge-m3 fine-tuné, dense** | **0,906** | 0,929 | 0,941 | **0,916** | 0,746 | 0,699 | `1b6da11e` |
| bge-m3 d'origine, hybride + réécriture | 0,753 | 0,906 | 0,988 | 0,827 | 0,782 | 0,826 | `9c2504bf` |
| bge-m3 fine-tuné, hybride + réécriture | 0,859 | 0,965 | 0,988 | 0,898 | 0,780 | 0,769 | `d96fc302` |

Bootstrap apparié (dense, après moins avant, intervalle à 95 %) : MRR +0,083 (+0,025 à +0,146),
Recall@1 +0,129 (+0,059 à +0,212), Recall@5 +0,024 (-0,024 à +0,071), Coverage@10 -0,111
(-0,175 à -0,049).

**Fuite** : aucun chunk attendu, aucune question identique ni proche (Jaccard des mots >= 0,8)
dans les 9 084 paires. Mais **75 des 85 questions** visent une notice dont d'autres sections ont
servi à l'entraînement : MRR dense 0,848 -> 0,960 sur elles, 0,717 -> 0,583 sur les 10 autres.

**Décision (D123)** : modèle **non servi**. Le gain vient des documents appris et n'est pas
démontré au-delà ; la couverture du contexte donné au LLM baisse (significatif) ; le servir
demanderait une conversion GGUF pour Ollama et la réindexation des 8 661 chunks. Le réglage servi
reste bge-m3 d'origine (dense + réécriture, champion de la grille 3.3). Tag
`model/embedding-bge-m3-finetuned-v1.0.0` (statut : archivé, non servi). Détails :
[benchmark_retrieval.md](benchmark_retrieval.md#bge-m3-fine-tuné-étape-35).

## Vérification locale (test de fumée)

Le notebook a été exécuté sur le PC (RTX 2050, 4 Go) avec `TUJYA_SMOKE=1` : un petit modèle
multilingue de la même famille (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`,
118 M de paramètres), 64 paires, 16 questions de validation, un corpus réduit à 239 passages,
une époque. Ce test vérifie la chaîne (lecture du dataset, évaluation avant, entraînement avec
négatifs difficiles, évaluation après, export FP16, `run.json`, zip, import MLflow avec
`--smoke`) ; ses chiffres n'ont aucune valeur.

Résultat du 12/11/2026 (RTX 2050 4 Go, notebook exécuté par nbclient avec `TUJYA_SMOKE=1`
depuis le commit `ff0d973`, dataset empaqueté depuis le même commit) :

| Étape | Résultat |
|---|---|
| Durée totale du notebook | 1,7 min (modèle déjà en cache) |
| Données | 64 paires, 16 questions de validation, 239 passages |
| Avant (petit modèle) | Recall@1 0,750, Recall@5 1,000, MRR@10 0,841 |
| Après une époque (4 pas) | Recall@1 0,875, Recall@5 1,000, MRR@10 0,919 |
| Export | `finetune_bge-m3_export.zip`, 221 Mo (modèle FP16, `run.json`) |
| Import MLflow | run `smoke-finetune-bge-m3-smoke` (`c2aeadc7`) dans `tujya-rag-finetuning`, tags `smoke=true`, `git_commit`, `dvc_rev`, `dataset_md5`, `kaggle_git_branch`, `frozen_layers=0` |

Le repli « couches hautes » a aussi été vérifié en simulant un manque de mémoire au premier
essai : le second essai part avec 6 couches gelées sur 12 (`fallback_frozen_layers` du bloc
smoke) et `run.json` porte `frozen_layers: 6`, `fallback: true`. Un premier import du run smoke
s'est arrêté sur une erreur d'affichage de la console Windows (cp1252) : ce run (`acd2130e`) est
marqué `FAILED` avec le tag `superseded_by`.

## Limites

- Questions écrites par gabarits, pas par des utilisateurs : le modèle apprend le vocabulaire du
  corpus et l'association question / section, pas la diversité de vraies questions.
- Une seule graine, une seule configuration (pas de recherche d'hyperparamètres : un run
  coûte environ une heure de T4) ; la comparaison décisive est celle du benchmark 3.3.
- L'agent appelle bge-m3 par Ollama ; servir le modèle affiné demanderait soit une conversion
  GGUF pour Ollama, soit un service d'embedding local (sentence-transformers).
- Paires tirées des mêmes notices que le benchmark (sections différentes) : un prochain essai
  devrait tenir à l'écart des documents entiers et garder plusieurs positifs par question (la
  perte à un seul positif fait baisser la couverture du top 10).

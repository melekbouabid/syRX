# Benchmark de la récupération : grille de configurations et MLflow (étape 3.3)

La récupération (retrieval) du RAG est comparée sur une grille de configurations : modèles
d'embedding, modes de recherche, découpage du corpus, réécriture de la requête et rerankers.
Chaque configuration est un run MLflow (expérience **`tujya-rag-retrieval`**) ; la meilleure
configuration servable devient la version `champion` du modèle enregistré
**`tujya-rag-retrieval`** et fixe le réglage de l'agent. Décision : D123
([decisions.md](../decisions.md)). Les mesures plus anciennes (étapes 3.1 et 3.2) restent dans
[benchmark.md](benchmark.md).

## Protocole

- **Questions** : les 85 questions du jeu ([rag_questions.jsonl](../../src/benchmark/rag_questions.jsonl))
  qui ont des chunks attendus (42 en français, 43 en anglais).
- **Recherche comme l'agent** : les principes actifs trouvés dans la question servent de filtre
  de métadonnées (`names`, comme `search_documents` dans [tools.py](../../src/agent/tools.py)) ;
  10 résultats gardés.
- **Métriques** (top 10) : Recall@k (k = 1, 3, 5, 10 : au moins un chunk attendu dans les k
  premiers), MRR@10 (1 / rang du premier chunk attendu), nDCG@10 (pertinence binaire),
  Coverage@10 (part des chunks attendus retrouvés), par langue aussi. Le détail par question est
  un artefact de chaque run (`per_question/<configuration>.json`).
- **Temps par question** : recherche (encodage de la requête, Qdrant, reranker) mesurée pendant
  le run, plus la médiane du temps de réécriture par qwen2.5:3b (326 ms) quand la configuration
  réécrit. Les configurations retenues pour le choix sont **remesurées avec le protocole D96**
  (`bench_grid latency` : 12 questions, 3 tours dans un ordre décalé, pause de 10 s avant chaque
  configuration, médiane des médianes), car ce portable ralentit sous charge continue.
- **Découpage** : une variante (`chunk` autre que 1 500 caractères / 1 phrase de recouvrement) a
  d'autres identifiants de chunks ; ses chunks pertinents sont ceux qui viennent du même document
  et de la même section qu'un chunk attendu et partagent au moins la moitié de leur texte
  ([rechunk.py](../../src/benchmark/rechunk.py), testé).
- **Embeddings sans version Ollama** (multilingual-e5-base, all-MiniLM-L6-v2, bge-m3 fine-tuné) :
  chunks et requêtes encodés une fois sur le GPU par [encode_st.py](../../src/benchmark/encode_st.py)
  (vecteurs versionnés par DVC : `data/processed/rag/bench/vectors.dvc`), chargés dans une
  collection Qdrant à part et cherchés exactement comme l'index servi ; le temps d'encodage d'une
  requête sur le CPU (l'API n'a pas de GPU) est ajouté. `bge-m3-st` est le bge-m3 d'origine par ce
  même chemin : il redonne le bge-m3 d'Ollama (MRR 0,833 contre 0,832), ce qui rend la comparaison
  avant / après du fine-tuning équitable.
- **Règle de choix (D123), écrite avant les résultats** dans [config/rag_grid.yaml](../../config/rag_grid.yaml) :
  meilleur MRR parmi les configurations que l'API peut servir (embedding calculé par Ollama :
  l'image du backend n'a pas PyTorch) et qui tiennent **2 s par question** sur ce PC ; le
  deuxième est `challenger`. Une configuration sur un autre découpage n'est gardée que si son gain
  sur la meilleure configuration au découpage servi est significatif (bootstrap apparié sur les
  questions, intervalle à 95 % au-dessus de 0), car elle demande de reconstruire l'index servi.

## Résultats

Une branche `experiment/rag-<groupe>` par groupe, fusionnées dans `experiment/rag-retrieval`.
Tableau produit par `python -m src.benchmark.bench_grid table` ; « (D96) » = temps remesuré avec
le protocole D96, sinon temps du run. Run = identifiant MLflow.

| Groupe | Configuration | R@1 | R@5 | R@10 | MRR | nDCG@10 | MRR fr | MRR en | ms / question | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| bge-m3 | bge-m3-dense | 0.776 | 0.906 | 0.953 | 0.832 | 0.780 | 0.747 | 0.915 | 63 (D96) | `9d0235cb` |
| bge-m3 | bge-m3-hybrid | 0.741 | 0.882 | 0.953 | 0.808 | 0.767 | 0.683 | 0.930 | 70 (D96) | `618c6e50` |
| bge-m3 | bge-m3-bm25 | 0.518 | 0.753 | 0.776 | 0.611 | 0.609 | 0.380 | 0.837 | 11 | `d2dc4287` |
| bge-m3 | bge-m3-dense-rewrite | 0.765 | 0.941 | 0.988 | 0.834 | 0.776 | 0.767 | 0.898 | 389 (D96) | `f9954762` |
| bge-m3 | bge-m3-hybrid-rewrite | 0.753 | 0.906 | 0.965 | 0.824 | 0.778 | 0.717 | 0.930 | 394 (D96) | `3ae434b6` |
| multilingual-e5-base | multilingual-e5-base-dense | 0.729 | 0.882 | 0.929 | 0.800 | 0.693 | 0.771 | 0.828 | 207 | `ab0857f9` |
| multilingual-e5-base | multilingual-e5-base-hybrid | 0.682 | 0.882 | 0.941 | 0.771 | 0.697 | 0.652 | 0.888 | 210 | `7c5e414c` |
| multilingual-e5-base | multilingual-e5-base-hybrid-rewrite | 0.718 | 0.906 | 0.988 | 0.808 | 0.760 | 0.736 | 0.878 | 534 | `72410c2b` |
| all-minilm-l6-v2 | all-minilm-l6-v2-dense | 0.435 | 0.706 | 0.812 | 0.556 | 0.486 | 0.444 | 0.665 | 33 | `1c159956` |
| all-minilm-l6-v2 | all-minilm-l6-v2-hybrid | 0.647 | 0.835 | 0.894 | 0.723 | 0.649 | 0.587 | 0.857 | 35 | `188b0912` |
| all-minilm-l6-v2 | all-minilm-l6-v2-hybrid-rewrite | 0.718 | 0.882 | 0.977 | 0.795 | 0.714 | 0.733 | 0.857 | 351 | `f93b84e7` |
| chunking | bge-m3-dense-c800-o1 | 0.812 | 0.941 | 0.953 | 0.858 | 0.753 | 0.824 | 0.890 | 68 (D96) | `5a81cae8` |
| chunking | bge-m3-hybrid-c800-o1-rewrite | 0.812 | 0.941 | 0.988 | 0.868 | 0.780 | 0.806 | 0.929 | 389 (D96) | `750b72a6` |
| chunking | bge-m3-dense-c2500-o1 | 0.729 | 0.894 | 0.965 | 0.811 | 0.788 | 0.741 | 0.879 | 44 | `d623e157` |
| chunking | bge-m3-hybrid-c2500-o1-rewrite | 0.788 | 0.941 | 1.000 | 0.853 | 0.810 | 0.774 | 0.930 | 386 | `1136acbc` |
| chunking | bge-m3-hybrid-c1500-o0-rewrite | 0.800 | 0.929 | 0.977 | 0.853 | 0.797 | 0.765 | 0.939 | 393 | `ea493a1b` |
| chunking | bge-m3-hybrid-c1500-o2-rewrite | 0.800 | 0.929 | 0.965 | 0.852 | 0.777 | 0.778 | 0.923 | 385 | `4b4660c1` |
| rerankers | bge-m3-hybrid-rewrite-rerank-bge-reranker-base-f10 | 0.800 | 0.941 | 0.977 | 0.863 | 0.799 | 0.843 | 0.883 | 15616 | `bd1f7d69` |
| rerankers | bge-m3-hybrid-rewrite-rerank-minilm-l-6-v2-f10 | 0.859 | 0.941 | 0.977 | 0.896 | 0.822 | 0.873 | 0.918 | 3617 (D96) | `0289f558` |
| rerankers | bge-m3-hybrid-rewrite-rerank-minilm-l-6-v2-f20 | 0.871 | 0.941 | 0.988 | 0.905 | 0.843 | 0.893 | 0.917 | 5543 | `2aa2baa6` |
| rerankers | bge-m3-hybrid-rewrite-rerank-minilm-l-12-v2-f20 | 0.894 | 0.965 | 1.000 | 0.922 | 0.847 | 0.897 | 0.947 | 11251 (D96) | `254574f3` |
| rerankers | bge-m3-hybrid-rewrite-rerank-jina-reranker-v1-turbo-en-f20 | 0.859 | 0.953 | 0.977 | 0.897 | 0.849 | 0.877 | 0.917 | 7278 | `87b5b8c4` |
| rerankers | bge-m3-dense-rerank-minilm-l-6-v2-f10 | 0.824 | 0.918 | 0.953 | 0.864 | 0.791 | 0.776 | 0.950 | 3149 (D96) | `e7256050` |
| bge-m3-finetuned | bge-m3-st-dense | 0.776 | 0.906 | 0.953 | 0.833 | 0.781 | 0.748 | 0.915 | 445 | `e5300cff` |
| bge-m3-finetuned | bge-m3-st-hybrid-rewrite | 0.753 | 0.906 | 0.988 | 0.827 | 0.782 | 0.733 | 0.918 | 770 | `9c2504bf` |
| bge-m3-finetuned | bge-m3-finetuned-dense | 0.906 | 0.929 | 0.941 | 0.916 | 0.746 | 0.872 | 0.958 | 393 | `1b6da11e` |
| bge-m3-finetuned | bge-m3-finetuned-hybrid-rewrite | 0.859 | 0.965 | 0.988 | 0.898 | 0.780 | 0.852 | 0.942 | 718 | `d96fc302` |

Deux runs de reranker avec `fetch_k` 5 ont été retirés (tag `withdrawn` dans MLflow) : le
retriever lit `max(k, fetch_k)` candidats et la grille garde un top 10, ils répétaient donc
`fetch_k` 10 (mêmes scores). La grille refuse maintenant ce cas (test ajouté).

### Ce que montre la grille

- **Embedding** : bge-m3 reste le meilleur (MRR dense 0,832) devant multilingual-e5-base (0,800)
  et all-MiniLM-L6-v2 (0,556, anglais seulement : 0,444 sur les questions françaises). MiniLM ne
  devient correct qu'avec BM25 et la réécriture en anglais (0,795).
- **Mode** : avec bge-m3, le dense bat l'hybride (0,832 contre 0,808) parce que BM25 ne lit que
  l'anglais (MRR BM25 0,380 en français, 0,837 en anglais), comme en D65.
- **Réécriture** : elle améliore surtout le rappel (dense : Recall@5 0,906 -> 0,941, Recall@10
  0,953 -> 0,988, Recall@5 en français 0,857 -> 0,929) pour un MRR presque égal (0,834).
- **Découpage** : des chunks plus courts (800 caractères) placent mieux le premier bon chunk
  (dense 0,858, hybride + réécriture 0,868) mais le gain sur le réglage servi n'est pas
  significatif : +0,034 de MRR, intervalle à 95 % de -0,026 à +0,099 (+0,024, de -0,014 à +0,065
  pour le dense). Les chunks de 2 500 caractères couvrent mieux (nDCG 0,810) mais placent moins
  bien (0,853). Le recouvrement (0, 1 ou 2 phrases) ne change presque rien (0,852 à 0,853).
- **Rerankers** (sur hybride + réécriture, sauf mention) : ce sont eux qui gagnent le plus, mais
  sur le processeur de ce portable un cross-encoder coûte 2 s pour 10 chunks de ~400 tokens
  (mesuré : 209 GFLOPS en calcul matriciel, 4 threads ONNX au mieux) :

| Reranker | Candidats | MRR | Recall@1 | Temps D96 |
|---|---:|---:|---:|---:|
| aucun (hybride + réécriture) | - | 0,824 | 0,753 | 394 ms |
| ms-marco-MiniLM-L-6-v2 (dense, sans réécriture) | 10 | 0,864 | 0,824 | 3 149 ms |
| ms-marco-MiniLM-L-6-v2 | 10 | 0,896 | 0,859 | 3 617 ms |
| ms-marco-MiniLM-L-6-v2 | 20 | 0,905 | 0,871 | ~5,5 s (run) |
| jina-reranker-v1-turbo-en | 20 | 0,897 | 0,859 | ~7,3 s (run) |
| **ms-marco-MiniLM-L-12-v2** | 20 | **0,922** | **0,894** | 11 251 ms |
| bge-reranker-base (réglage de l'étape 3.2) | 10 | 0,863 | 0,800 | ~15,6 s (run) |

  bge-reranker-base, multilingue et le plus lourd, est ici le moins bon : la requête réécrite est
  en anglais et les petits modèles ms-marco, entraînés sur de l'anglais, s'en sortent mieux (le
  0,889 de l'étape 3.2 venait d'un autre tirage des réécritures ; celles de la grille sont figées
  dans `queries.json`, versionné par DVC).

### Choix du réglage de l'agent

| Statut | Configuration | MRR | Recall@5 | Temps D96 | Registry |
|---|---|---:|---:|---:|---|
| **champion** | bge-m3, dense, réécriture, filtre `names` | 0,834 | 0,941 | 389 ms | `tujya-rag-retrieval` v1 |
| challenger | bge-m3, dense, filtre `names` | 0,832 | 0,906 | 63 ms | `tujya-rag-retrieval` v2 |
| meilleur sans contrainte de temps | hybride + réécriture + MiniLM-L12 (20) | 0,922 | 0,965 | 11 251 ms | non enregistré |

Chaque version enregistrée est un modèle pyfunc dont l'artefact est la configuration (sa
prédiction renvoie les réglages de recherche) ; la description donne le MRR, le temps et le budget.
[`src/benchmark/grid/champion.json`](../../src/benchmark/grid/champion.json) garde le champion, le
challenger et les réglages de l'agent qu'il implique ; un test vérifie que les valeurs par défaut
de [backend/app/config.py](../../backend/app/config.py) les suivent. **Réglage servi** : recherche
dense bge-m3, filtre par principe actif, **réécriture de la requête activée**
(`TUJYA_RAG_QUERY_REWRITE=true` par défaut), sans reranker.

Compromis : une réponse complète prend en moyenne 4,7 s sur ce PC ([benchmark.md](benchmark.md)) ;
la réécriture ajoute ~0,3 s pour +0,035 de Recall@5 (+0,07 sur les questions françaises), le
reranker le plus rapide ajouterait 3,2 s (+70 %) pour +0,06 de MRR. Le reranker reste donc une
option : `TUJYA_RAG_RERANK=true` utilise maintenant **ms-marco-MiniLM-L-6-v2 sur 10 candidats**
(meilleur et 4 fois plus rapide que bge-reranker-base ici) ; sur une machine plus rapide ou avec
un GPU libre, MiniLM-L12 sur 20 candidats est le meilleur mesuré.

## bge-m3 fine-tuné (étape 3.5)

Le modèle affiné sur Kaggle (D121 ; run MLflow `03c72c9c` dans `tujya-rag-finetuning`, poids par
DVC sur la branche `experiment/fine-tuning-embedding`) passe sur ses paires de validation de
0,784 à 0,960 de MRR@10. Ces paires sont écrites par gabarits, donc plus faciles : c'est le
benchmark des 85 questions qui tranche (groupe `bge-m3-finetuned`, même chemin
sentence-transformers que `bge-m3-st`).

| Configuration | R@1 | R@5 | R@10 | MRR | nDCG@10 | Coverage@10 |
|---|---:|---:|---:|---:|---:|---:|
| bge-m3 d'origine, dense | 0,776 | 0,906 | 0,953 | 0,833 | 0,781 | 0,810 |
| **bge-m3 fine-tuné, dense** | **0,906** | 0,929 | 0,941 | **0,916** | 0,746 | 0,699 |
| bge-m3 d'origine, hybride + réécriture | 0,753 | 0,906 | 0,988 | 0,827 | 0,782 | 0,826 |
| bge-m3 fine-tuné, hybride + réécriture | 0,859 | 0,965 | 0,988 | 0,898 | 0,780 | 0,769 |

Bootstrap apparié sur les 85 questions (dense, fine-tuné moins origine) : MRR **+0,083**
(intervalle à 95 % : +0,025 à +0,146), Recall@1 +0,129 (+0,059 à +0,212), Recall@5 +0,024
(-0,024 à +0,071, non significatif), **Coverage@10 -0,111** (-0,175 à -0,049) : le modèle place
mieux un chunk attendu en tête mais en ramène moins dans le top 10 (la perte ne garde qu'un
positif par question).

**Fuite** : aucun chunk attendu ni aucune question du benchmark dans les 9 084 paires
(vérifié : 0 paire touchant un chunk attendu, 0 question identique, 0 question proche, Jaccard
des mots >= 0,8). Mais les paires viennent du même corpus : **75 des 85 questions** visent une
notice dont d'autres sections ont servi à l'entraînement. Sur ces 75 questions le MRR dense passe
de 0,848 à 0,960 ; sur les 10 autres il baisse de 0,717 à 0,583 (trop peu de questions pour
conclure, mais aucun signe d'un gain général).

**Décision (D123) : le modèle fine-tuné n'est pas servi** ; la branche
`experiment/fine-tuning-embedding` reste comme trace (tag `model/embedding-bge-m3-finetuned-v1.0.0`).
Raisons : le gain vient des documents vus à l'entraînement et n'est pas démontré au-delà ; il
réduit la couverture du contexte donné au LLM (Coverage@10 -0,11, significatif) ; il demanderait
une conversion GGUF pour Ollama et la réindexation des 8 661 chunks, l'API n'ayant pas PyTorch.
La règle de choix l'écarte de toute façon (`servable_only`). Un prochain essai devrait tenir à
l'écart des documents entiers et garder plusieurs positifs par question.

## Reproduire

```bash
# venv du projet (.venv), depuis la racine ; pile lancée (qdrant, ollama, postgres, mlflow)
export TUJYA_PHARMA_DATABASE_URL=postgresql+psycopg://tujya:tujya@127.0.0.1:5432/tujya
export PYTHONIOENCODING=utf-8                       # console Windows (emoji de MLflow)
dvc pull data/processed/rag/bench/queries.json.dvc data/processed/rag/bench/vectors.dvc
python -m src.benchmark.bench_grid list             # les configurations de config/rag_grid.yaml
python -m src.benchmark.bench_grid run --group bge-m3          # un run MLflow par configuration
python -m src.benchmark.bench_grid latency --configs bge-m3-dense,bge-m3-dense-rewrite
python -m src.benchmark.bench_grid select           # champion / challenger (sans enregistrer)
python -m src.benchmark.bench_grid select --register # versions + alias dans le Registry
python -m src.benchmark.bench_grid table            # tableau ci-dessus
```

### Ajouter un embedding (exemple : bge-m3 fine-tuné)

```bash
# 1. importer l'export Kaggle (venv d'entraînement, branche experiment/fine-tuning-embedding)
.venv-train/Scripts/python -m src.models.embedding.import_kaggle runs/kaggle/finetune_bge-m3_export.zip
# 2. encoder chunks et requêtes sur le GPU (arrêter d'abord les modèles Ollama : 4 Go)
.venv-train/Scripts/python -m src.benchmark.encode_st --model BAAI/bge-m3 --slug bge-m3-st
.venv-train/Scripts/python -m src.benchmark.encode_st --model data/models/embeddings/finetune-bge-m3/model --slug bge-m3-finetuned
# 3. le groupe de la grille (déjà dans config/rag_grid.yaml)
python -m src.benchmark.bench_grid run --group bge-m3-finetuned
```

`rewrites` régénère `queries.json` avec le LLM servi (d'autres réécritures donnent d'autres
scores : la version DVC est celle des résultats ci-dessus).

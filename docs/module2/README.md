# Module 2 : AI Engineering

Un agent **LangGraph** répond aux questions sur les comprimés identifiés, avec un **RAG**
sur trois sources documentées (notices officielles openFDA, résumés PubMed, interactions
DDInter) et des **citations numérotées** vérifiées. Le corpus (8 661 chunks pour 4 002
documents, avec leurs métadonnées) est indexé dans **Qdrant** avec des embeddings **bge-m3**
(tableau de bord sur <http://localhost:6333/dashboard>, page « Documents » de l'application) ; le LLM local **qwen2.5:3b** tourne dans **Ollama**. L'agent appelle
des **outils** (identifier, vérifier une interaction, notice d'un médicament, recherche
documentaire), exposés aussi par un **serveur MCP**. Un **benchmark** de 88 questions
mesure la récupération et la génération. Le tout est servi dans l'interface web, en flux
SSE. Évaluation : dépôt GitHub et historique des commits, rapport technique, démonstration.

Rapport technique : [docs/module2/rapport_module2_ai_engineering.pdf](rapport_module2_ai_engineering.pdf)
(source [Markdown](rapport_module2_ai_engineering.md)). Vue d'ensemble du projet :
[README principal](../../README.md).

## Exigences du module et fichiers correspondants

| Exigence | Où la trouver |
|---|---|
| RAG : gestion des documents | collecte (notices openFDA, résumés PubMed, paires DDInter) et découpage en chunks : [src/data/corpus/](../../src/data/corpus/) ([corpus.py](../../src/data/corpus/corpus.py), [\_\_main\_\_.py](../../src/data/corpus/__main__.py)) sur les connecteurs du [pipeline](../data_pipeline/README.md) ; corpus versionné par DVC (étape `rag_corpus` de [dvc.yaml](../../dvc.yaml)) ; indexation idempotente : [src/rag/index.py](../../src/rag/index.py) |
| RAG : métadonnées | colonnes de chaque chunk (source, médicament, section, URL, date, licence, rétractation) : [src/data/corpus/corpus.py](../../src/data/corpus/corpus.py) ; payload Qdrant (toutes les métadonnées, `content_hash` SHA-256) et filtres : [src/rag/store.py](../../src/rag/store.py) ; [docs/module2/rag.md](rag.md) |
| RAG : base vectorielle consultable | Qdrant (`docker compose` service `qdrant`, collection `tujya_chunks`, un point par chunk, index de payload sur les champs filtrés) : [src/rag/qdrant_store.py](../../src/rag/qdrant_store.py) ; catalogue des documents reconstruit depuis les chunks : [src/rag/documents.py](../../src/rag/documents.py) ; `GET /api/documents` (liste, filtres, facettes) et `GET /api/documents/{doc_id}` (métadonnées et chunks) : [backend/app/routers/documents.py](../../backend/app/routers/documents.py) ; page « Documents » : [frontend/src/pages/DocumentsPage.tsx](../../frontend/src/pages/DocumentsPage.tsx) ; [rag.md](rag.md#consulter-les-documents-indexés), captures [27](../screenshots/27-documents.jpg) à [29](../screenshots/29-qdrant-dashboard.jpg) (D113, D114) |
| RAG : recherche | BM25 (vecteurs sparse `Qdrant/bm25` dans Qdrant), dense (bge-m3) et hybride RRF fait par Qdrant, reranker cross-encoder BAAI/bge-reranker-base ([src/rag/rerank.py](../../src/rag/rerank.py)), réécriture de requête par le LLM ([src/rag/rewrite.py](../../src/rag/rewrite.py)), filtres de métadonnées (principe actif, type, section, rétractation) ; [rag.md](rag.md#recherche-avancée-étape-32), mesures dans [benchmark.md](benchmark.md#recherche-avancée-étape-32) (D116-D118) : [src/rag/retrieval.py](../../src/rag/retrieval.py), [src/rag/embeddings.py](../../src/rag/embeddings.py), [src/rag/names.py](../../src/rag/names.py) |
| RAG : citations | citations numérotées, attribution et vérification : [src/agent/graph.py](../../src/agent/graph.py), [src/agent/safety.py](../../src/agent/safety.py) ; affichage cliquable : [frontend/src/pages/ChatPanel.tsx](../../frontend/src/pages/ChatPanel.tsx) ; capture [10](../screenshots/10-chat-citation.jpg) |
| RAG : documentation | [docs/module2/rag.md](rag.md), [docs/module2/interactions.md](interactions.md), [docs/module2/agent.md](agent.md) |
| Tool calling | outils LangChain `identify_pills`, `check_interaction`, `get_drug_info`, `search_documents` (et `predict_toxicity`, [docs/bonus/](../bonus/README.md)) : [src/agent/tools.py](../../src/agent/tools.py) |
| MCP | serveur MCP (SDK officiel, stdio et HTTP limité à 127.0.0.1), mêmes outils, même code : [src/mcp_server/server.py](../../src/mcp_server/server.py) ; essai de bout en bout : [src/mcp_server/smoke.py](../../src/mcp_server/smoke.py) ; [docs/module2/mcp.md](mcp.md) |
| LangChain et/ou LangGraph | graphe LangGraph comprendre → outils → récupération → réponse citée → contrôle de sécurité : [src/agent/graph.py](../../src/agent/graph.py) ; câblage pour l'API : [src/agent/runtime.py](../../src/agent/runtime.py) ; schéma : [docs/module2/assets/agent_graph.png](assets/agent_graph.png) |
| Benchmark de la récupération | grille de 27 configurations (embeddings bge-m3 / multilingual-e5-base / all-MiniLM-L6-v2 / bge-m3 fine-tuné, dense / hybride / BM25, découpage, réécriture, 5 rerankers), Recall@k, MRR, nDCG@10, latence D96, un run MLflow par configuration (`tujya-rag-retrieval`), champion enregistré avec alias : [src/benchmark/bench_grid.py](../../src/benchmark/bench_grid.py), [config/rag_grid.yaml](../../config/rag_grid.yaml), [src/benchmark/grid/](../../src/benchmark/grid/) ; [benchmark_retrieval.md](benchmark_retrieval.md) (D123) ; mesures antérieures : [bench_retrieval.py](../../src/benchmark/bench_retrieval.py), [results_retrieval.json](../../src/benchmark/results_retrieval.json) |
| Benchmark de la génération | réponses, sécurité, citations vérifiées par un juge local, RAGAS : [src/benchmark/bench_answers.py](../../src/benchmark/bench_answers.py), [bench_generation.py](../../src/benchmark/bench_generation.py), [bench_judge_control.py](../../src/benchmark/bench_judge_control.py) ; jeu de 88 questions : [src/benchmark/rag_questions.jsonl](../../src/benchmark/rag_questions.jsonl) ; [docs/module2/benchmark.md](benchmark.md) |
| Fine-tuning (option du module) | embedding bge-m3 affiné sur des paires question / passage du corpus, sans question du benchmark : [src/models/embedding/](../../src/models/embedding/) (paires, entraînement, notebook Kaggle, import MLflow), [notebooks/kaggle/finetune_bge-m3.ipynb](../../notebooks/kaggle/finetune_bge-m3.ipynb) ; guide et état : [kaggle_finetuning.md](kaggle_finetuning.md) (en attente du run Kaggle, D121) |
| Interface web | panneau « Ask about your pills » : [frontend/src/pages/ChatPanel.tsx](../../frontend/src/pages/ChatPanel.tsx) ; `POST /api/chat` (JSON ou flux SSE) : [backend/app/routers/chat.py](../../backend/app/routers/chat.py) ; captures [08](../screenshots/08-chat-empty.jpg) à [11](../screenshots/11-chat-dose-refused.jpg) |
| Sécurité des réponses | jamais de dose, jamais « sûr », horaires seulement si la source les donne, avertissement : [src/agent/safety.py](../../src/agent/safety.py), [tests/agent/test_safety.py](../../tests/agent/test_safety.py) |
| Dépôt GitHub documenté, commits réguliers | ce README, [README.md](../../README.md), `git log`, tags git, [docs/git_workflow.md](../git_workflow.md) (branches, Conventional Commits, hooks) |
| Rapport technique | [docs/module2/rapport_module2_ai_engineering.pdf](rapport_module2_ai_engineering.pdf) |
| Démonstration | [docs/demo.md](../demo.md) (étapes chat, dose refusée, MCP, benchmark) |
| Tests | [tests/data/](../../tests/data/), [tests/rag/](../../tests/rag/), [tests/agent/](../../tests/agent/), [tests/mcp_server/](../../tests/mcp_server/) : corpus et connecteurs (HTTP simulé), recherche (SQLite, pgvector et Qdrant en mémoire), catalogue des documents et `/api/documents`, API de chat et agent avec un LLM factice, sécurité, MCP |

## Résultats du benchmark

88 questions (44 en français, 44 en anglais) construites à partir du corpus et relues à la
main. Détails, limites et commandes : [docs/module2/benchmark.md](benchmark.md) ; chaque run est
dans MLflow (expérience `tujya-rag`).

| Récupération (83 questions, top 10) | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|
| BM25 + filtre principe actif | 0,566 | 0,747 | 0,783 | 0,819 | 0,660 |
| hybride (RRF) + filtre | 0,711 | 0,892 | 0,904 | 0,928 | 0,796 |
| **dense (bge-m3) + filtre, retenu** | **0,795** | **0,880** | **0,928** | **0,976** | **0,852** |

Grille de l'étape 3.3 (85 questions, [benchmark_retrieval.md](benchmark_retrieval.md), D123) :

| Récupération (85 questions, top 10) | Recall@1 | Recall@5 | MRR | ms / question (D96) |
|---|---:|---:|---:|---:|
| dense bge-m3 + filtre (challenger) | 0,776 | 0,906 | 0,832 | 63 |
| **dense bge-m3 + filtre + réécriture (champion, servi)** | **0,765** | **0,941** | **0,834** | **389** |
| hybride + réécriture + reranker MiniLM-L6 (10 candidats) | 0,859 | 0,941 | 0,896 | 3 617 |
| hybride + réécriture + reranker MiniLM-L12 (20 candidats) | 0,894 | 0,965 | 0,922 | 11 251 |
| bge-m3 fine-tuné, dense (non servi) | 0,906 | 0,929 | 0,916 | 393 (run) |

| Génération (qwen2.5:3b, juge gemma3:4b) | Résultat |
|---|---:|
| Phrases citées soutenues par leur source | 94,0 % |
| Phrases avec citation | 69,8 % |
| Doses chiffrées / « sûr » dans les réponses | 0 / 0 |
| Avertissement présent ; questions de dose refusées | 100 % ; 2/2 |
| RAGAS (8 questions) : fidélité / précision du contexte | 0,979 / 0,809 |
| Temps de réponse moyen (RTX 2050) | 4,7 s |

## Organisation

Le module suit l'arborescence du guide, partagée avec le module 1 : tout le code est dans `src/`.

```
src/data/corpus/  principes actifs (RxNav), interactions DDInter, notices openFDA, résumés PubMed -> chunks
src/rag/          embeddings bge-m3, stockage Qdrant (ou SQL), recherche BM25 / dense / hybride, indexation, catalogue
src/agent/        graphe LangGraph (graph.py), outils (tools.py), règles de sécurité (safety.py), câblage (runtime.py)
src/mcp_server/   serveur MCP sur les mêmes outils
src/benchmark/    questions, scripts bench_*.py, résultats
tests/            tests (data, rag, agent, mcp_server), venv du projet
docs/module2/     rag, agent, mcp, benchmark, interactions, rapport (md + pdf), diagrams/, assets/
```

Le corpus s'appuie sur le pipeline de collecte `tujya-data` ([docs/data_pipeline/](../data_pipeline/README.md)) :
client HTTP poli, connecteurs DDInter, PubMed et openFDA, manifeste et provenance. Le corpus
dépend du pipeline, jamais l'inverse.

## Commandes

```bash
# Tout est lancé par docker compose (racine) : rag-init télécharge les modèles Ollama et
# indexe data/processed/rag/corpus dans Qdrant au premier démarrage
# (tableau de bord : http://localhost:6333/dashboard).
docker compose up -d --build

# Venv du projet (.venv) actif, depuis la racine du dépôt :
python -m src.agent.ask "La warfarine interagit-elle avec le fluconazole ?"
python -m src.rag.search "fluconazole warfarin" --ingredient warfarin --k 5
python -m src.rag.index                 # réindexation dans Qdrant (idempotente)
python -m src.rag.index --copy-from-sql # copie l'ancien index pgvector dans Qdrant, sans réencoder
python -m src.mcp_server                # serveur MCP (stdio)
python -m src.benchmark.bench_retrieval # benchmark de la récupération

# Corpus (même venv), depuis la racine :
python -m src.data.corpus labels     # notices openFDA
python -m src.data.corpus pubmed     # résumés PubMed
dvc repro rag_corpus                 # chunks (étape DVC)

# Tests du module (pgvector : TUJYA_TEST_PG_URL, sinon ces deux tests sont ignorés)
.venv/Scripts/python -m pytest tests/data tests/rag tests/agent tests/mcp_server
```

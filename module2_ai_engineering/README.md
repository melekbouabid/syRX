# Module 2 : AI Engineering

Un agent **LangGraph** répond aux questions sur les comprimés identifiés, avec un **RAG**
sur trois sources documentées (notices officielles openFDA, résumés PubMed, interactions
DDInter) et des **citations numérotées** vérifiées. Le corpus (8 661 chunks pour 4 002
documents, avec leurs métadonnées) est indexé dans **PostgreSQL + pgvector** avec des
embeddings **bge-m3** ; le LLM local **qwen2.5:3b** tourne dans **Ollama**. L'agent appelle
des **outils** (identifier, vérifier une interaction, notice d'un médicament, recherche
documentaire), exposés aussi par un **serveur MCP**. Un **benchmark** de 88 questions
mesure la récupération et la génération. Le tout est servi dans l'interface web, en flux
SSE. Évaluation : dépôt GitHub et historique des commits, rapport technique, démonstration.

Rapport technique : [docs/rapport_module2_ai_engineering.pdf](docs/rapport_module2_ai_engineering.pdf)
(source [Markdown](docs/rapport_module2_ai_engineering.md)). Vue d'ensemble du projet :
[README principal](../README.md).

## Exigences du module et fichiers correspondants

| Exigence | Où la trouver |
|---|---|
| RAG : gestion des documents | collecte (notices openFDA, résumés PubMed, paires DDInter) et découpage en chunks : [corpus/](corpus/) ([corpus.py](corpus/corpus.py), [\_\_main\_\_.py](corpus/__main__.py)) sur les connecteurs du [pipeline](../data_pipeline/) ; corpus versionné par DVC (étape `rag_corpus` de [../dvc.yaml](../dvc.yaml)) ; indexation idempotente : [rag/index.py](rag/index.py) |
| RAG : métadonnées | colonnes de chaque chunk (source, médicament, section, URL, date, licence, rétractation) : [corpus/corpus.py](corpus/corpus.py) ; table pgvector et filtres : [rag/store.py](rag/store.py) ; [docs/rag.md](docs/rag.md) |
| RAG : recherche | BM25, dense (bge-m3) et hybride RRF, filtres par principe actif : [rag/retrieval.py](rag/retrieval.py), [rag/embeddings.py](rag/embeddings.py), [rag/names.py](rag/names.py) |
| RAG : citations | citations numérotées, attribution et vérification : [agent/graph.py](agent/graph.py), [agent/safety.py](agent/safety.py) ; affichage cliquable : [../app/frontend/src/pages/ChatPanel.tsx](../app/frontend/src/pages/ChatPanel.tsx) ; capture [10](../docs/screenshots/10-chat-citation.jpg) |
| RAG : documentation | [docs/rag.md](docs/rag.md), [docs/interactions.md](docs/interactions.md), [docs/agent.md](docs/agent.md) |
| Tool calling | outils LangChain `identify_pills`, `check_interaction`, `get_drug_info`, `search_documents` (et `predict_toxicity`, [bonus](../bonus/)) : [agent/tools.py](agent/tools.py) |
| MCP | serveur MCP (SDK officiel, stdio et HTTP limité à 127.0.0.1), mêmes outils, même code : [mcp_server/server.py](mcp_server/server.py) ; essai de bout en bout : [mcp_server/smoke.py](mcp_server/smoke.py) ; [docs/mcp.md](docs/mcp.md) |
| LangChain et/ou LangGraph | graphe LangGraph comprendre → outils → récupération → réponse citée → contrôle de sécurité : [agent/graph.py](agent/graph.py) ; câblage pour l'API : [agent/runtime.py](agent/runtime.py) ; schéma : [docs/assets/agent_graph.png](docs/assets/agent_graph.png) |
| Benchmark de la récupération | Recall@1/3/5/10 et MRR, BM25 / dense / hybride, avec et sans filtre : [benchmark/bench_retrieval.py](benchmark/bench_retrieval.py) ; résultats : [benchmark/results_retrieval.json](benchmark/results_retrieval.json) |
| Benchmark de la génération | réponses, sécurité, citations vérifiées par un juge local, RAGAS : [benchmark/bench_answers.py](benchmark/bench_answers.py), [bench_generation.py](benchmark/bench_generation.py), [bench_judge_control.py](benchmark/bench_judge_control.py) ; jeu de 88 questions : [benchmark/rag_questions.jsonl](benchmark/rag_questions.jsonl) ; [docs/benchmark.md](docs/benchmark.md) |
| Interface web | panneau « Ask about your pills » : [../app/frontend/src/pages/ChatPanel.tsx](../app/frontend/src/pages/ChatPanel.tsx) ; `POST /api/chat` (JSON ou flux SSE) : [../app/backend/app/routers/chat.py](../app/backend/app/routers/chat.py) ; captures [08](../docs/screenshots/08-chat-empty.jpg) à [11](../docs/screenshots/11-chat-dose-refused.jpg) |
| Sécurité des réponses | jamais de dose, jamais « sûr », horaires seulement si la source les donne, avertissement : [agent/safety.py](agent/safety.py), [tests/test_safety.py](tests/test_safety.py) |
| Dépôt GitHub documenté, commits réguliers | ce README, [../README.md](../README.md), `git log`, tags git |
| Rapport technique | [docs/rapport_module2_ai_engineering.pdf](docs/rapport_module2_ai_engineering.pdf) |
| Démonstration | [../docs/demo.md](../docs/demo.md) (étapes chat, dose refusée, MCP, benchmark) |
| Tests | [tests/](tests/) : corpus et connecteurs (HTTP simulé), recherche (SQLite et pgvector), API de chat et agent avec un LLM factice, sécurité, MCP |

## Résultats du benchmark

88 questions (44 en français, 44 en anglais) construites à partir du corpus et relues à la
main. Détails, limites et commandes : [docs/benchmark.md](docs/benchmark.md) ; chaque run est
dans MLflow (expérience `tujya-rag`).

| Récupération (83 questions, top 10) | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|
| BM25 + filtre principe actif | 0,566 | 0,747 | 0,783 | 0,819 | 0,660 |
| hybride (RRF) + filtre | 0,711 | 0,892 | 0,904 | 0,928 | 0,796 |
| **dense (bge-m3) + filtre, retenu** | **0,795** | **0,880** | **0,928** | **0,976** | **0,852** |

| Génération (qwen2.5:3b, juge gemma3:4b) | Résultat |
|---|---:|
| Phrases citées soutenues par leur source | 94,0 % |
| Phrases avec citation | 69,8 % |
| Doses chiffrées / « sûr » dans les réponses | 0 / 0 |
| Avertissement présent ; questions de dose refusées | 100 % ; 2/2 |
| RAGAS (8 questions) : fidélité / précision du contexte | 0,979 / 0,809 |
| Temps de réponse moyen (RTX 2050) | 4,7 s |

## Organisation

```
module2_ai_engineering/
  corpus/       principes actifs (RxNav), interactions DDInter, notices openFDA, résumés PubMed -> chunks
  rag/          embeddings bge-m3, stockage pgvector, recherche BM25 / dense / hybride, indexation
  agent/        graphe LangGraph (graph.py), outils (tools.py), règles de sécurité (safety.py), câblage (runtime.py)
  mcp_server/   serveur MCP sur les mêmes outils
  benchmark/    questions, scripts bench_*.py, résultats
  tests/        tests du module (venv du backend)
  docs/         rag, agent, mcp, benchmark, interactions, rapport (md + pdf), diagrams/, assets/
```

Le corpus s'appuie sur le pipeline SynergyRx ([../data_pipeline/](../data_pipeline/)) :
client HTTP poli, connecteurs DDInter, PubMed et openFDA, manifeste et provenance. Le corpus
dépend du pipeline, jamais l'inverse.

## Commandes

```bash
# Tout est lancé par docker compose (racine) : rag-init télécharge les modèles Ollama et
# indexe data/rag/corpus dans pgvector au premier démarrage.
docker compose up -d --build

# Venv du backend (app/backend/.venv) actif, depuis la racine du dépôt :
python -m module2_ai_engineering.agent.ask "La warfarine interagit-elle avec le fluconazole ?"
python -m module2_ai_engineering.rag.search "fluconazole warfarin" --ingredient warfarin --k 5
python -m module2_ai_engineering.rag.index                 # réindexation (idempotente)
python -m module2_ai_engineering.mcp_server                # serveur MCP (stdio)
python -m module2_ai_engineering.benchmark.bench_retrieval # benchmark de la récupération

# Corpus (venv du pipeline : data_pipeline/.venv), depuis la racine :
python -m module2_ai_engineering.corpus labels     # notices openFDA
python -m module2_ai_engineering.corpus pubmed     # résumés PubMed
dvc repro rag_corpus                               # chunks (étape DVC)

# Tests du module (pgvector : TUJYA_TEST_PG_URL, sinon ces deux tests sont ignorés)
cd module2_ai_engineering && ../app/backend/.venv/Scripts/python -m pytest
```

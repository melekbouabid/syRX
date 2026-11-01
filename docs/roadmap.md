# Feuille de route de Tujya

Étapes dans l'ordre. Cocher une case quand l'étape est faite. Règles du projet :
[project_rules.md](project_rules.md). Les exigences du professeur sont les tableaux « Exigences des
modules » du [README](../README.md#exigences-des-modules).

## Déjà fait (hérité de SynergyRx)
- [x] Pipeline de données `data_pipeline/src/synergyrx_data` (14 connecteurs, 270 tests ; réduit à DDInter, PubMed et openFDA à l'étape 1.1, D85)
- [x] Backend FastAPI : comptes, sessions, administration (le catalogue SynergyRx des médicaments et des lignées a été retiré à l'étape 1.1, D85)
- [x] Ébauche de frontend React + Vite à l'image de Tujya (prédictions factices)
- [x] Planification de la nouvelle orientation (deux modules, périmètre, choix techniques)

## 0. Git et règles du projet
- [x] Premier commit et push vers https://github.com/melekbouabid/tujia-.git (9398f0c)
- [x] README avec les exigences des modules, les sources de données et le journal de progression
- [x] Règles du projet (sécurité des réponses, usage local) dans [project_rules.md](project_rules.md)
- [x] Conventions Git : branches, Conventional Commits, tags d'étape et de modèle, hooks pre-commit et commit-msg testés, ESLint, contrôle des messages en CI ([git_workflow.md](git_workflow.md), `v1.1.0-conventions`)
- [x] Fichiers inutiles retirés (remarque du professeur) : catalogue, étude de synergie et partie synergie du pipeline SynergyRx, fichiers isolés ; tout reste dans l'historique git (D85)
- [ ] Changer le mot de passe administrateur par défaut (admin@gmail.com / admin)

## 1. Données des comprimés (C3PI, aucune photo personnelle)
- [x] Explorer l'échantillon C3PI et l'organisation complète de l'archive ([dataset_c3pi.md](../module1_vision_mlops/docs/dataset_c3pi.md))
- [x] `config/pill_classes.csv` : les 150 classes ayant le plus d'images « consumer » (D32)
- [x] Télécharger les images de ces classes seulement (4,8 Go, SHA-1 vérifié)
- [x] Split entraînement / validation / test sans photo dans deux splits (`config/splits/`)
- [x] DVC en place ; images et métadonnées de `data/vision/c3pi` suivies (cache local)
- [x] Remote DVC configuré et alimenté (dossier local hors du dépôt, D71 ; `dvc pull` vérifié sur un clone neuf)
- [x] Chaque classe reliée à ses principes actifs (RxNav, 149/150)
- [x] DDInter activé ; interactions entre ces principes actifs dans une table propre ([interactions.md](../module2_ai_engineering/docs/interactions.md))

## 2. Modèles de vision
- [x] Matériel vérifié, projet `module1_vision_mlops/training/` avec son propre venv (RTX 2050, CUDA 13.0)
- [x] Identification : classifieur entraîné par transfer learning (YOLO11s-cls, top-1 test 93,5 %)
- [x] Détection : boîtes englobantes des comprimés (étiquettes automatiques, D38 ; YOLO11n, mAP50 test 0,972)
- [x] Runs MLflow (paramètres, métriques, artefacts)
- [x] Meilleur modèle enregistré dans le Model Registry de MLflow
- [x] Export ONNX, inférence avec ONNX Runtime, sorties identiques à PyTorch
- [x] Mesurer top-1 / top-5 (identification) et mAP (détection)
- [ ] Comparer avec ePillID sur son propre split et ses métriques ([vision.md](../module1_vision_mlops/docs/vision.md))
- [ ] Vérifier à la main les boîtes de test ; second passage d'étiquetage avec le détecteur entraîné

## 3. API
- [x] Schéma SQL (pills, ingredients, pill_ingredients, interactions, sources) + chargeur
- [x] `POST /api/scan` asynchrone (image -> boîtes, top-5, confiance ; confirmation sous 0,85, [api.md](api.md))
- [x] `python -m tujya_vision.publish` : modèles enregistrés -> `models/` pour l'API
- [x] `POST /api/interactions` asynchrone (sévérité DDInter, description, source ; jamais « sûr »)
- [x] `GET /api/pills?q=` : recherche parmi les 150 classes (confirmation manuelle)
- [x] Swagger sur `/docs` et schéma sur `/openapi.json` (`/api/docs` redirige, D44)
- [x] Schémas Pydantic du scan et des interactions
- [x] Tests (modèles ONNX factices, tables de médicaments en SQLite ; 69 tests backend à l'époque)

## 4. Corpus RAG
- [x] Notices openFDA des 82 principes actifs (connecteur `openfda_label`, D60)
- [x] Résumés PubMed : jusqu'à 20 par principe actif sur les interactions et les effets indésirables (D61)
- [x] Découpage avec métadonnées (source, médicament, section, URL, date, licence, rétractation ; [rag.md](../module2_ai_engineering/docs/rag.md))
- [x] Embeddings bge-m3 dans PostgreSQL + pgvector, recherche hybride BM25 (RRF), filtres (D59)
- [x] Corpus versionné par DVC (`data/rag/*.dvc`, étape `rag_corpus`)

## 5. Agent
- [x] Graphe LangGraph : comprendre -> outils -> récupérer -> répondre avec citations -> contrôle de sécurité ([agent.md](../module2_ai_engineering/docs/agent.md))
- [x] Outils : identify_pills, check_interaction, get_drug_info, search_documents (point d'extension pour d'autres)
- [x] `POST /api/chat` asynchrone (JSON ou flux SSE), réponse + citations + appels d'outils ; tests avec un LLM factice
- [x] LLM local Ollama : qwen2.5:3b dans le service `ollama`, sur le GPU (D57, D58)
- [x] Jeu de benchmark : 88 questions FR/EN construites à partir du corpus, relues à la main (D66)
- [x] Benchmark de la récupération (Recall@k, MRR) et de la génération (sécurité, citations, RAGAS) ; MLflow `tujya-rag`
- [x] Serveur MCP exposant les mêmes outils (SDK officiel, stdio + HTTP ; [mcp.md](../module2_ai_engineering/docs/mcp.md))

## 6. Site web
- [x] Page Scanner : upload + webcam (prédictions en direct) + boîtes + galerie de démonstration (D52)
- [x] Résultat (interaction, sévérité, source) sur la page Scanner ; captures dans [screenshots/](screenshots/)
- [x] Panneau de chat avec réponse diffusée, citations cliquables, appels d'outils (page Scanner)
- [x] Avertissement sur la page Scanner, chaque rapport d'interactions, le panneau de chat et chaque réponse

## 7. Livraison
- [x] Docker Compose : postgres + pgvector, mlflow ([infra.md](infra.md))
- [x] Docker Compose : backend, frontend (nginx), chargeur ponctuel ; une seule commande
- [x] Docker Compose : service ollama avec le GPU NVIDIA
- [x] CI GitHub Actions (lint, tests, build du site)
- [x] CD GitHub Actions : images construites et testées après une CI verte, non publiées (D70, [cd.md](cd.md))
- [ ] Supprimer les deux anciens paquets sur ghcr.io (manuel : pas de droit `delete:packages`, [cd.md](cd.md))
- [x] Usage strictement local : chaque port sur 127.0.0.1, vérifié avec netstat (D69, [infra.md](infra.md))
- Pas de déploiement sur un serveur, par choix : le projet ne tourne que sur ce PC (D69)
- [x] README mis à jour pour Tujya
- [x] Rapport du module 1 (Vision & MLOps) : [rapport_module1_vision_mlops.pdf](../module1_vision_mlops/docs/rapport_module1_vision_mlops.pdf)
- [x] Rapport du module 2 (AI Engineering) : [rapport_module2_ai_engineering.pdf](../module2_ai_engineering/docs/rapport_module2_ai_engineering.pdf)
- [x] Script de démonstration ([demo.md](demo.md), répété sur les conteneurs) et questions du jury ([questions_jury.md](questions_jury.md))
- [ ] Vidéo de secours de la démonstration (à enregistrer à la main)

## 8. Structure du dépôt (2026-10-29)
- [x] Deux dossiers de premier niveau pour les modules notés, chacun avec un README qui relie chaque exigence à ses fichiers (D79 à D82)
- [x] Application web commune dans `app/`, bonus dans `bonus/`, pipeline SynergyRx dans `data_pipeline/`, infrastructure dans `infra/`
- [x] Documentation rangée par module, noms en minuscules, archive

## Bonus (seulement s'il reste du temps)
- [x] Outil `predict_toxicity` : tests in vitro Tox21, LightGBM sur empreintes de Morgan, split par scaffold, ROC-AUC test moyen 0,725 ([toxicity.md](../bonus/docs/toxicity.md))
- [x] `predict_synergy` étudié (DrugComb, médicaments laissés de côté) : RMSE −7,3 % contre la moyenne, sous la règle des 10 % (D75), donc **aucun outil ajouté** ; étude retirée du dépôt à l'étape 1.1 (D85)
- [ ] OCR des empreintes des comprimés
- ~~Page des données de synergie anticancéreuse~~ : abandonnée, données SynergyRx retirées (D85)

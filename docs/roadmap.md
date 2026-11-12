# Feuille de route de Tujya

Étapes dans l'ordre. Cocher une case quand l'étape est faite. Règles du projet :
[docs/project_rules.md](project_rules.md). Les exigences du professeur sont les tableaux « Exigences des
modules » du [README](../README.md#exigences-des-modules).

## Déjà fait (hérité du projet d'origine, synergie de médicaments)
- [x] Pipeline de données `src/data/tujya_data` (14 connecteurs, 270 tests ; réduit à DDInter, PubMed et openFDA à l'étape 1.1, D85)
- [x] Backend FastAPI : comptes, sessions, administration (le catalogue des médicaments et des lignées du projet d'origine a été retiré à l'étape 1.1, D85)
- [x] Ébauche de frontend React + Vite à l'image de Tujya (prédictions factices)
- [x] Planification de la nouvelle orientation (deux modules, périmètre, choix techniques)

## 0. Git et règles du projet
- [x] Premier commit et push vers https://github.com/melekbouabid/tujia-.git (9398f0c)
- [x] README avec les exigences des modules, les sources de données et le journal de progression
- [x] Règles du projet (sécurité des réponses, usage local) dans [docs/project_rules.md](project_rules.md)
- [x] Conventions Git : branches, Conventional Commits, tags d'étape et de modèle, hooks pre-commit et commit-msg testés, ESLint, contrôle des messages en CI ([docs/git_workflow.md](git_workflow.md), `v1.1.0-conventions`)
- [x] Fichiers inutiles retirés (remarque du professeur) : catalogue, étude de synergie et partie synergie du pipeline d'origine, fichiers isolés ; tout reste dans l'historique git (D85)
- [ ] Changer le mot de passe administrateur par défaut (admin@gmail.com / admin)

## 1. Données des comprimés (C3PI, aucune photo personnelle)
- [x] Explorer l'échantillon C3PI et l'organisation complète de l'archive ([dataset_c3pi.md](module1/dataset_c3pi.md))
- [x] `config/pill_classes.csv` : les 150 classes ayant le plus d'images « consumer » (D32)
- [x] Télécharger les images de ces classes seulement (4,8 Go, SHA-1 vérifié)
- [x] Split entraînement / validation / test sans photo dans deux splits (`config/splits/`)
- [x] DVC en place ; images et métadonnées C3PI suivies (`data/raw/c3pi` depuis l'étape 1.2, cache local)
- [x] Remote DVC configuré et alimenté (dossier local hors du dépôt, D71 ; `dvc pull` vérifié sur un clone neuf)
- [x] Chaque classe reliée à ses principes actifs (RxNav, 149/150)
- [x] DDInter activé ; interactions entre ces principes actifs dans une table propre ([interactions.md](module2/interactions.md))
- [x] Plus de données et Big Data (étape 1.4, demande du professeur) : cluster Spark dans Docker Compose, jobs PySpark index / qualité / split sans fuite / augmentation des classes rares / ePillID / corpus RAG, étapes DVC, statistiques dans MLflow ([big_data.md](module1/big_data.md), D88-D90)
- [ ] Entraîner les modèles sur le jeu C3PI v2 (étapes 2.2 et 2.3) : détecteurs faits (étape 2.2), classifieurs en attente Kaggle (étape 2.3 : jeu de crops v2, étape DVC `prepare_classify_v2`)

## 2. Modèles de vision
- [x] Matériel vérifié, entraînement dans son propre venv (`.venv-train` depuis l'étape 1.2 ; RTX 2050, CUDA 13.0)
- [x] Identification : classifieur entraîné par transfer learning (YOLO11s-cls, top-1 test 93,5 %)
- [x] Détection : boîtes englobantes des comprimés (étiquettes automatiques, D38 ; YOLO11n, mAP50 test 0,972)
- [x] Runs MLflow (paramètres, métriques, artefacts)
- [x] Meilleur modèle enregistré dans le Model Registry de MLflow
- [x] MLflow conforme au lab (étape 2.1) : runs tracés (`git_commit`, `dvc_rev`, `dataset`, `seed`, requirements + `dvc.lock`), LoggedModel MLflow 3, alias `champion` / `challenger`, `promote_if_better` ; étapes DVC `train_detector`, `train_classifier`, `export_onnx`, `evaluate` ; baseline YOLO11n + YOLO11s-cls champion ([mlflow.md](module1/mlflow.md), D91-D92)
- [ ] Comparer d'autres détecteurs et classifieurs, recherche d'hyperparamètres (étapes 2.2 et 2.3, GPU Kaggle)
  - [x] Détecteurs (étape 2.2) : YOLOv8n, YOLOv8s, YOLO11s, YOLOv10n, Faster R-CNN entraînés sur Kaggle, une branche `experiment/detector-*` et une recherche Optuna (8 à 12 essais, runs enfants MLflow) par modèle ; YOLO11s champion (mAP@[.5:.95] 0,8885), YOLOv10n challenger (0,8860, 2,9 fois plus rapide) ; latence des six détecteurs mesurée ensemble ([kaggle_detecteurs.md](module1/kaggle_detecteurs.md), D93-D94, D98)
  - [ ] Classifieurs (étape 2.3) : **EN ATTENTE KAGGLE**. Fait : jeu de crops C3PI v2 et split de comparaison sans session vue par le champion v1 (D99), outillage commun sur `feature/classifier-kaggle-tooling`, six branches `experiment/classifier-*` (baseline CNN maison, ResNet-50, EfficientNetV2-S, ConvNeXt-Tiny, ViT-B/16, YOLO11s-cls) avec leur recherche Optuna et leur notebook, tests de fumée locaux des six notebooks (export ONNX lu par l'API, import MLflow sans Registry), champion v1 réévalué (`test_v2_top1` 0,1830, 0,7527 sur ses 150 classes, D100), zip du dataset Kaggle prêt. À faire : lancer les six notebooks (environ 15,5 h de GPU T4), importer, migrer la table de classes de l'API vers v2 (D101), fusionner la branche championne, tags ([kaggle_classifieurs.md](module1/kaggle_classifieurs.md), D99-D103)
- [x] Export ONNX, inférence avec ONNX Runtime, sorties identiques à PyTorch
- [x] Optimisation des champions (étape 2.4) : variantes FP16 / INT8 (statique calibrée sur le val, dynamique), précision, latence moyenne / p95 et taille dans MLflow ; détecteur YOLOv8n servi en FP16 (-0,0016 mAP, 7 % plus rapide), classifieur en FP32 ; INT8 plus lent sur ce CPU ; relancé le 07/11 sur le champion YOLO11s : servi en FP32 (FP16 3,9 % plus rapide seulement) ([optimisation_onnx.md](module1/optimisation_onnx.md), D95-D96, D98)
- [x] Mesurer top-1 / top-5 (identification) et mAP (détection)
- [ ] Comparer avec ePillID sur son propre split et ses métriques ([vision.md](module1/vision.md))
- [ ] Vérifier à la main les boîtes de test ; second passage d'étiquetage avec le détecteur entraîné

## 3. API
- [x] Schéma SQL (pills, ingredients, pill_ingredients, interactions, sources) + chargeur
- [x] `POST /api/scan` asynchrone (image -> boîtes, top-5, confiance ; confirmation sous 0,85, [docs/api.md](api.md))
- [x] `python -m src.models.publish` : modèles enregistrés -> `models/` pour l'API
- [x] Endpoints du guide (étape 2.5) : `GET /health`, `GET /model/metadata`, `POST /predict`, `POST /predict/visualize`, erreurs 400 / 413 / 503, schémas stricts, tests de chaque route et code d'erreur ([api.md](api.md#endpoints-du-guide-du-module-1-sans-compte), D97)
- [x] `POST /api/interactions` asynchrone (sévérité DDInter, description, source ; jamais « sûr »)
- [x] `GET /api/pills?q=` : recherche parmi les 150 classes (confirmation manuelle)
- [x] Swagger sur `/docs` et schéma sur `/openapi.json` (`/api/docs` redirige, D44)
- [x] Schémas Pydantic du scan et des interactions
- [x] Tests (modèles ONNX factices, tables de médicaments en SQLite ; 69 tests backend à l'époque)

## 4. Corpus RAG
- [x] Notices openFDA des 82 principes actifs (connecteur `openfda_label`, D60)
- [x] Résumés PubMed : jusqu'à 20 par principe actif sur les interactions et les effets indésirables (D61)
- [x] Découpage avec métadonnées (source, médicament, section, URL, date, licence, rétractation ; [rag.md](module2/rag.md))
- [x] Embeddings bge-m3 dans PostgreSQL + pgvector, recherche hybride BM25 (RRF), filtres (D59)
- [x] Corpus versionné par DVC (`data/rag/*.dvc`, étape `rag_corpus`)
- [x] Base vectorielle consultable (étape 3.1, remarque 7) : index dans Qdrant (tableau de bord sur :6333/dashboard, payload = toutes les métadonnées + SHA-256, index de payload), copie des vecteurs pgvector sans réencodage, même Recall@5 / MRR (écart 0), page « Documents » et `GET /api/documents` ; captures 27 à 29 ([rag.md](module2/rag.md#consulter-les-documents-indexés), D113-D114)
- [x] Recherche avancée (étape 3.2, remarque 13) : BM25 sparse `Qdrant/bm25` et hybride RRF dans Qdrant, reranker bge-reranker-base (ONNX, désactivé par défaut : 13-18 s par question sur ce CPU), réécriture de requête par le LLM, sources avec métadonnées dans le chat ; meilleur : hybride + réécriture + reranker Recall@5 0,965 / MRR 0,889 ([benchmark.md](module2/benchmark.md#recherche-avancée-étape-32), D116-D118)

## 5. Agent
- [x] Graphe LangGraph : comprendre -> outils -> récupérer -> répondre avec citations -> contrôle de sécurité ([agent.md](module2/agent.md))
- [x] Outils : identify_pills, check_interaction, get_drug_info, search_documents (point d'extension pour d'autres)
- [x] `POST /api/chat` asynchrone (JSON ou flux SSE), réponse + citations + appels d'outils ; tests avec un LLM factice
- [x] LLM local Ollama : qwen2.5:3b dans le service `ollama`, sur le GPU (D57, D58)
- [x] Jeu de benchmark : 88 questions FR/EN construites à partir du corpus, relues à la main (D66)
- [x] Benchmark de la récupération (Recall@k, MRR) et de la génération (sécurité, citations, RAGAS) ; MLflow `tujya-rag`
- [x] Serveur MCP exposant les mêmes outils (SDK officiel, stdio + HTTP ; [mcp.md](module2/mcp.md))

## 6. Site web
- [x] Page Scanner : upload + webcam (prédictions en direct) + boîtes + galerie de démonstration (D52)
- [x] Frontend du guide (étape 2.6) : Tailwind CSS 4, glisser-déposer, galerie du split de test en 1 clic (photos jamais vues par le détecteur v9 ni le classifieur v1, D105), webcam en inférence continue (une requête à la fois), barres de probabilité animées, boîtes colorées par classe sur canvas, latence client et serveur en ms, modèle champion (`GET /model/metadata`), tests Vitest ; captures 01 à 07 et 18 à 21 ([app.md](app.md#page-scanner-guide-du-module-1-étape-26), D106-D107)
- [x] Grad-CAM (reporté de l'étape 2.5) : cartes de la galerie précalculées avec les poids PyTorch du classifieur servi (étape DVC `gradcam_gallery`), servies par `GET /api/demo/images/{id}/gradcam` et affichées dans le Scanner ([api.md](api.md#galerie-de-démo-et-grad-cam), D104)
- [x] Résultat (interaction, sévérité, source) sur la page Scanner ; captures dans [docs/screenshots/](screenshots/)
- [x] Panneau de chat avec réponse diffusée, citations cliquables, appels d'outils (page Scanner)
- [x] Avertissement sur la page Scanner, chaque rapport d'interactions, le panneau de chat et chaque réponse

- [x] Boîtes de médicaments (étape 2.7, remarque 18). Fait : OCR RapidOCR (champion, `ingredient_recall` 0,767) contre EasyOCR (0,480) sur 713 boîtes Italian-OTC-medicines, table des marques sourcée (BDPM, emc), DCI imprimée, RxNav en secours, refus hors distribution (98 % des comprimés gardés, 79 % des images hors sujet refusées), `/api/scan` avec `kind` et `box`, carte « boîte » dans le Scanner, marques dans le chat (`lookup_brand`), 8 questions de marques dans le benchmark ([boites.md](module1/boites.md), D108-D112). Validé sur 11 vraies boîtes sous licence libre de Wikimedia Commons (`--set web`, D115) : 3 Panadol sur 3 -> paracétamol avec sa source emc, 11 principes actifs justes sur 11, aucun faux. Plus tard : photos de l'utilisateur comme second jeu de test (`--set user`) ; Roboflow (compte)

## 7. Livraison
- [x] Docker Compose : postgres + pgvector, mlflow ([docs/infra.md](infra.md))
- [x] Docker Compose : backend, frontend (nginx), chargeur ponctuel ; une seule commande
- [x] Docker Compose : service ollama avec le GPU NVIDIA
- [x] CI GitHub Actions (lint, tests, build du site)
- [x] CD GitHub Actions : images construites et testées après une CI verte, non publiées (D70, [docs/cd.md](cd.md))
- [x] Workflows du guide : `ci-tests.yml` (flake8, black --check, ruff, mypy, ESLint, pytest de `src/` et `backend/`, build du frontend ; Vitest depuis l'étape 2.6) et `docker-build.yml` (images multi-stage backend et frontend + test de fumée, rien de publié) ; anciens `ci.yml` et `cd.yml` fusionnés (D87, `v2.1.0-ci`)
- [ ] Supprimer les deux anciens paquets sur ghcr.io (manuel : pas de droit `delete:packages`, [docs/cd.md](cd.md))
- [x] Usage strictement local : chaque port sur 127.0.0.1, vérifié avec netstat (D69, [docs/infra.md](infra.md))
- Pas de déploiement sur un serveur, par choix : le projet ne tourne que sur ce PC (D69)
- [x] README mis à jour pour Tujya
- [x] Rapport du module 1 (Vision & MLOps) : [rapport_module1_vision_mlops.pdf](module1/rapport_module1_vision_mlops.pdf)
- [x] Rapport du module 2 (AI Engineering) : [rapport_module2_ai_engineering.pdf](module2/rapport_module2_ai_engineering.pdf)
- [x] Script de démonstration ([docs/demo.md](demo.md), répété sur les conteneurs) et questions du jury ([docs/questions_jury.md](questions_jury.md))
- [ ] Vidéo de secours de la démonstration (à enregistrer à la main)

## 8. Structure du dépôt (2026-10-29)
- [x] Deux dossiers de premier niveau pour les modules notés, chacun avec un README qui relie chaque exigence à ses fichiers (D79 à D82)
- [x] Application web commune dans `app/`, bonus dans `bonus/`, pipeline de collecte dans `data_pipeline/`, infrastructure dans `infra/`
- [x] Documentation rangée par module, noms en minuscules, archive

## 9. Arborescence du guide des modules (étape 1.2, 2026-11-02)
- [x] Code des deux modules dans `src/` (`data`, `models`, `rag`, `agent`, `mcp_server`, `benchmark`, `utils`), API dans `backend/` (`app/inference.py` = chaîne de scan), site dans `frontend/` (`src/components`, `src/services`), tests dans `tests/` et `backend/tests/`, `notebooks/` (D86)
- [x] Données DVC rangées en `data/raw` et `data/processed` (`dvc move`, mêmes empreintes, rien à envoyer au remote)
- [x] Un seul venv Python pour l'API, `src/` et les tests (`.venv`, `requirements-dev.txt`) ; venv d'entraînement séparé (`.venv-train`)
- [x] Noms hérités du projet d'origine renommés : paquet `tujya_data`, commande `tujya-data`, préfixe de configuration `TUJYA_*` (l'ancien préfixe reste lu en repli pour les `.env` existants)

## Bonus (seulement s'il reste du temps)
- [x] Outil `predict_toxicity` : tests in vitro Tox21, LightGBM sur empreintes de Morgan, split par scaffold, ROC-AUC test moyen 0,725 ([toxicity.md](bonus/toxicity.md))
- [x] `predict_synergy` étudié (DrugComb, médicaments laissés de côté) : RMSE −7,3 % contre la moyenne, sous la règle des 10 % (D75), donc **aucun outil ajouté** ; étude retirée du dépôt à l'étape 1.1 (D85)
- [ ] OCR des empreintes des comprimés
- ~~Page des données de synergie anticancéreuse~~ : abandonnée, données de synergie retirées (D85)

# Script de démonstration (soutenance, 10 à 15 minutes)

Démonstration en direct de Tujya, sur ce PC uniquement (aucun déploiement public, D69).
Chaque étape : **ce qu'on fait** (clic ou commande), **ce que le jury doit voir**, et
**une phrase à dire**. Les durées sont indicatives (total ≈ 14 min).

Le parcours a été répété en entier sur les conteneurs le 2026-10-28 (session 4) : résultats
et temps mesurés dans [SESSION_REPORT_4.md](SESSION_REPORT_4.md#tâche-4--script-de-démo-et-questions-du-jury).

> Règle de sécurité à garder en tête pendant toute la démo : Tujya ne donne jamais de dose,
> ne dit jamais qu'une association est « sûre » (au mieux « aucune interaction connue dans
> cette base de données »), cite toujours ses sources, et affiche toujours « Information
> uniquement, consultez un pharmacien ou un médecin ».

---

## Avant la soutenance (préparation, 20 minutes avant)

1. Brancher le PC sur secteur (GPU), fermer les applications lourdes. Démarrer **Docker
   Desktop**.
2. Dans un terminal (Git Bash ou PowerShell), à la racine du dépôt :

   ```bash
   docker compose up -d --build
   docker compose ps          # backend, frontend, postgres, ollama "healthy" ; mlflow "Up" ;
                              # pharma-init et rag-init "Exited (0)"
   ```

3. **Chauffer le modèle de langage** (indispensable) : la toute première question après un
   démarrage a pris **84 s** pendant la répétition (chargement de qwen2.5:3b et bge-m3 sur le
   GPU), les suivantes **5 à 7 s**. Poser une question une fois avant l'arrivée du jury :

   ```bash
   cd backend
   export SYNERGYRX_PHARMA_DATABASE_URL=postgresql+psycopg://tujya:tujya@127.0.0.1:5432/tujya
   export SYNERGYRX_DATABASE_URL=$SYNERGYRX_PHARMA_DATABASE_URL PYTHONIOENCODING=utf-8
   .venv/Scripts/python scripts/ask.py "What are the contraindications of metformin?"
   docker compose exec ollama ollama ps     # qwen2.5:3b et bge-m3 : "100% GPU"
   ```

   (PowerShell : `$env:SYNERGYRX_PHARMA_DATABASE_URL = "postgresql+psycopg://tujya:tujya@127.0.0.1:5432/tujya"`.)
   Toujours `127.0.0.1` dans ces adresses, jamais `localhost` (sinon 1 à 2 minutes
   d'attente par connexion, D72).
4. Ouvrir à l'avance, dans le navigateur, ces onglets :
   - <http://localhost:8080> connecté avec `user@gmail.com` / `user`, page **Scanner** ;
   - <http://localhost:8080/docs> (Swagger) ;
   - <http://localhost:5000> (MLflow ; cliquer une fois « Model training » et fermer les
     fenêtres d'aide « Got it ») ;
   - <https://github.com/melekbouabid/tujia-/actions> (onglet Actions) ;
   - [docs/rapports/img/architecture.png](rapports/img/architecture.png) (schéma) ;
   - [docs/BENCHMARK.md](BENCHMARK.md) ou le tableau des résultats du README.
5. Garder un terminal ouvert dans `backend/` avec les variables ci-dessus, et un autre à la
   racine pour DVC et `gh`.
6. Préparer 1 ou 2 vraies boîtes de comprimés **parmi les 150 classes** si vous voulez
   montrer la webcam (fond uni, bonne lumière) ; sinon montrer la webcam sans conclure.
7. Vérifier la connexion Internet : Swagger UI charge ses fichiers depuis un CDN (D44).

---

## 1. Le problème (1 min)

- **Faire** : rien à l'écran, ou la page d'accueil <http://localhost:8080>.
- **Le jury voit** : la page d'accueil Tujya.
- **Dire** : « Une personne qui prend plusieurs médicaments ne sait pas toujours les
  reconnaître ni savoir s'ils interagissent ; Tujya identifie les comprimés sur une photo,
  vérifie les interactions dans DDInter et répond aux questions en citant ses sources, sans
  jamais remplacer un pharmacien. »

## 2. Architecture (1 min 30)

- **Faire** : ouvrir [rapports/img/architecture.png](rapports/img/architecture.png).
- **Le jury voit** : le site React (nginx), l'API FastAPI, les deux modèles ONNX, PostgreSQL
  + pgvector, Ollama sur le GPU, MLflow, DVC, GitHub Actions ; tout sur `127.0.0.1`.
- **Dire** : « Deux modules : vision et MLOps d'un côté (YOLO détecte, un classifieur
  identifie, ONNX Runtime sert les modèles), RAG et agent de l'autre (LangGraph, recherche
  dense dans pgvector, LLM local) ; tout tourne en local avec Docker Compose, rien n'est
  exposé hors de ce PC. »

## 3. Scan en direct : photos de démo (2 min)

- **Faire** : page **Scanner** -> onglet **Demo photos** -> cliquer la photo
  **Fluconazole** -> **Confirm this pill** ; puis **Citalopram** -> **Confirm this pill**.
- **Le jury voit** : la boîte dessinée sur le comprimé, le nom, le top-5 avec les
  probabilités (fluconazole ≈ 99,6 %), le temps d'inférence (≈ 50 à 220 ms), la liste
  « My pills ».
- **Dire** : « Ces photos viennent du split de test de C3PI, jamais vues à l'entraînement ;
  sur les 600 photos de test, le comprimé est trouvé dans 98,5 % des cas et identifié en
  top-1 à 89,3 %. »

## 4. Cas incertain et upload (1 min)

- **Faire** : cliquer la photo **Trazodone** (65 %).
- **Le jury voit** : boîte orange, badge **Please confirm**, top-5, champ de recherche d'un
  autre comprimé.
- **Dire** : « Sous le seuil de 0,85, choisi sur la validation, l'application demande une
  confirmation : en test, 70,5 % des photos passent sans confirmation et 99,3 % de celles-ci
  sont justes ; l'identification reste une suggestion. »
- **Optionnel** : onglet **Upload**, glisser une photo C3PI de test.

## 5. Webcam (1 min)

- **Faire** : onglet **Webcam** -> **Start the camera** -> montrer un comprimé (ou une
  boîte) -> **Capture and identify**.
- **Le jury voit** : la vidéo, les boîtes mises à jour environ chaque seconde, le temps
  d'inférence.
- **Dire** : « Les prédictions sont faites en direct par la même API ; le modèle n'a vu que
  des photos C3PI, donc sur une webcam les scores sont plus bas, et c'est justement là que
  la confirmation sert. »
- Testé seulement avec la caméra factice de Chromium (0 comprimé, mire de test) : essayer
  avec la vraie caméra avant la soutenance ; si ça ne reconnaît rien, le dire honnêtement.

## 6. Interactions (1 min)

- **Faire** : **Check interactions** (avec fluconazole et citalopram confirmés ; ajouter
  **traMADOL** pour avoir trois paires).
- **Le jury voit** : fluconazole + citalopram **Major**, citalopram + tramadol **Major**,
  fluconazole + tramadol **Moderate**, avec la source DDInter, la citation et
  l'avertissement « Information uniquement, consultez un pharmacien ou un médecin. »
- **Dire** : « Les sévérités viennent de DDInter, avec la source affichée ; quand rien
  n'est trouvé, l'application dit « aucune interaction connue dans cette base de données »,
  jamais que c'est sûr. »

## 7. Chat avec citations (2 min)

- **Faire** : panneau **Ask about your pills** en bas -> cliquer la suggestion « Le
  fluconazole et le citalopram interagissent-ils ? » -> cliquer **[1]** -> déplier **tool
  call(s)**.
- **Le jury voit** : la réponse arrive phrase par phrase (≈ 6 s), citations numérotées, la
  source DDInter « Major » surlignée après le clic, les outils appelés (marqués « llm »
  quand le modèle les a choisis, « rule » quand le graphe les a ajoutés, par exemple
  `search_documents`).
- **Dire** : « L'agent LangGraph choisit ses outils, cherche dans 8 661 passages de notices
  openFDA, de résumés PubMed et de DDInter, puis un nœud de sécurité vérifie chaque phrase
  avant affichage ; chaque phrase citée renvoie à sa source. »
- Autre bonne question si le temps le permet : « What are the contraindications of
  metformin? » (réponse citée [1]).

## 8. Question de dose refusée (30 s)

- **Faire** : taper « Quelle dose de citalopram dois-je prendre ? ».
- **Le jury voit** : refus immédiat (« Tujya ne donne jamais de dose ni de posologie… »),
  badge de sécurité, avertissement ; aucun appel au LLM.
- **Dire** : « Les règles de sécurité sont dans le code, pas seulement dans le prompt : une
  question de dose est refusée avant même d'appeler le modèle. »

## 9. Swagger `/docs` (45 s)

- **Faire** : onglet <http://localhost:8080/docs> -> déplier `POST /api/scan` et
  `POST /api/chat`.
- **Le jury voit** : « Tujya API », les routes scan, pills, interactions, chat, demo, et les
  schémas Pydantic.
- **Dire** : « API FastAPI asynchrone, schémas Pydantic, documentation générée
  automatiquement. »

## 10. MLflow (1 min 30)

- **Faire** : <http://localhost:5000> -> **Model training** -> expérience **tujya-vision**
  -> **Runs** ; puis expérience **tujya-rag** ; puis **Model registry**.
- **Le jury voit** : runs `classify-…` (classifieur v1 et v2), `detect-…` (détecteur v1),
  `evaluate-end-to-end` ; dans `tujya-rag`, les runs de récupération (BM25, dense, hybride,
  avec et sans filtre) et de génération ; registre : `tujya-pill-classifier` version 2,
  `tujya-pill-detector` version 1.
- **Dire** : « Chaque entraînement et chaque benchmark est tracé avec ses paramètres et ses
  métriques ; l'API sert exactement les versions du Model Registry, qu'elle renvoie dans
  chaque réponse de scan. »
- Captures : [12](screenshots/12-mlflow-vision.jpg), [13](screenshots/13-mlflow-rag.jpg),
  [14](screenshots/14-mlflow-registry.jpg), [15](screenshots/15-mlflow-classifier.jpg).

## 11. DVC (45 s)

- **Faire** (terminal à la racine) :

  ```bash
  vision/.venv/Scripts/dvc.exe status        # Data and pipelines are up to date.
  vision/.venv/Scripts/dvc.exe status -c     # Cache and remote 'localstore' are in sync.
  vision/.venv/Scripts/dvc.exe remote list   # localstore  C:\Users\msi\dvc-storage\tujya  (default)
  vision/.venv/Scripts/dvc.exe dag           # étapes prepare et rag_corpus
  ```

- **Le jury voit** : données et pipelines à jour, remote synchronisé, le graphe des étapes.
- **Dire** : « Les 15 306 fichiers de données (5 Go) sont versionnés par DVC, jamais dans
  git ; un clone neuf récupère tout avec `dvc pull`, ce que j'ai vérifié fichier par
  fichier. »

## 12. GitHub Actions CI/CD (45 s)

- **Faire** : `gh run list --limit 4`, ou l'onglet Actions du dépôt.
- **Le jury voit** : runs `ci` et `cd` verts ; dans un run `cd` : « backend image (build +
  smoke test) », « frontend image (build + smoke test) », job `publish` ignoré.
- **Dire** : « À chaque push : lint, types, tests (pipeline, backend, vision) et build du
  frontend ; puis la CD construit et teste les images Docker, sans les publier, puisque
  l'application ne doit tourner que sur ce PC. »

## 13. Serveur MCP (45 s)

- **Faire** (terminal `backend/`, variables de la préparation) :
  `.venv/Scripts/python scripts/mcp_smoke.py`
- **Le jury voit** (≈ 11 s) : `tools: ['identify_pills', 'check_interaction',
  'get_drug_info', 'search_documents']`, `check_interaction -> … Major`,
  `search_documents -> ['Metformin label …']`.
- **Dire** : « Les mêmes outils que l'agent sont exposés par un serveur MCP officiel, donc
  utilisables depuis n'importe quel client MCP, comme Claude Desktop. »

## 14. Résultats du benchmark (1 min)

- **Faire** : tableau « Résultats du benchmark RAG » du README ou docs/BENCHMARK.md.
- **Le jury voit** : récupération sur 83 questions : dense + filtre Recall@5 **0,928**, MRR
  **0,852** (hybride 0,904 / 0,796 ; BM25 0,783 / 0,660) ; génération : 94,0 % des phrases
  citées soutenues par leur source, 0 dose chiffrée, 0 « sûr », 2/2 questions de dose
  refusées, 4,7 s en moyenne.
- **Dire** : « J'ai mesuré la récupération et la génération sur 88 questions relues à la
  main ; le dense battait l'hybride sur ce jeu, surtout en français, donc c'est le réglage
  par défaut. »

## 14 bis. Bonus : toxicité (optionnel, 45 s)

- **Faire** : menu **Toxicity** -> laisser « warfarin » et « fluconazole » -> **Predict**.
- **Le jury voit** (≈ 1 s) : une carte par molécule, les 12 tests Tox21 avec la probabilité
  d'activité et le ROC-AUC du modèle pour chaque test, le bandeau « Prédiction d'un modèle sur
  des tests in vitro, pas une évaluation clinique », la note « ce modèle ne dit rien de leur
  association ». Exemple « citalopram » : badge « far from the training molecules ».
- **Dire** : « En bonus, un modèle LightGBM sur Tox21 avec un split par scaffold, ROC-AUC
  moyen 0,725 ; et comme le petit LLM lisait mal les probabilités, dans le chat ces phrases
  sont écrites par le code, pas par le modèle. »
- Dans le chat : « La warfarine est-elle toxique selon le modèle Tox21 ? » -> la phrase fixe
  « Modèle Tox21 pour warfarin (…) : … SR-MMP (0,83) [1]. »

## 15. Limites (1 min)

- **Dire**, honnêtement :
  - boîtes de détection produites automatiquement (pas d'annotation humaine ; ~90 % justes
    sur un audit de 48 photos), donc le mAP50 de 0,972 est mesuré contre ces boîtes ;
  - 150 classes fermées : un comprimé hors liste sera mal identifié (d'où la confirmation) ;
  - split par photo, pas par comprimé physique ; pas encore comparé à ePillID ;
  - un LLM de 3B (4 Go de VRAM) ajoute parfois des phrases hors sujet ; 30 % des phrases
    sans citation ;
  - benchmark construit et relu par la même personne, juge local de 4B, RAGAS sur 8 questions.
- **Conclure** : « Tout est reproductible en local, versionné et testé ; avec plus de temps :
  annotations humaines, modèle plus grand, comparaison ePillID. »

---

## Plan B

| Problème | Que faire |
|---|---|
| Première réponse du chat très lente (> 30 s) | Le modèle n'était pas chargé : continuer à parler pendant le chargement, ou montrer la réponse CLI de la préparation ; captures [09](screenshots/09-chat-answer.jpg) et [11](screenshots/11-chat-dose-refused.jpg) |
| La webcam ne reconnaît rien | Le dire (modèle entraîné sur C3PI uniquement) et revenir aux photos de démo |
| `/docs` reste blanc | Pas d'Internet (Swagger UI vient d'un CDN) : montrer <http://localhost:8080/openapi.json> et la capture [07](screenshots/07-swagger.jpg) |
| Un conteneur n'est pas « healthy » | `docker compose logs <service> --tail 30`, puis `docker compose up -d` |
| Scan en 503 | `models/` vide : `cd vision && .venv/Scripts/python -m tujya_vision.publish` (MLflow démarré) |
| Pas d'Internet pour GitHub | `gh run list` ne marchera pas : montrer `.github/workflows/` et le rapport de session 4 |
| Le script MCP met plus d'une minute | une adresse contient `localhost` : la remplacer par `127.0.0.1` |

## Rejouer la démo automatiquement

```bash
cd frontend
BASE=http://localhost:8080 node scripts/screenshots.mjs        # scan, interactions, webcam, Swagger
BASE=http://localhost:8080 node scripts/chat-screenshots.mjs   # chat, citation, dose refusée
node scripts/mlflow-screenshots.mjs                            # MLflow (expériences, registre)
```

Chaque script échoue si une étape n'affiche pas ce qu'elle doit ; `OUT=<dossier>` écrit les
captures ailleurs que dans `docs/screenshots/`.

# Corpus RAG et recherche

Les documents à partir desquels l'agent de Tujya répond : ce qu'ils sont, comment ils ont été
collectés, comment ils sont découpés en chunks, quelles métadonnées porte chaque chunk,
comment on y cherche et comment les consulter. Construit le 2026-10-28, index dans Qdrant
depuis le 2026-11-11. Décisions : D59 à D62, D113 (Qdrant), D114 (page Documents), D116 à
D118 (BM25 dans Qdrant, reranker, réécriture de requête) dans
[docs/decisions.md](../decisions.md).

```
config/pill_classes.csv (82 principes actifs des 150 classes de comprimés)
  -> notices openFDA        python -m src.data.corpus labels   -> data/rag/raw, interim
  -> résumés PubMed         python -m src.data.corpus pubmed   -> data/rag/raw, interim
  -> paires DDInter         data/processed/pharma/interactions_between_classes.csv (interactions.md)
  -> chunks + métadonnées   dvc repro rag_corpus  (= python -m src.data.corpus corpus)
                            -> data/processed/rag/corpus/chunks.parquet, summary.json
  -> embeddings + index     python -m src.rag.index  (venv du projet `.venv`)
                            -> collection Qdrant tujya_chunks (http://localhost:6333/dashboard)
  -> consultation           page « Documents » de l'application, GET /api/documents
```

Commandes lancées depuis la racine du dépôt avec le venv du projet (`.venv`) ;
sans `.env`, ajouter `--set http.require_contact_email=false` (comme dans
[docs/module2/interactions.md](interactions.md)). Les deux étapes de collecte réutilisent les connecteurs
du pipeline ([docs/data_pipeline/](../data_pipeline/README.md) : PoliteClient avec limites de débit,
reprises, cache HTTP, manifeste avec SHA-256 et provenance), dans leur propre dossier
`data/rag/` avec leur propre manifeste `data/rag/manifest.json`, pour ne jamais se mêler aux
fichiers du projet d'origine sous `data/raw`.

## Sources et volumes

| Type de document | Source | Licence | Documents | Chunks |
|---|---|---|---:|---:|
| `label` | notices openFDA (SPL), https://open.fda.gov/apis/drug/label/ | CC0 / domaine public américain (conditions openFDA) | 82 (une par principe actif) | 2 895 |
| `abstract` | PubMed par les E-utilities du NCBI | conditions NLM ; les résumés peuvent relever du droit d'auteur de l'éditeur (usage de recherche, non redistribués) | 1 570 | 3 416 |
| `interaction` | DDInter 2.0 (paires entre les 82 principes actifs) | CC BY-NC-SA 4.0 | 2 350 | 2 350 |
| **Total** | | | **4 002** | **8 661** |

Les 82 principes actifs ont tous une notice et entre 17 et 20 résumés (19,7 en moyenne).
Longueur moyenne d'un chunk : 687 caractères (notices 1 164, résumés 627).

### Notices openFDA (connecteur `openfda_label`)

[`src/data/tujya_data/connectors/openfda.py`](../../src/data/tujya_data/connectors/openfda.py),
enregistré avec les autres connecteurs (désactivé par défaut, pour que l'exécution
`tujya-data update` ne le lance pas). Pour chaque principe actif, il demande à
`api.fda.gov/drug/label.json` les 25 notices les plus récentes (`sort=effective_time:desc`)
dont `openfda.generic_name` ou `openfda.substance_name` contient le principe actif (ou un
alias de recherche : valproate -> valproic acid / divalproex, vitamin B12 -> cyanocobalamin,
vitamin B6 -> pyridoxine), produits sur ordonnance d'abord, puis tout type de produit si
aucun ne correspond.

**Règle de sélection (D60)**, le premier critère qui diffère décide :

1. la notice porte sur le principe actif (l'un de ses noms génériques ou de substance
   contient chaque mot du nom : « CYCLOBENZAPRINE HYDROCHLORIDE » porte sur cyclobenzaprine) ;
2. produit à un seul principe actif avant une association ;
3. sur ordonnance (`HUMAN PRESCRIPTION DRUG`) avant OTC ;
4. le plus de sections indexées (une notice de reconditionneur sans « drug interactions »
   perd) ;
5. le `effective_time` le plus récent.

Résultat : 81 notices sur ordonnance et 1 OTC (fexofénadine, vendue sans ordonnance aux
États-Unis) ; 80 à un seul principe actif ; la pseudoéphédrine et la vitamine B6 n'existent
qu'en association (brompheniramine + pseudoephedrine + dextromethorphan ; doxylamine +
pyridoxine). Dates d'effet du 2021-09-30 au 2026-09-11, 80 sur 82 depuis 2024. Chaque fichier
brut `data/rag/raw/openfda_label/current/labels/<principe actif>.json` garde la notice
choisie et la liste des candidates, pour que le choix soit vérifiable. Débit : 3 requêtes/s
(openFDA autorise 240/min sans clé) ; les réponses sont en cache sur le disque.

**Sections gardées** (dans l'ordre de la notice) : boxed_warning, indications_and_usage,
contraindications, warnings_and_cautions, warnings, precautions, general_precautions,
drug_interactions, drug_and_or_laboratory_test_interactions, adverse_reactions,
use_in_specific_populations, pregnancy, lactation, nursing_mothers, pediatric_use,
geriatric_use, information_for_patients, mechanism_of_action. Les notices d'ancien format
rangent les interactions sous `precautions` ; les deux formats sont couverts. 11,9 sections
par notice en moyenne.

**Jamais stockées (D60) :** `dosage_and_administration`, `dosage_forms_and_strengths`,
`overdosage`, `how_supplied`, `package_label_principal_display_panel`, et les sections
chiffrées `clinical_pharmacology`, `pharmacokinetics`, `clinical_studies`. Elles sont retirées
avant l'écriture du fichier brut, le schéma des lignes refuse toute section nommée « dosage »
ou « overdos », et le découpage les refuse une troisième fois. L'agent ne peut donc pas citer
une section de posologie. Limite : d'autres sections mentionnent encore des quantités
(1 010 chunks sur 8 661 contiennent « <nombre> mg », par exemple « patients receiving 200 mg »
dans les effets indésirables) ; le nœud de sécurité de l'agent retire de ses réponses toute
phrase contenant une dose ([docs/module2/agent.md](agent.md)).

### Résumés PubMed (connecteur `pubmed` existant, mode `ingredients`)

Le connecteur PubMed a reçu un troisième mode de requête, `ingredients` (D61) : une requête
par principe actif, `<bloc principe actif> AND (interactions OR adverse effects) AND
hasabstract AND english[la]`, où le bloc du principe actif est celui qui existait déjà
(`[tiab]`, `[mh]`, `[nm]`) et le bloc du sujet est `"Drug Interactions"[mh] OR
interaction*[tiab] OR "adverse effects"[sh] OR "Drug-Related Side Effects and Adverse
Reactions"[mh] OR "adverse effect*"[tiab] OR "adverse event*"[tiab] OR "adverse
reaction*"[tiab] OR contraindicat*[tiab]`, publiés depuis le 2000-01-01, triés par
pertinence (`sort=relevance`, nouvelle option), **au plus 20 notices par principe actif**.
Chaque requête a atteint ce plafond (82 requêtes tronquées) : PubMed en contient bien plus.
1 617 résultats, 1 570 articles distincts (44 trouvés par deux principes actifs), années 1998
à 2026 (médiane 2014) : 805 recherches, 266 cas cliniques, 248 revues, 178 essais cliniques,
65 méta-analyses, 8 commentaires ; **4 articles rétractés** (17 chunks), gardés avec
`is_retracted = true` et exclus de la recherche par défaut.

### Lignes DDInter

Les 2 350 paires dont les deux principes actifs appartiennent aux classes de comprimés
([docs/module2/interactions.md](interactions.md)), un chunk chacune : « DDInter 2.0 lists an interaction
with severity Major between fluconazole and warfarin. DDInter's bulk files give only the two
drugs and the level, not the mechanism. » « Unknown » s'écrit « an interaction whose severity
DDInter does not grade », jamais « no interaction ». Les 40 508 paires avec n'importe quel
médicament restent hors de l'index texte (elles noieraient les autres documents sous des
phrases presque identiques) ; l'outil `check_interaction` les lit toutes dans la base (D62).

## Découpage ([`src/data/corpus/corpus.py`](../../src/data/corpus/corpus.py))

- **Notices : par section.** Un chunk par section ; une section de plus de 1 500 caractères
  est coupée en fins de phrase, en morceaux de 1 500 caractères au plus, la dernière phrase
  d'un morceau étant répétée au début du suivant (chevauchement d'une phrase). Une seule
  « phrase » plus longue que la limite (tableaux aplatis) est coupée sur les espaces.
- **Résumés : par paragraphe.** Les résumés structurés ont un paragraphe par étiquette
  (« BACKGROUND: », « RESULTS: »...), qui devient la `section` du chunk ; les résumés non
  structurés forment un seul paragraphe (`section = "abstract"`) ; les paragraphes longs sont
  coupés comme ci-dessus.
- **DDInter : un chunk par paire.**
- Chaque chunk reçoit un titre qui donne son contexte (« Fluconazole label (Diflucan): Drug
  interactions », le titre de l'article, « DDInter 2.0: fluconazole + warfarin (Major) »). Le
  titre et le texte sont ce que voient BM25 et les embeddings.
- `chunk_id` est une empreinte de (document, section, position) : la même entrée donne les
  mêmes identifiants, et l'indexation peut être incrémentale.

## Schéma des métadonnées (une ligne par chunk, `data/processed/rag/corpus/chunks.parquet`)

| Colonne | Type | Sens |
|---|---|---|
| `chunk_id` | chaîne | identifiant stable (16 caractères hexadécimaux) |
| `doc_id` | chaîne | `openfda:<id de notice>`, `pubmed:<PMID>`, `ddinter:<id a>:<id b>` |
| `doc_type` | chaîne | `label`, `abstract` ou `interaction` |
| `ingredients` | liste de chaînes | le ou les principes actifs dont parle le document (noms RxNorm en minuscules) |
| `mentioned_ingredients` | liste de chaînes | autres principes actifs des classes cités dans le texte du chunk |
| `section` | chaîne | section de la notice (par exemple `drug_interactions`), étiquette du paragraphe du résumé (`results`), ou `interaction` |
| `chunk_index` | entier | position du chunk dans son document |
| `title` | chaîne | titre de contexte (voir plus haut) |
| `text` | chaîne | texte du chunk |
| `source_name` | chaîne | « openFDA drug label », « PubMed (<revue>) », « DDInter 2.0 » |
| `url` | chaîne | page DailyMed de la notice (`drugInfo.cfm?setid=...`), page PubMed, DDInter |
| `date` | chaîne | `effective_time` de la notice (AAAAMMJJ), année de publication, année de version de DDInter |
| `version` | chaîne | `set_id` et version de la notice, `PMID <n>`, `DDInter 2.0` |
| `license` | chaîne | licence de la source |
| `article_type` | chaîne ou null | PubMed seulement : research, review, meta_analysis, clinical_trial, case_report, commentary |
| `is_retracted` | booléen | PubMed seulement (publication rétractée ou « RetractionIn ») ; faux ailleurs |
| `language` | chaîne | `en` |
| `retrieved_at` | chaîne | date de téléchargement de la source (UTC, ISO 8601) |

## Versionnement (DVC)

- `data/rag/raw.dvc`, `data/rag/interim.dvc`, `data/rag/manifest.json.dvc` : les notices et
  le XML PubMed téléchargés, les tables analysées et le manifeste (URL, SHA-256, taille,
  licence, date de téléchargement de chaque fichier brut), suivis par `dvc add` (ils viennent
  du réseau, ce n'est donc pas une étape de pipeline).
- Étape `rag_corpus` de [`dvc.yaml`](../../dvc.yaml) : `python -m src.data.corpus
  corpus`, de `data/rag/interim`, `data/rag/raw/pubmed`, des deux tables DDInter et
  principes actifs et du code de découpage vers `data/processed/rag/corpus` (`dvc.lock` enregistre les
  empreintes). `dvc repro rag_corpus` reconstruit les chunks dès qu'une entrée change.
- Envoyé sur le remote DVC par défaut avec les images (dossier local hors du dépôt, D71,
  [docs/infra.md](../infra.md)) ; un nouveau clone récupère le corpus par `dvc pull`.
- Rien sous `data/` n'est commité dans git (certaines sources interdisent la redistribution).

## Embeddings et index ([`src/rag/`](../../src/rag/), dans l'API)

- **Embeddings : bge-m3** (1 024 dimensions, multilingue) par `/api/embed` d'Ollama (D59).
  Chaque chunk est encodé sous la forme `titre + "\n" + texte`.
- **Stockage : Qdrant** (D113, comme dans le cours du module 2), service `qdrant` de
  [docker-compose.yml](../../docker-compose.yml) (`qdrant/qdrant:v1.20.0`, port
  127.0.0.1:6333, volume persistant, télémétrie coupée), code
  [`src/rag/qdrant_store.py`](../../src/rag/qdrant_store.py). Collection `tujya_chunks` :
  un point par chunk, vecteur nommé `dense` (1 024 dimensions, cosinus) ; identifiant du point
  = UUID v5 du `chunk_id` (Qdrant n'accepte que des entiers ou des UUID), le `chunk_id` reste
  dans le payload. **Payload = toutes les métadonnées du schéma ci-dessus** (source, type de
  document, section, langue, date, version, licence, principes actifs, URL, rétractation),
  plus `content_hash` (SHA-256 du texte, des métadonnées et du modèle) et `embedding_model`.
  Index de payload (mot-clé) sur les champs filtrés : `chunk_id`, `doc_id`, `doc_type`,
  `ingredients`, `mentioned_ingredients`, `section`, `language`, `source_name`, et booléen sur
  `is_retracted`. Modèle, dimension, nombre de chunks et `indexed_at` dans les métadonnées de
  la collection.
- **Ancien stockage, gardé** : table `rag_chunks` de la base PostgreSQL `tujya` (pgvector ;
  SQLite dans les tests), même interface ([`src/rag/store.py`](../../src/rag/store.py),
  choix par `TUJYA_VECTOR_STORE=qdrant|sql`, [`src/rag/stores.py`](../../src/rag/stores.py)).
  `python -m src.rag.index --copy-from-sql` a copié ses 8 661 vecteurs dans Qdrant sans les
  réencoder : les deux index contiennent exactement les mêmes vecteurs.
- **Indexation :** `python -m src.rag.index` (venv du projet `.venv` ; `--store`,
  `--qdrant-url`, `--database-url`, `--chunks`, `--pull` pour télécharger d'abord les modèles
  Ollama).
  Idempotente : un chunk n'est réencodé que si son empreinte (texte, métadonnées, modèle) a
  changé, les chunks disparus du corpus sont supprimés, un nouveau modèle d'embeddings
  reconstruit la collection. Exécution du 2026-10-28 : 8 661 chunks encodés en 364 s sur la
  RTX 2050 ; deuxième passage `{"embedded": 0, "unchanged": 8661, "seconds": 0.1}` (même
  résultat sur Qdrant le 2026-11-11).

## Recherche ([`src/rag/retrieval.py`](../../src/rag/retrieval.py))

| Mode | Fonctionnement | Temps par requête (8 661 chunks, portable) |
|---|---|---:|
| `bm25` | Okapi BM25 (k1 1,2, b 0,75) en mémoire sur titre + texte ; minuscules, accents retirés, mots vides anglais et français, « s » du pluriel retiré | ~6 ms |
| `dense` | vecteur bge-m3 de la requête, recherche cosinus dans Qdrant avec le filtre de payload (en SQL : `ORDER BY embedding <=> :q`, filtres dans le WHERE) | 110-160 ms (surtout l'encodage de la requête) |
| `hybrid` | les deux listes (50 candidats chacune) fusionnées par reciprocal rank fusion, score = somme de 1 / (60 + rang) | ~170 ms |

Le chargement du moteur de recherche (tous les textes pour BM25) prend 1,1 s, une seule
fois ; il se recharge seul quand `indexed_at` change. Pas d'index HNSW : à cette taille, le
parcours exact est rapide et exact.

**Filtres** (tous les modes) : `ingredients` (chunks qui portent sur l'un d'eux OU le
citent), `doc_types` (label, abstract, interaction), `sections`, `exclude_retracted` (vrai
par défaut). Dans Qdrant : filtre `should` (`ingredients` OU `mentioned_ingredients` dans la
liste), `must` sur `doc_type` et `section`, `must_not` sur `is_retracted` ; en SQL :
`ingredients && :names OR mentioned_ingredients && :names`.

Pour essayer : `python -m src.rag.search "warfarin bleeding" --ingredient
fluconazole --doc-type label` (affiche les trois modes avec leurs rangs et leurs temps).

Le meilleur mode est mesuré dans [docs/module2/benchmark.md](benchmark.md) : dense avec le filtre par
principe actif est le réglage par défaut de l'agent depuis D65 (`TUJYA_RAG_SEARCH_MODE`).

## Recherche avancée (étape 3.2)

Ce que le cours du module 2 montre (BM25, hybride dense + sparse, reranking, réécriture de
requête, filtres de métadonnées), mesuré dans [benchmark.md](benchmark.md#recherche-avancée-étape-32).

- **BM25 dans Qdrant (D116)** : chaque point a aussi un vecteur sparse `bm25`
  ([`src/rag/sparse.py`](../../src/rag/sparse.py)), produit par le modèle `Qdrant/bm25` de
  fastembed (celui qu'enveloppe `FastEmbedSparse` de LangChain dans le TP : mots vides anglais,
  racinisation Snowball, termes hachés) ; la collection lui applique le modificateur IDF, donc
  Qdrant calcule lui-même le BM25 complet. Le mode `bm25` et le mode `hybrid` tournent alors dans
  Qdrant ; le stockage SQL garde le BM25 en mémoire.
- **Hybride dans Qdrant** : une requête `query_points` avec deux `prefetch` (50 candidats
  denses, 50 candidats BM25, chacun avec les filtres de métadonnées) fusionnés par Qdrant en RRF.
  Qdrant compte les positions à partir de 0 : `k = 61` chez lui donne exactement 1 / (60 + rang),
  la fusion du stockage SQL (vérifié par un test, en mémoire et sur le serveur).
- **Reranker cross-encoder (D117, D123)** : un cross-encoder ONNX par fastembed
  ([`src/rag/rerank.py`](../../src/rag/rerank.py)), donc sans PyTorch dans l'API ;
  `Xenova/ms-marco-MiniLM-L-6-v2` par défaut depuis l'étape 3.3 (`TUJYA_RAG_RERANK_MODEL` ;
  `BAAI/bge-reranker-base`, le modèle du TP, reste possible). La recherche renvoie
  `max(k, fetch_k)` candidats (`fetch_k` 10 par défaut, `TUJYA_RAG_FETCH_K`), le cross-encoder
  lit la question avec chacun (titre + texte) et garde les `k` meilleurs ; chaque source garde son
  score et son rang avant reranking. Désactivé par défaut (`TUJYA_RAG_RERANK`) : 3 à 11 s par
  question sur le processeur de ce PC ([benchmark_retrieval.md](benchmark_retrieval.md)).
- **Réécriture de requête (D118)** : le LLM réécrit la question en une requête de recherche
  anglaise courte qui garde les noms de médicaments ([`src/rag/rewrite.py`](../../src/rag/rewrite.py)) ;
  la recherche (et le reranking) portent sur cette requête, la réponse sur la question. Une
  réécriture vide, trop longue ou en erreur garde la question ; les mots collés par un petit
  LLM (`what_are_the_...`) sont séparés. **Activée par défaut depuis l'étape 3.3**
  (`TUJYA_RAG_QUERY_REWRITE`, D123) : champion de la grille de récupération, Recall@5 0,906 ->
  0,941 pour environ 0,3 s de plus.
- **Filtres de métadonnées** (tous les modes, dans Qdrant par index de payload) : principe actif
  (dont parle ou que cite le chunk), type de document, section, publications rétractées ;
  l'outil `search_documents` les expose à l'agent et au serveur MCP.
- **Sources dans le chat** : chaque citation montre son type, sa section, sa date, sa source, sa
  licence et, si la recherche a été reranquée, le score du reranker ; l'identifiant du document
  renvoie à la page Documents.

Modèles fastembed téléchargés une fois depuis Hugging Face (sans compte) dans
`TUJYA_RAG_MODEL_CACHE_DIR` (`models/fastembed`, git-ignoré ; volume `fastembed` dans Docker) :
`Qdrant/bm25` (quelques ko), `BAAI/bge-reranker-base` (1,1 Go). `python -m src.rag.index` ajoute
les vecteurs sparse à une collection plus ancienne sans rien réencoder (copie des points, 9 s pour
8 661 chunks le 2026-11-11), puis ne recalcule les vecteurs sparse que si le modèle change.

## Consulter les documents indexés

Remarque du professeur : « la base vectorielle n'est pas accessible, on ne voit pas les
documents utilisés ». Trois façons de les voir (D114) :

- **Tableau de bord de Qdrant** : <http://localhost:6333/dashboard> (service `docker compose`
  `qdrant`), collection `tujya_chunks` : les points, leur payload complet et leur vecteur,
  recherche filtrée par payload (par exemple `doc_type:label`). Capture
  [29](../screenshots/29-qdrant-dashboard.jpg).
- **Page « Documents » de l'application** (menu de l'utilisateur connecté,
  [`DocumentsPage.tsx`](../../frontend/src/pages/DocumentsPage.tsx)) : carte de l'index (Qdrant,
  collection, 4 002 documents, 8 661 points, modèle d'embeddings, date d'indexation, champs
  indexés, lien vers le tableau de bord), liste paginée des documents avec leur type, source,
  principes actifs, date et nombre de chunks, filtres par mots du titre, principe actif (dont
  parle le document ou qu'il cite), type et source. Un clic ouvre le document : toutes ses
  métadonnées (identifiant, source, date, version, langue, licence), le lien vers la source
  d'origine et ses chunks dans l'ordre, chacun avec sa section, son identifiant et son
  SHA-256. Captures [27](../screenshots/27-documents.jpg) et
  [28](../screenshots/28-document-chunks.jpg).
- **API** : `GET /api/documents` et `GET /api/documents/{doc_id}` (compte connecté,
  [docs/api.md](../api.md#get-apidocuments)).

Les documents sont reconstruits à partir des chunks de l'index
([`src/rag/documents.py`](../../src/rag/documents.py)) : la page montre exactement ce que
l'agent peut retrouver, pas une copie séparée. Les phrases de posologie chiffrée sont
remplacées par « […] » à l'affichage, comme dans les réponses de l'agent. Vérification dans un
vrai navigateur : `cd frontend && node scripts/documents-screenshots.mjs` (contrôle la carte
de l'index, les trois types, le filtre type + principe actif et le détail d'un document ;
le 2026-11-11 : 28 notices sur 4 002 documents pour « notice + warfarine »).

## Tests

- [`tests/rag/test_rag.py`](../../tests/rag/test_rag.py) (SQLite, embeddings factices par hachage) :
  idempotence de l'index, changement de modèle, BM25, les trois modes, filtres, rechargement.
- [`tests/rag/test_rag_pgvector.py`](../../tests/rag/test_rag_pgvector.py) : la même chose sur
  PostgreSQL + pgvector, dans son propre schéma (job CI `module2`, avec un service
  `pgvector/pgvector:pg16`).
- [`tests/rag/test_rag_qdrant.py`](../../tests/rag/test_rag_qdrant.py) (client Qdrant en
  mémoire) : collection, payload complet et index de payload, indexation idempotente qui suit
  le corpus, changement de modèle, recherche dense et filtres, mêmes classements que le
  stockage SQL, copie depuis SQL sans encodage, rechargement après réindexation.
- [`tests/rag/test_hybrid_rerank.py`](../../tests/rag/test_hybrid_rerank.py) (encodeur sparse
  local et reranker factice, Qdrant en mémoire et serveur si `TUJYA_TEST_QDRANT_URL`) : poids
  BM25, vecteurs sparse écrits une fois et recalculés seulement pour un autre encodeur, migration
  d'une ancienne collection sans encodage, BM25 de Qdrant = BM25 en mémoire, fusion RRF de Qdrant =
  la nôtre, filtres sur sparse et hybride, reranking (fetch_k lus, k gardés, égalités), nettoyage
  et repli de la réécriture ; [`tests/agent/test_search_tool.py`](../../tests/agent/test_search_tool.py) :
  l'outil avec reranking, requête réécrite et métadonnées des sources.
- [`tests/rag/test_documents.py`](../../tests/rag/test_documents.py) : regroupement des chunks
  en documents, filtres et facettes, `/api/documents` (connexion exigée, filtres, détail avec
  chunks, posologie masquée, 503 si l'index est vide ou Qdrant injoignable), chat sur Qdrant.
- [`tests/data/test_corpus.py`](../../tests/data/test_corpus.py) (respx, sans réseau) : règle de sélection
  des notices, posologie jamais stockée, exécution du connecteur (une notice par principe
  actif, 404 géré, fichier brut réutilisé), requêtes PubMed par principe actif et leurs
  résultats, découpage et chevauchement, métadonnées des chunks, identifiants stables,
  posologie refusée par le découpage.

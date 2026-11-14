# Benchmark du RAG : récupération et génération

Mesuré le 2026-10-28 sur le portable (RTX 2050 4 Go, Docker Desktop), avec le vrai corpus
(8 661 chunks, [docs/module2/rag.md](rag.md)), les embeddings bge-m3 et qwen2.5:3b ([docs/module2/agent.md](agent.md)).
Chaque exécution est journalisée dans MLflow, expérience **`tujya-rag`**
(http://localhost:5000) : paramètres, métriques, résultats par question en artefacts.
Décisions : D65 à D68.

## Le jeu de questions ([`src/benchmark/`](../../src/benchmark/))

**88 questions, 44 en français et 44 en anglais**, sur les 82 principes actifs des classes :

| Type | Questions | Chunks attendus (règle) |
|---|---:|---|
| `interaction` (A, B) | 24 | chunks de la notice de A dans les sections interactions / mises en garde / précautions / contre-indications qui citent B, ceux de B qui citent A, et la ligne DDInter de la paire |
| `contraindications` | 12 | les chunks de contre-indications de la notice |
| `boxed_warning` | 10 | les chunks de l'encadré (boxed warning) |
| `mechanism` | 10 | les chunks du mécanisme d'action |
| `adverse_reactions` | 10 | les chunks des effets indésirables |
| `warnings` | 8 | les chunks des mises en garde et précautions (ou mises en garde) |
| `abstract` | 9 | les chunks d'un article PubMed (cas cliniques, une revue systématique, une méta-analyse) |
| `safety` | 5 | aucun : deux demandes de dose, deux questions « est-ce sûr », une question d'horaire ; notées sur la réponse seulement |

Construction ([`bench_build.py`](../../src/benchmark/bench_build.py), spécification
[`spec.json`](../../src/benchmark/spec.json) écrite par [`make_spec.py`](../../src/benchmark/make_spec.py)) :

1. Candidats tirés du corpus : paires dont la notice de l'un cite l'autre principe actif dans
   une section interactions ou mises en garde et que DDInter note (178 paires ; 24 choisies
   pour couvrir de nombreuses classes de médicaments) ; principes actifs dont la notice a la
   section demandée (par exemple 34 notices ont un encadré ; 10 choisies) ; cas cliniques et
   revues PubMed avec un résultat clair (9 choisis dans un échantillon aléatoire de 50 titres).
2. Questions issues de modèles fixes en français et en anglais (DCI et articles français,
   par exemple « Quelles sont les contre-indications de la metformine ? »), en alternant les
   langues.
3. Chunks attendus selon les règles ci-dessus ; réponses de référence extraites du premier
   chunk attendu (les phrases avec une dose, une consigne de posologie ou un numéro de
   déclaration sont sautées, les renvois internes des notices retirés).
4. **Relu à la main** : chaque candidat a été lu. Corrigés : la grammaire française des
   modèles ; un encadré vide (gemfibrozil : la section n'est qu'un titre) remplacé par la
   carbamazépine ; un principe actif sans section d'effets indésirables (fexofénadine)
   remplacé ; **14 références d'interaction réécrites à la main** à partir des phrases de la
   notice de la paire, parce que la phrase extraite était hors sujet (par exemple
   « propranolol does not affect the pharmacokinetics of verapamil » au lieu de la mise en
   garde sur la bradycardie) ou citait une dose (« do not exceed simvastatin 20 mg ») ; les 9
   références des résumés sont des paraphrases écrites après lecture de chaque résumé.

**Limites.** Les chunks attendus sont trouvés par des règles sur une notice par principe
actif : un autre chunk peut aussi répondre à la question et compte alors comme un échec (les
scores sont une borne basse). Les questions viennent de modèles, elles sont donc plus propres
que celles de vrais utilisateurs. Le jeu a été construit par la personne qui a construit le
système. Les questions sur les résumés n'ont qu'un article attendu alors que d'autres
peuvent être pertinents. Les références sont extractives (phrases de notice) pour 60 des 83
questions ; 23 ont été écrites à la main.

## Récupération

[`bench_retrieval.py`](../../src/benchmark/bench_retrieval.py) : les 83 questions qui ont des chunks
attendus, top 10, six configurations (trois modes x deux réglages du filtre). **Recall@k** =
part des questions avec au moins un chunk attendu dans le top k (taux de succès) ;
**Coverage@10** = part moyenne des chunks attendus d'une question dans le top 10 ; **MRR** =
moyenne de 1 / rang du premier chunk attendu (top 10). Filtre `names` = les principes actifs
que l'agent trouve dans la question (noms français ou anglais) utilisés comme filtre de
métadonnées, comme le fait l'agent ; `none` = la question brute seule.

| Mode | Filtre | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Coverage@10 | ms / requête |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | none | 0,554 | 0,711 | 0,735 | 0,783 | 0,637 | 0,671 | 5 |
| dense (bge-m3) | none | 0,783 | 0,867 | 0,904 | 0,976 | 0,841 | 0,823 | 295 |
| hybride (RRF) | none | 0,699 | 0,855 | 0,892 | 0,940 | 0,779 | 0,804 | 211 |
| BM25 | names | 0,566 | 0,747 | 0,783 | 0,819 | 0,660 | 0,710 | 26 |
| **dense (bge-m3)** | **names** | **0,795** | **0,880** | **0,928** | **0,976** | **0,852** | **0,829** | 180 |
| hybride (RRF) | names | 0,711 | 0,892 | 0,904 | 0,928 | 0,796 | 0,791 | 193 |

Par langue (Recall@5 / MRR) :

| Configuration | Français | Anglais |
|---|---|---|
| BM25, names | 0,585 / 0,449 | 0,976 / 0,865 |
| dense, names | 0,878 / 0,765 | 0,976 / 0,937 |
| hybride, names | 0,805 / 0,643 | **1,000 / 0,944** |

Par type de question (Recall@5, filtre `names`) : le mode dense est à 1,00 pour les
contre-indications, les mécanismes et les effets indésirables, 0,92 pour les interactions,
0,89 pour les résumés, 0,90 pour les encadrés, 0,75 pour les mises en garde ; l'hybride est
meilleur pour les encadrés (1,00) et les interactions (0,96) mais bien moins bon pour les
résumés (0,67).

**Choix (D65) : recherche dense avec le filtre par principe actif**, réglage par défaut de
l'agent (`TUJYA_RAG_SEARCH_MODE=dense`) : meilleurs Recall@1, Recall@5 et MRR au total.
BM25 ne reconnaît que les mots anglais, il échoue donc sur les questions françaises
(Recall@5 0,59) et tire l'hybride vers le bas ; sur les seules questions anglaises, l'hybride
est un peu meilleur (MRR 0,944 contre 0,937). Un choix selon la langue (hybride pour
l'anglais) ou une requête BM25 traduite serait l'étape suivante ; ce n'est pas réglé ici,
pour ne pas ajuster le système à son propre jeu de test. Le filtre par principe actif aide un
peu chaque mode (+0,01 à +0,05 de Recall@5).

### Qdrant contre pgvector (étape 3.1, D113)

Le passage à Qdrant ne change pas la récupération : `python -m src.rag.index
--copy-from-sql` a copié les mêmes vecteurs bge-m3, et `bench_retrieval.py --store qdrant`
puis `--store sql` donnent, configuration par configuration, **exactement** les mêmes
Recall@k, Coverage@k et MRR (écart 0). Seules les latences changent (dense + filtre :
112 ms par requête sur Qdrant, 107 ms sur pgvector, surtout l'encodage de la question).

| Jeu de questions | Store | dense + filtre : Recall@5 / MRR | hybride + filtre | BM25 + filtre |
|---|---|---:|---:|---:|
| 83 questions (tableau ci-dessus) | pgvector puis Qdrant | 0,928 / 0,852 | 0,904 / 0,796 | 0,783 / 0,660 |
| 85 questions (+ 2 questions de marque de l'étape 2.7) | Qdrant et pgvector, identiques | 0,906 / 0,832 | 0,882 / 0,778 | 0,765 / 0,644 |

La baisse de 0,928 à 0,906 vient du jeu de questions, pas du store : l'étape 2.7 a ajouté
huit questions de marque (« Panadol... »), dont deux ont des chunks attendus ; aucune des deux
n'est trouvée dans le top 5 (77 succès sur 83 puis 77 sur 85), car un nom de marque n'est pas
dans les notices américaines : l'agent le traduit en principe actif par la table des marques
avant de chercher (D112), ce que le benchmark de récupération seul ne fait pas. Runs MLflow
(`tujya-rag`, 11/11/2026) : paramètre `store` de chaque run (`qdrant` ou `pgvector`) ;
[`results_retrieval.json`](../../src/benchmark/results_retrieval.json) = le run Qdrant.

### Recherche avancée (étape 3.2, D116-D118)

Même jeu (85 questions avec chunks attendus), même index Qdrant, filtre `names` ; BM25 et hybride
tournent maintenant dans Qdrant (vecteurs sparse `Qdrant/bm25`). `-rewrite` : requête réécrite
par qwen2.5:3b ; `-rerank` : bge-reranker-base relit les 10 premiers résultats (`--fetch-k 10`,
pour tenir le temps de calcul sur ce PC : le Recall@10 ne peut donc pas changer). Runs MLflow
`tujya-rag` du 11/11/2026 (paramètres `lexical`, `rerank`, `fetch_k`, `query_rewrite`),
[`results_retrieval.json`](../../src/benchmark/results_retrieval.json).

| Configuration (filtre `names`) | Recall@1 | Recall@5 | Recall@10 | MRR | Recall@5 / MRR FR | Recall@5 / MRR EN | ms / requête |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 en mémoire (avant, étape 3.1) | 0,553 | 0,765 | 0,800 | 0,644 | | | 38 |
| BM25 Qdrant (`Qdrant/bm25`) | 0,518 | 0,753 | 0,776 | 0,611 | 0,524 / 0,380 | 0,977 / 0,837 | 20 |
| **dense (réglage de l'agent, D65)** | 0,776 | 0,906 | 0,953 | 0,832 | 0,857 / 0,747 | 0,954 / 0,915 | 102 |
| hybride Qdrant (RRF) | 0,729 | 0,871 | 0,953 | 0,803 | 0,762 / 0,672 | 0,977 / 0,930 | 124 |
| BM25 + réécriture | 0,694 | 0,882 | 0,953 | 0,779 | 0,786 / 0,690 | 0,977 / 0,866 | 20 + LLM |
| dense + réécriture | 0,765 | 0,941 | **1,000** | 0,835 | 0,929 / 0,782 | 0,954 / 0,887 | 94 + LLM |
| hybride + réécriture | 0,788 | 0,929 | 0,988 | 0,860 | 0,881 / 0,800 | 0,977 / 0,918 | 118 + LLM |
| dense + reranker | 0,800 | 0,941 | 0,953 | 0,861 | 0,905 / 0,810 | 0,977 / 0,911 | 18 341 |
| **hybride + réécriture + reranker** | **0,824** | **0,965** | 0,988 | **0,889** | 0,929 / 0,878 | 1,000 / 0,900 | 12 975 |

Ce que cela montre :

- **La réécriture corrige le défaut connu du lexical** (D65 : BM25 ne lit que l'anglais) :
  BM25 passe de 0,753 à 0,882 de Recall@5 et de 0,524 à 0,786 sur les questions françaises ;
  l'hybride gagne 0,058 de Recall@5 et 0,057 de MRR. Un premier passage donnait l'inverse pour
  BM25 (0,612) : le petit LLM collait parfois les mots (`what_are_the_contraindications_of_metformin`),
  un seul terme inconnu pour BM25 ; corrigé dans `clean()` (test ajouté) puis relancé.
- **Le reranker améliore surtout le haut du classement** : sur le dense, Recall@1 0,776 -> 0,800,
  Recall@5 0,906 -> 0,941, MRR 0,832 -> 0,861 ; sur hybride + réécriture, 0,929 -> 0,965 et
  0,860 -> 0,889 (meilleure configuration mesurée). Il ne peut rien pour une question dont le bon
  chunk n'est pas dans les 10 candidats.
- **Coût** : la réécriture ajoute un appel LLM (environ 1 s sur le GPU) ; le reranker coûte 13 à
  18 s par question sur le processeur de ce portable (10 paires de ~300 tokens, PC bridé sous
  charge, D96), inutilisable en démonstration.
- **Le BM25 de Qdrant n'est pas meilleur que le nôtre** : `Qdrant/bm25` ne retire que les mots
  vides anglais et racinise en anglais, alors que notre tokeniseur retirait aussi les mots vides
  français ; sur les questions anglaises, les deux sont à 0,977.

Réglage de l'agent à cette étape : **inchangé (dense, sans réécriture ni reranker)** en attendant l'étape 3.3,
qui compare les configurations sur la génération (D117, D118). Les deux options sont un réglage :
`TUJYA_RAG_QUERY_REWRITE=true`, `TUJYA_RAG_RERANK=true`, `TUJYA_RAG_SEARCH_MODE=hybrid`.

### Grille de configurations (étape 3.3, D123)

27 configurations (embeddings, modes, découpage, réécriture, rerankers, bge-m3 fine-tuné), un run
MLflow chacune (`tujya-rag-retrieval`), champion enregistré : voir
[benchmark_retrieval.md](benchmark_retrieval.md). **Réglage servi depuis : dense bge-m3 +
réécriture de la requête** (MRR 0,834, Recall@5 0,941, 389 ms par question), sans reranker.

## Génération

[`bench_answers.py`](../../src/benchmark/bench_answers.py) fait tourner le vrai agent (qwen2.5:3b,
recherche dense, nœud de sécurité) sur les 88 questions ;
[`bench_generation.py`](../../src/benchmark/bench_generation.py) note les réponses. Juge :
**gemma3:4b**, un modèle différent du générateur (D67).

**Temps de réponse** (modèles chargés, RTX 2050) : moyenne **4,7 s**, médiane 4,5 s,
maximum 9,8 s par question (planificateur ~1 s, réponse 1 à 7 s). La première question après
un démarrage charge aussi les modèles (10 à 20 s).

**Outils.** Sur les 24 questions d'interaction, le modèle planificateur a lui-même appelé
`check_interaction` ; il a appelé `get_drug_info` sur 59 questions ; `search_documents` a été
ajouté par règle sur 85 (le modèle l'appelle rarement de lui-même).

### Sécurité (les 88 réponses)

| Contrôle | Résultat |
|---|---:|
| Réponses terminées par l'avertissement | 100,0 % |
| Phrases avec une dose chiffrée dans les réponses finales | 0 |
| Phrases avec un conseil de dose sans chiffre (règle ajoutée après l'exécution) | 4 |
| Phrases qualifiant une association de « safe » / « sûr » | 0 |
| Conseil d'horaire sans citation | 0 |
| Questions de dose refusées (avant tout appel au LLM) | 100,0 % (2 / 2) |
| Réponses dont le nœud de sécurité a retiré une phrase | 6 |
| Réponses refusées (question de dose ou aucune source) | 3 |

Les deux questions « est-ce sûr ? » n'ont reçu aucun « sûr » : l'une a répondu avec la note
Major de DDInter (mais sous forme de liste brute de ses sources, un échec du modèle de 3B),
l'autre par « aucune interaction connue dans cette base de données ». La question d'horaire
(lévothyroxine + calcitriol) a reçu la note « Unknown » de DDInter, et sa phrase
d'espacement non citée a été retirée. Les comptes de sécurité sont recalculés avec les
règles finales : **aucune réponse ne donne de dose chiffrée**, mais 4 réponses ont gardé une
phrase d'ajustement de dose sans chiffre, paraphrasée des notices (« dose adjustment of
warfarin may be necessary », « carefully consider dose reduction »). Les règles de cette
exécution ne cherchaient que des quantités ; la règle retire maintenant aussi les conseils de
dose sans chiffre (trouvé sur les conteneurs juste après l'exécution ; tests dans
`test_safety.py`), ces 4 phrases seraient donc retirées aujourd'hui.

### Citations (82 questions avec réponse, 265 phrases de réponse)

| Contrôle | Résultat |
|---|---:|
| Phrases de réponse avec une citation [n] | 69,8 % |
| Réponses avec au moins une citation | 92,7 % |
| **Phrases citées soutenues par la source citée** (juge) | **94,0 %** |
| ... réponses en français / en anglais | 90,0 % / 96,5 % |
| Contrôle du juge : « soutenue » pour une phrase associée à la source d'une autre question | 10,0 % (4 / 40) |

Le contrôle montre que le juge discrimine (94 % contre 10 % de « oui »), mais il est
indulgent : une phrase qui cite le bon passage de la notice tout en ajoutant un détail faux
peut passer. Beaucoup de citations ont été ajoutées après la génération : 93 par le sens
(bge-m3) et 43 par recouvrement de mots (D68) ; le modèle a écrit les autres lui-même. 30 %
des phrases restent sans citation (typiquement les introductions et les paraphrases en
français sans source la plus proche évidente).

### RAGAS 0.4.3 (sous-ensemble stratifié de 8 questions avec réponse, une par type, juge gemma3:4b)

| Métrique | Moyenne | Notées |
|---|---:|---:|
| Fidélité (affirmations de la réponse soutenues par les contextes) | 0,979 | 6 / 8 |
| Pertinence de la réponse (questions régénérées depuis la réponse comparées à la question, bge-m3) | n/d | 0 / 8 |
| Précision du contexte avec référence (contextes utiles classés en premier) | 0,809 | 8 / 8 |

À lire comme point de comparaison pour de futures exécutions, pas comme des valeurs
absolues : un juge de 4B avec la moitié de ses couches sur le CPU, 8 questions (une par type ;
le sous-ensemble a pris la première question de chaque type, donc 7 sur 8 sont en français),
sorties du juge illisibles comptées comme manquantes (« Notées »). **La pertinence de la
réponse n'a pas pu être calculée** : sur les 8 réponses, les questions générées par le juge
n'étaient pas lisibles par RAGAS.

### Ce que montrent les réponses (lues à la main)

- Bien : contre-indications, mécanismes, encadrés et notes DDInter sont le plus souvent
  justes et cités ; aucune dose ni aucun « sûr » n'a atteint une réponse ; les questions de
  dose sont refusées.
- Faible : le modèle de 3B reste souvent proche du texte de la notice et ajoute des phrases
  hors sujet du même passage (par exemple fluconazole + citalopram : tolvaptan et triazolam,
  qui sont dans la section interactions du fluconazole) ; il ajoute parfois un détail absent
  de la source (lévothyroxine : « insuffisance rénale sévère ») ; une fois, il a répondu à une
  question sur l'encadré par la phrase « aucune interaction connue » ; une fois, il a affiché
  ses sources au lieu d'une réponse. Un modèle plus grand (7-8B) demanderait plus de 4 Go de
  mémoire vidéo.

Tout ceci est dans MLflow (`generation-qwen2.5:3b-dense`, `citation-judge-control`) avec les
notes par question ([`generation_per_question.jsonl`](../../src/benchmark/generation_per_question.jsonl),
sans texte des sources).

## Reproduire

```bash
# venv du projet `.venv`, depuis n'importe quel dossier ; pile lancée (postgres, ollama, mlflow)
export TUJYA_PHARMA_DATABASE_URL=postgresql+psycopg://tujya:tujya@127.0.0.1:5432/tujya
python -m src.benchmark.bench_build        # rag_questions.jsonl depuis le corpus
python -m src.benchmark.bench_retrieval    # six exécutions -> MLflow, results_retrieval.json
python -m src.benchmark.bench_answers      # l'agent sur les 88 questions -> data/processed/rag/bench/
python -m src.benchmark.bench_generation   # sécurité, citations, RAGAS -> MLflow, results_generation.json
```

RAGAS et le client MLflow ne sont pas dans `requirements.txt` (benchmark seulement) :
`pip install ragas==0.4.3 --no-deps` plus ses dépendances sauf `scikit-network` (pas de
wheel pour Python 3.14 ; il ne sert qu'au générateur de jeux de test de RAGAS), et
`mlflow-skinny`.

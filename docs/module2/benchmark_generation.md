# Benchmark de la génération : RAGAS, LLM, prompts et MLflow GenAI (étape 3.4)

La génération des réponses de l'agent est comparée sur une grille de configurations : le LLM qui
fait tourner tout l'agent (planificateur d'outils, réécriture de la requête, réponse) et la
version du prompt de réponse. La récupération est le réglage servi de l'étape 3.3 (bge-m3 dense,
filtre par principe actif, réécriture de la requête, sans reranker : D123,
[benchmark_retrieval.md](benchmark_retrieval.md)). Chaque configuration est un run MLflow
(expérience **`tujya-rag-generation`**) ; la meilleure devient la version `champion` du modèle
enregistré **`tujya-rag-generation`** et fixe le réglage de l'agent. Décision : D124
([decisions.md](../decisions.md)). Le premier benchmark de génération (88 questions, RAGAS sur 8)
reste décrit dans [benchmark.md](benchmark.md).

## Ce qui est mesuré

- **Questions** : les **96** questions du jeu ([rag_questions.jsonl](../../src/benchmark/rag_questions.jsonl)),
  à chaque configuration (aucun sous-ensemble). Les questions de dose sont refusées sans appel au
  LLM ; les questions sans référence (dose, « est-ce sûr ? », la plupart des marques) n'ont pas de
  note qui demande une référence.
- **Sécurité** (règles de [project_rules.md](../project_rules.md), vérifiées par
  [safety.py](../../src/agent/safety.py)) : phrases avec une dose, « sûr » / « safe », conseil
  d'horaire sans citation, avertissement final, questions de dose refusées.
- **Citations** : pour chaque phrase qui cite [n], le juge dit si la source n la soutient.
- **RAGAS 0.4.4** (dépendance figée dans [requirements-dev.txt](../../requirements-dev.txt)) sur
  **toutes les réponses** : fidélité (*faithfulness*), pertinence de la réponse (*answer
  relevancy*, embeddings bge-m3), précision du contexte et rappel du contexte (avec la
  référence). Les 4 premières sources de la réponse (900 caractères chacune, comme le LLM les
  voit) sont les contextes.
- **MLflow 3 GenAI** : `mlflow.genai.evaluate` dans le run de la configuration, avec les juges
  intégrés **Correctness** (réponse comparée à la référence) et **Safety**, et trois scorers du
  projet (`@scorer`) : **citation présente**, **aucune dose**, **avertissement présent**. Les
  résultats sont dans l'onglet *Evaluations* du run.
- **Temps** : secondes par question mesurées pendant la génération (modèles chargés, RTX 2050
  4 Go), médiane.

## MLflow : traces, prompts, évaluations

- **Traces** (onglet *Traces* de `tujya-rag-generation`) : `bench_answers --trace` active
  `mlflow.langchain.autolog` ; chaque question est une trace `question` (tags : configuration,
  LLM, version du prompt, identifiant de la question) qui contient les nœuds LangGraph
  (`understand`, `tools`, `retrieve`, `answer`, `safety`), les appels au LLM et **un span par
  appel d'outil** (`check_interaction`, `get_drug_info`, `search_documents`...). MLflow 3.17 perd
  le span d'un run LangGraph lancé sans `serialized` (la trace restait « pas entièrement
  exportée ») : le script lui donne le nom par défaut.
- **Prompts** (onglet *Prompts*) : [register_prompts.py](../../src/benchmark/register_prompts.py)
  enregistre `tujya-answer-fr` et `tujya-answer-en` (version 1 = v1, version 2 = v2, gabarits de
  chat : message système + message utilisateur `{{sources}}` / `{{question}}`) et
  `tujya-tool-planner` (v1). Le texte de référence reste [prompts.py](../../src/agent/prompts.py)
  (l'application ne dépend pas du serveur MLflow) ; le script refuse une version enregistrée dont
  le texte diffère. Chaque run garde l'URI de ses prompts (tags `prompt_fr`, `prompt_en`) ; la
  version championne reçoit l'alias `champion`.
- **Évaluations** : le run de chaque configuration (paramètres, métriques RAGAS, sécurité,
  citations, GenAI ; artefacts : notes par question sans texte des sources, configuration).

## Les deux prompts

- **v1** : le prompt des étapes 3.1 à 3.3 (règles absolues, puis « 3 à 6 phrases courtes :
  d'abord la réponse directe, puis ce que disent les notices »).
- **v2** : une méthode en étapes : repérer les passages qui répondent, **première phrase = la
  réponse directe**, puis **un seul fait d'une seule source par phrase, au plus près de ses
  mots**, rien qui ne soit dans les sources. Les mêmes règles de sécurité (testé :
  [test_prompts.py](../../tests/agent/test_prompts.py)).

## Juge

**phi4-mini** (3,8 B, Microsoft, 2,5 Go dans Ollama), une autre famille que les deux
générateurs, servi comme **`phi4-mini-judge`** : le même modèle avec toutes ses couches sur le GPU
([judge.Modelfile](../../infra/docker/ollama/judge.Modelfile)). Le placement automatique d'Ollama
garde ~1 Go libre et laissait 6 couches sur 33 au CPU : 17 jetons/s générés et 119 jetons/s lus,
contre 35 et 1 775 tout sur le GPU, soit ~6 fois plus rapide sur RAGAS (~50 s par question au
lieu de ~5 min). Les embeddings bge-m3 de RAGAS tournent alors sur le CPU pour ne pas évincer le
juge.

Trois pièges trouvés en route, corrigés dans le code : (1) les juges MLflow passent par l'API
compatible OpenAI d'Ollama **sans limite de jetons** ; phi4-mini y est parfois parti en boucle
jusqu'au délai de 5 min, puis n'a plus produit que du texte incohérent jusqu'au redémarrage
d'Ollama : le Modelfile fixe 1 024 jetons et la chaîne redémarre Ollama avant chaque
évaluation (les notes RAGAS, limitées à 700 jetons, n'ont pas été touchées : même part de
sorties plafonnées qu'avant) ; (2) `mlflow.genai.evaluate` lance 10 juges en parallèle alors
qu'Ollama en sert un à la fois : un seul à la fois ; (3) RAGAS et les juges MLflow dans le même
processus se sont bloqués (verrou d'import entre fils) : RAGAS tourne dans un processus enfant
qui remplit le cache, la télémétrie de MLflow et de RAGAS est coupée (rien ne sort du PC). qwen2.5:3b jugeait trop sévèrement ses propres réponses (D67) et s'est contredit à
l'essai (verdict Safety « non » avec une justification « sûr ») ; gemma3:4b, le juge de D67,
n'est plus installé et ne tient qu'à moitié sur le GPU. Un juge de 4 B est bien plus faible que
les juges de la taille de GPT-4 pour lesquels RAGAS est conçu : **les notes comparent des
configurations de ce système, ce ne sont pas des valeurs absolues**. Le verdict Safety de
phi4-mini est rapporté mais n'est pas une barrière : à l'essai (3 questions, avant les runs), il a
jugé « non sûre » une phrase citée qui donnait la gravité « Major » de DDInter.

## Règle de choix (D124), écrite avant les résultats

Dans [config/rag_generation.yaml](../../config/rag_generation.yaml) :
1. **Barrières** (règles déterministes) : 0 phrase de dose, 0 « sûr », 0 conseil d'horaire non
   cité, avertissement sur 100 % des réponses, 100 % des questions de dose refusées.
2. **Note** = moyenne de fidélité, pertinence de la réponse, précision et rappel du contexte
   (RAGAS) et Correctness (MLflow) ; à 0,01 près de la meilleure, la configuration la plus rapide
   l'emporte ; budget : 15 s médianes par question.
3. La deuxième configuration qui passe les barrières est `challenger`.

Grille (une évaluation complète prend environ 1 h 30 sur ce GPU de 4 Go, la grille garde
donc **deux LLM** plutôt que de réduire le nombre de questions) : qwen2.5:3b avec v1 puis v2 ;
llama3.2:3b avec le meilleur des deux prompts sur qwen2.5:3b.

## Résultats

Une branche `experiment/rag-generation`, un run MLflow par configuration (96 questions
chacune). Note = moyenne des cinq métriques de la règle ; « s / question » = temps médian de
l'agent complet pendant la génération (modèles chargés). Tableau produit par les fichiers de
[src/benchmark/generation/](../../src/benchmark/generation/).

| Configuration | Note | Fidélité | Pertinence | Précision ctx | Rappel ctx | Correctness | Safety | Citations soutenues | Phrases citées | s / question (méd.) | Run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| llama3.2-3b + v1 | 0,768 | 0,906 | 0,747 | 0,933 | 0,739 | 0,512 | 0,833 | 0,936 | 0,796 | 24,2 | `c5f24fbe` |
| qwen2.5-3b + v1 | 0,761 | 0,931 | 0,704 | 0,940 | 0,695 | 0,536 | 0,833 | 0,951 | 0,741 | 4,9 | `dd31625f` |
| qwen2.5-3b + v2 | 0,708 | 0,880 | 0,673 | 0,941 | 0,667 | 0,378 | 0,844 | 0,882 | 0,842 | 4,5 | `31760b16` |

Nombre de réponses notées par métrique (les autres : pas de référence, réponse refusée ou sortie
du juge illisible) et barrières de sécurité :

| Configuration | n fidélité | n pertinence | n précision | n rappel | n Correctness | n Safety | doses | « sûr » | avertissement | doses refusées | réécrites |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3.2-3b + v1 | 77 | 93 | 83 | 76 | 84 | 96 | 0 | 0 | 100% | 100% | 12 |
| qwen2.5-3b + v1 | 85 | 94 | 84 | 76 | 84 | 96 | 0 | 0 | 100% | 100% | 7 |
| qwen2.5-3b + v2 | 87 | 94 | 84 | 78 | 82 | 96 | 0 | 0 | 100% | 100% | 8 |

**Choix (D124)** : les trois configurations passent toutes les barrières (0 dose, 0 « sûr »,
0 conseil d'horaire non cité, avertissement sur 100 % des réponses, questions de dose refusées).
llama3.2:3b a la meilleure note (0,768) mais **24 s médianes par question** : il ne tient pas
avec bge-m3 sur les 4 Go du GPU, Ollama échange les deux modèles à chaque question ; il sort du
budget de 15 s, et son avance sur qwen2.5:3b n'est pas significative (bootstrap apparié sur les
62 questions notées par les deux : +0,023, intervalle à 95 % [-0,028 ; +0,069]). **Champion :
qwen2.5:3b avec le prompt v1** (version 1 du modèle enregistré `tujya-rag-generation`, alias
`champion` ; alias `champion` sur la version 1 de `tujya-answer-fr` et `tujya-answer-en`),
c'est le réglage déjà servi : rien ne change dans l'agent. **Challenger : qwen2.5:3b avec v2**
(version 2). v1 bat v2 de façon significative (+0,050 par question, intervalle [+0,010 ;
+0,089]).

Ce que montrent les chiffres :
- **v2 cite plus mais moins juste** : 84 % des phrases ont une citation (74 % avec v1), mais 88 %
  des phrases citées sont soutenues par leur source (95 % avec v1) ; fidélité 0,880 contre 0,931
  et Correctness 0,378 contre 0,536. En demandant « un fait d'une source par phrase », le modèle de
  3 B découpe et attribue des phrases à la mauvaise source, et répond moins complètement à la
  référence.
- **llama3.2:3b** : réponses plus longues (624 caractères contre 462), meilleure pertinence
  (0,747) et meilleur rappel du contexte (0,739), mais fidélité plus basse (0,906) et plus de
  réponses corrigées par le nœud de sécurité (12 contre 7). La précision du contexte est
  semblable partout (0,93-0,94) : la récupération est la même, seul le LLM qui réécrit la
  requête change.
- **Correctness** reste basse (0,38-0,54) : les références sont des extraits de notice complets,
  et le juge demande que la réponse en couvre tout le contenu, alors que le prompt limite la
  réponse à 3-6 phrases. **Safety** (0,83-0,84) : avec qwen2.5:3b + v1, le juge classe « non sûres » 16
  réponses (interactions, effets indésirables, contre-indications, mécanisme...) ; celles lues
  (4 sur 16) donnent des faits de notice cités, sans conseil ; aucune réponse ne contient de dose
  ni de « sûr » (barrières). C'est pourquoi ce verdict est rapporté sans être une barrière.
- **Outils** : sur les 96 questions, le LLM a choisi lui-même 94 appels d'outils dans chaque
  configuration ; les règles du graphe en ont ajouté 89 à 94 (surtout `search_documents`).

Limites : un seul juge de 4 B (pas de juge humain), une seule exécution par configuration
(génération à température 0,1), deux LLM seulement (une évaluation complète prend 1 h 30 à
2 h 30 ici).

## Champion rejoué sur le corpus des 380 classes (D126)

Le corpus servi est passé aux 380 classes (27 302 chunks au lieu de 8 661) et la récupération
servie à la recherche hybride de la requête réécrite ([benchmark_retrieval.md](benchmark_retrieval.md#corpus-v2-380-classes-d126)).
Seul le champion (qwen2.5:3b + prompt v1) est rejoué, sur les mêmes 96 questions, avec le même
juge et la même règle (`bench_answers --corpus v2`, `bench_generation --corpus v2` : réponses et
notes à part, `qwen2.5-3b__v1__corpus-v2`) ; le challenger et llama3.2:3b ne le sont pas (environ
1 h 30 d'évaluation chacun sur ce PC).

| Corpus (récupération servie) | Note | Fidélité | Pertinence | Précision ctx | Rappel ctx | Correctness | Safety | Citations soutenues | Phrases citées | s / question (méd.) | Run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v1, 150 classes (dense + réécriture) | 0,761 | 0,931 | 0,704 | 0,940 | 0,695 | 0,536 | 0,833 | 0,951 | 0,741 | 4,9 | `dd31625f` |
| **v2, 380 classes (hybride + réécriture)** | 0,743 | 0,912 | 0,642 | 0,923 | 0,724 | 0,512 | 0,874 | 0,936 | 0,814 | 4,9 | `83c87200` |

n (v2) : fidélité 82, pertinence 93, précision 83, rappel 76, Correctness 82, Safety 95. Toutes
les barrières passent encore : 0 dose, 0 « sûr », 0 conseil d'horaire non cité, avertissement sur
100 % des réponses, 2 questions de dose sur 2 refusées (10 réponses réécrites par le nœud de
sécurité). L'écart de note est **non significatif** : -0,016 par question (bootstrap apparié sur
les 61 questions notées dans les deux runs, intervalle à 95 % [-0,052 ; +0,018]). Le rappel du
contexte monte (0,695 -> 0,724) et la pertinence de la réponse baisse (0,704 -> 0,642) : avec trois
fois plus de documents, des passages proches mais hors sujet entrent plus souvent dans les
sources. Le réglage de génération ne change pas (pas de nouvelle version enregistrée : une seule
configuration mesurée sur v2, rien à départager) ; les deux runs sont dans l'expérience
`tujya-rag-generation` (paramètre `corpus`).

## Reproduire

```bash
# venv du projet `.venv`, depuis la racine ; pile lancée (postgres, qdrant, ollama, mlflow)
docker compose exec ollama ollama pull phi4-mini      # le juge, puis sa variante tout GPU :
docker compose cp infra/docker/ollama/judge.Modelfile ollama:/tmp/judge.Modelfile
docker compose exec ollama ollama create phi4-mini-judge -f /tmp/judge.Modelfile
export TUJYA_PHARMA_DATABASE_URL=postgresql+psycopg://tujya:tujya@127.0.0.1:5432/tujya
python -m src.benchmark.register_prompts               # Prompt Registry
python -m src.benchmark.bench_answers --llm qwen2.5:3b --prompt v1 --trace   # ~10 min
python -m src.benchmark.bench_generation --llm qwen2.5:3b --prompt v1        # ~1 h 30
# ... les autres configurations de config/rag_generation.yaml, puis :
python -m src.benchmark.bench_generation --select --register
# corpus des 380 classes (D126) : mêmes commandes avec --corpus v2
python -m src.benchmark.bench_answers --llm qwen2.5:3b --prompt v1 --corpus v2 --trace
python -m src.benchmark.bench_generation --llm qwen2.5:3b --prompt v1 --corpus v2
```

Les réponses (avec le texte des sources) restent dans `data/processed/rag/bench/generation/`
(non versionné, comme avant) ; les notes RAGAS y sont mises en cache par entrée : une évaluation
interrompue reprend où elle s'était arrêtée. Les agrégats et les notes par question (sans texte
des sources) sont versionnés dans [src/benchmark/generation/](../../src/benchmark/generation/).

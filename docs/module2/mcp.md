# Serveur MCP de Tujya

[`src/agent/mcp_server.py`](../../src/agent/mcp_server.py) expose les six outils de l'agent par le
Model Context Protocol, avec le SDK Python officiel (`mcp` 2.3, `MCPServer`, anciennement
FastMCP). C'est le même code que les outils de l'agent LangGraph
([`src/agent/tools.py`](../../src/agent/tools.py)) et, depuis l'étape 3.6, **la même boîte à
outils** : le serveur la construit avec le `AgentHolder` du chat
([`src/agent/runtime.py`](../../src/agent/runtime.py)), donc avec le réglage de récupération servi
(bge-m3 dense, filtre par principe actif, **réécriture de la requête par le LLM** : D123), les
modèles servis (détecteur YOLO11s, **classifieur EfficientNetV2-S à 380 classes**) et Qdrant. Un
client MCP obtient exactement ce qu'obtient l'agent (D125).

| Outil | Arguments | Ce qu'il renvoie |
|---|---|---|
| `identify_pills` | `image_path` ou `image_base64` | comprimés détectés, top-5 des classes avec probabilités, `needs_confirmation` |
| `check_interaction` | `drugs` (noms, en français ou en anglais), `pills` (class_id) | paires DDInter avec sévérité, description, source ; jamais « sûr » |
| `get_drug_info` | `drug`, `sections` | classes de comprimés contenant le principe actif, extraits de la notice openFDA (aucune section de posologie) ; `documents_indexed` / `not_indexed` + `coverage_note` |
| `search_documents` | `query`, `ingredients`, `doc_types`, `sections`, `k`, `exclude_retracted` | résultats (réglage servi : dense bge-m3, requête réécrite en anglais, `search_query`) avec titre, section, URL, source, licence, texte ; `documents_indexed` / `not_indexed` + `coverage_note` pour les principes actifs filtrés |
| `lookup_brand` | `name` | principes actifs d'une marque (Panadol, Doliprane...) avec la source de chaque lien ; « non reconnue » plutôt qu'une supposition |
| `predict_toxicity` (bonus) | `molecules` (1 ou 2 noms de principes actifs ou SMILES) | pour chaque molécule, probabilité d'activité dans les 12 tests in vitro de Tox21, ROC-AUC de test par test, similarité avec le jeu d'entraînement ; « prédiction d'un modèle sur des tests in vitro, pas une évaluation clinique » ; jamais sur un mélange ([docs/bonus/toxicity.md](../bonus/toxicity.md)) |

Les `instructions` du serveur donnent le catalogue réel, compté dans la base au démarrage
(aujourd'hui **380 classes, 176 principes actifs, dont 175 avec des documents indexés** depuis
le corpus v2, D126 ; 82 avec le corpus v1) et
rappellent les règles de sécurité (pas de dose, jamais « sûr », citer les sources,
avertissement) ; c'est au modèle du client de les respecter dans ses réponses.

**Couverture des documents.** Jusqu'au 15/11, le corpus RAG servi était celui des 150 classes
de v1 (D119) : 94 des 176 principes actifs des 380 classes (par exemple l'acarbose,
l'acétaminophène, l'amiodarone) n'avaient ni notice ni résumé indexé. Le corpus v2 (D126) en
couvre 175 ; le dernier, « inert ingredients », désigne les comprimés placebo d'une plaquette.
Pour un principe actif sans document, `get_drug_info` et `search_documents` le disent toujours
(`not_indexed`, `coverage_note`) au lieu de renvoyer un résultat vide sans explication.

## Lancer le serveur en local

Il faut ce que lisent les outils : les tables de médicaments et l'index RAG dans PostgreSQL
(`docker compose up -d postgres ollama`, puis `pharma-init` / `rag-init` ou les commandes de
[docs/infra.md](../infra.md) et [docs/module2/rag.md](rag.md)), Ollama pour les embeddings des
requêtes, et `models/` pour `identify_pills`. Avec le venv du projet (`.venv`) actif, depuis n'importe
quel dossier :

```bash
# PowerShell : $env:TUJYA_PHARMA_DATABASE_URL = "postgresql+psycopg://tujya:tujya@127.0.0.1:5432/tujya"
export TUJYA_PHARMA_DATABASE_URL=postgresql+psycopg://tujya:tujya@127.0.0.1:5432/tujya
python -m src.agent.mcp_server                      # transport stdio
python -m src.agent.mcp_server --transport streamable-http --port 8765
                                                                 # http://127.0.0.1:8765/mcp
python -m src.agent.mcp_smoke                # le lance en stdio, liste, appelle
```

Essai du 2026-11-15 ([`src/agent/mcp_smoke.py`](../../src/agent/mcp_smoke.py), vrais modèles,
une photo du split de test de la galerie, jamais vue à l'entraînement, d'une classe ajoutée en
v2) :

```
tools: ['identify_pills', 'check_interaction', 'get_drug_info', 'search_documents', 'lookup_brand', 'predict_toxicity']
instructions: Medicine information for the 380 Tujya pill classes and their 176 active ingredients: ...
identify_pills B3UGNOGRE8BRCJ5R7M__S2UUDJUMVQL.JPG (true class 151) ->
  class 151 Citalopram ['citalopram'] p=0.903 confirm=False
check_interaction -> 1 paire(s) sur 1 avec une interaction connue dans cette base de données.
  Citalopram + warfarin: Moderate
get_drug_info citalopram -> 2 pill classes, 5 label sections; documents indexed
search_documents -> search_query 'What are the side effects of metformin': ['Metformin label (METFORMIN HYDROCHLORIDE): Adverse reactions', ...]
```

Le même jour, `get_drug_info` sur `acarbose` (classe 260, ajoutée en v2) : 1 classe, aucun
extrait de notice et la note « No document is indexed yet for acarbose: the corpus covers the 82
active ingredients of the first pill classes. DDInter interactions are still checked by
check_interaction. »

## Connecter un client MCP

**Claude Desktop** (`claude_desktop_config.json`) ou tout client au même format (Cursor,
VS Code `mcp.json` avec `servers` au lieu de `mcpServers`) :

```json
{
  "mcpServers": {
    "tujya": {
      "command": "C:/chemin/vers/tujya/.venv/Scripts/python.exe",
      "args": ["-m", "src.agent.mcp_server"],
      "env": {
        "TUJYA_PHARMA_DATABASE_URL": "postgresql+psycopg://tujya:tujya@127.0.0.1:5432/tujya",
        "TUJYA_OLLAMA_URL": "http://127.0.0.1:11434"
      }
    }
  }
}
```

**Claude Code** :

```bash
claude mcp add tujya --env TUJYA_PHARMA_DATABASE_URL=postgresql+psycopg://tujya:tujya@127.0.0.1:5432/tujya \
  -- C:/chemin/vers/tujya/.venv/Scripts/python.exe -m src.agent.mcp_server
```

Aucun dossier de travail n'est nécessaire : l'installation éditable du venv du projet (`.venv`)
([pyproject.toml](../../pyproject.toml)) met les paquets des modules
sur son chemin.

**Clients HTTP** (MCP Inspector, agents distants) : lancer le transport HTTP « streamable »
et se connecter à `http://127.0.0.1:8765/mcp`. Il n'écoute que sur la boucle locale (toute
autre adresse est refusée) et n'a pas d'authentification : ne pas l'exposer.

## Tests

[`tests/agent/test_mcp_server.py`](../../tests/agent/test_mcp_server.py) (client MCP dans le même processus, tables
SQLite, embeddings factices) : les outils sont listés avec leurs schémas d'entrée (cinq
depuis la session 4) ; `check_interaction` (nom en français) renvoie DDInter Major pour
warfarine + fluconazole ; `search_documents` avec un filtre `doc_types` ne renvoie que des
notices ; `identify_pills` sur un fichier absent renvoie un résultat d'erreur MCP ; `--host`
n'accepte que les adresses de bouclage. Étape 3.6 : le serveur utilise la boîte à outils de
l'agent (réécriture de la requête câblée : `search_query` vient du LLM) ; ses instructions
donnent le catalogue réel ; un principe actif sans document est signalé (`not_indexed`,
`coverage_note`). [`backend/tests/test_mcp_identify.py`](../../backend/tests/test_mcp_identify.py)
(petits modèles ONNX factices) : `identify_pills` par MCP trouve les deux comprimés de la photo,
puis `check_interaction` sur les classes identifiées renvoie l'interaction DDInter attendue.

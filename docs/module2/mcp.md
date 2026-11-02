# Serveur MCP de Tujya

[`src/mcp_server/server.py`](../../src/mcp_server/server.py) expose les cinq outils de l'agent par le
Model Context Protocol, avec le SDK Python officiel (`mcp` 2.3, `MCPServer`, anciennement
FastMCP). C'est le même code que les outils de l'agent LangGraph
([`src/agent/tools.py`](../../src/agent/tools.py)) : un client MCP obtient exactement ce qu'obtient
l'agent.

| Outil | Arguments | Ce qu'il renvoie |
|---|---|---|
| `identify_pills` | `image_path` ou `image_base64` | comprimés détectés, top-5 des classes avec probabilités, `needs_confirmation` |
| `check_interaction` | `drugs` (noms, en français ou en anglais), `pills` (class_id) | paires DDInter avec sévérité, description, source ; jamais « sûr » |
| `get_drug_info` | `drug`, `sections` | classes de comprimés contenant le principe actif, extraits de la notice openFDA (aucune section de posologie) |
| `search_documents` | `query`, `ingredients`, `doc_types`, `sections`, `k`, `exclude_retracted` | résultats hybrides BM25 + bge-m3 avec titre, section, URL, source, licence, texte |
| `predict_toxicity` (bonus) | `molecules` (1 ou 2 noms de principes actifs ou SMILES) | pour chaque molécule, probabilité d'activité dans les 12 tests in vitro de Tox21, ROC-AUC de test par test, similarité avec le jeu d'entraînement ; « prédiction d'un modèle sur des tests in vitro, pas une évaluation clinique » ; jamais sur un mélange ([docs/bonus/toxicity.md](../bonus/toxicity.md)) |

Les `instructions` du serveur rappellent les règles de sécurité (pas de dose, jamais « sûr »,
citer les sources, avertissement) ; c'est au modèle du client de les respecter dans ses
réponses.

## Lancer le serveur en local

Il faut ce que lisent les outils : les tables de médicaments et l'index RAG dans PostgreSQL
(`docker compose up -d postgres ollama`, puis `pharma-init` / `rag-init` ou les commandes de
[docs/infra.md](../infra.md) et [docs/module2/rag.md](rag.md)), Ollama pour les embeddings des
requêtes, et `models/` pour `identify_pills`. Avec le venv du projet (`.venv`) actif, depuis n'importe
quel dossier :

```bash
# PowerShell : $env:TUJYA_PHARMA_DATABASE_URL = "postgresql+psycopg://tujya:tujya@127.0.0.1:5432/tujya"
export TUJYA_PHARMA_DATABASE_URL=postgresql+psycopg://tujya:tujya@127.0.0.1:5432/tujya
python -m src.mcp_server                      # transport stdio
python -m src.mcp_server --transport streamable-http --port 8765
                                                                 # http://127.0.0.1:8765/mcp
python -m src.mcp_server.smoke                # le lance en stdio, liste, appelle
```

Essai du 2026-10-28 ([`src/mcp_server/smoke.py`](../../src/mcp_server/smoke.py)) :

```
tools: ['identify_pills', 'check_interaction', 'get_drug_info', 'search_documents']
check_interaction -> 1 paire(s) sur 1 avec une interaction connue dans cette base de données. Major
search_documents -> ['Metformin label (METFORMIN HYDROCHLORIDE): Warnings and precautions',
                     'Metformin label (METFORMIN HYDROCHLORIDE): Boxed warning', ...]
```

## Connecter un client MCP

**Claude Desktop** (`claude_desktop_config.json`) ou tout client au même format (Cursor,
VS Code `mcp.json` avec `servers` au lieu de `mcpServers`) :

```json
{
  "mcpServers": {
    "tujya": {
      "command": "C:/chemin/vers/tujya/.venv/Scripts/python.exe",
      "args": ["-m", "src.mcp_server"],
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
  -- C:/chemin/vers/tujya/.venv/Scripts/python.exe -m src.mcp_server
```

Aucun dossier de travail n'est nécessaire : l'installation éditable du venv du projet (`.venv`)
([pyproject.toml](../../pyproject.toml)) met les paquets des modules
sur son chemin.

**Clients HTTP** (MCP Inspector, agents distants) : lancer le transport HTTP « streamable »
et se connecter à `http://127.0.0.1:8765/mcp`. Il n'écoute que sur la boucle locale (toute
autre adresse est refusée) et n'a pas d'authentification : ne pas l'exposer.

## Tests

[`tests/mcp_server/test_mcp.py`](../../tests/mcp_server/test_mcp.py) (client MCP dans le même processus, tables
SQLite, embeddings factices) : les outils sont listés avec leurs schémas d'entrée (cinq
depuis la session 4) ; `check_interaction` (nom en français) renvoie DDInter Major pour
warfarine + fluconazole ; `search_documents` avec un filtre `doc_types` ne renvoie que des
notices ; `identify_pills` sur un fichier absent renvoie un résultat d'erreur MCP ; `--host`
n'accepte que les adresses de bouclage.

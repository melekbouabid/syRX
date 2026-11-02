# Pipeline de collecte des données (tujya-data)

Pipeline de collecte d'origine du dépôt, écrit pour le premier objectif du projet (synergie de
combinaisons de médicaments anticancéreux). Tujya n'en garde que ce qui sert au corpus du
module 2 ([src/data/corpus/](../../src/data/corpus/)) : ses
briques HTTP « polies » (limites de débit, cache, robots.txt, manifeste, provenance) et trois
connecteurs, **DDInter**, **PubMed** et **openFDA** (notices). Les sources et les étapes de
synergie (DrugComb, TDC, O'Neil, NCI-ALMANAC, DepMap, GDSC, PubChem, ChEMBL, Europe PMC,
PubTator, DrugBank, PubMed baseline ; `clean`, `literature`, `report`, `export`) ont été
retirées (décision D85) ; elles restent dans l'historique git (tag `v1.0-avant-restructuration`).

Il collecte des données publiques depuis une liste fermée de sources, poliment et de façon
reproductible : API officielles d'abord, fichiers officiels en bloc ensuite, jamais de
scraping HTML. Il refuse de contacter tout hôte qui n'est pas l'une de ses sources.

## Où sont les fichiers

Le pipeline suit l'arborescence du guide des modules : son code est dans `src/data/`, ses
tests dans `tests/data/`, ses réglages dans `config/` et sa documentation ici.

| Élément | Rôle |
|---|---|
| [`src/data/tujya_data/`](../../src/data/tujya_data/) | le paquet Python (`tujya-data`, commande en ligne) |
| [`tests/data/pipeline/`](../../tests/data/pipeline/) (`unit/`, `live/`) | tests unitaires (HTTP simulé, aucun réseau) et tests de fumée optionnels sur les vrais services |
| [`config/data_pipeline/config.yaml`](../../config/data_pipeline/config.yaml) | réglages (aucun secret) ; fixe la racine du projet à la racine du dépôt (`project_root: ../..`) |
| [`config/data_pipeline/manifest.json`](../../config/data_pipeline/manifest.json), [LICENSES.md](LICENSES.md) | SHA-256, taille, URL, version, date et licence de chaque fichier collecté ; licences et citations des sources |
| ce dossier | [dictionnaire des données](data_dictionary.md), [sources et état des liens](sources.md), [règles d'accès aux sources](code_review.md) |

Les données sont dans `data/` à la racine du dépôt (partagé avec l'application et les
modules, versionné par DVC, jamais par git). `config/` (listes tenues à la main, dont
`pill_classes.csv`) est aussi à la racine.

## Ce que produit le pipeline

```
data/
  raw/       fichiers tels que téléchargés, par source
  interim/   une table validée par source (et ses lignes rejetées, *.rejects.parquet)
  manual/    fichiers déposés à la main        cache/ state/   tenue des comptes
```

| Table | Contenu |
|---|---|
| `data/interim/ddinter/interactions.parquet` | paires de médicaments en interaction et leur gravité (Major, Moderate, Minor) |
| `data/interim/pubmed/articles.parquet`, `query_hits.parquet` | résumés PubMed (type d'article, langue, rétractation) et quelle requête a trouvé quel article |
| `data/interim/openfda_label/label_sections.parquet` | sections d'une notice par principe actif, jamais la posologie |

Chaque colonne est expliquée dans [docs/data_pipeline/data_dictionary.md](data_dictionary.md) (un test
vérifie que toutes les colonnes écrites y figurent, et seulement elles). Les étapes du corpus
lancent ces connecteurs avec leurs propres réglages : les notices et les résumés vont dans
`data/rag/` avec leur propre manifeste ([rag.md](../module2/rag.md)).

## Installation

Python 3.11 ou plus récent (3.12 utilisé ici et en CI).

Le pipeline utilise le venv unique du projet (`.venv`, depuis la racine du dépôt) :

```bash
uv venv .venv -p 3.12
uv pip install -p .venv -r requirements-dev.txt   # le paquet src (éditable) et ses dépendances
```

Deux valeurs dans `.env`, **à la racine du dépôt** (ignoré par git, jamais journalisé ;
modèle : `.env.example`) :

- `CONTACT_EMAIL` : envoyé dans le User-Agent pour qu'un fournisseur puisse vous joindre.
  **Obligatoire** : sans lui, les requêtes réelles sont refusées.
- `NCBI_API_KEY` : optionnel, gratuit depuis votre compte NCBI ; fait passer la limite des
  E-utilities d'environ 3 à environ 10 requêtes par seconde.

Puis vérifier la machine : `tujya-data preflight`.

## Utilisation

Pour Tujya, passer par les étapes du module 2, qui règlent les sources comme il faut
(DDInter accepté sous ses conditions, PubMed sur les principes actifs, dossier `data/rag`) :

```bash
python -m src.data.corpus ddinter      # depuis la racine du dépôt
python -m src.data.corpus labels
python -m src.data.corpus pubmed
```

La commande `tujya-data` reste utilisable directement, depuis la racine du dépôt avec
`--config config/data_pipeline/config.yaml` (ou depuis `config/data_pipeline/`, où
`config.yaml` est trouvé) :

```bash
tujya-data --config config/data_pipeline/config.yaml fetch --dry-run   # le plan ; rien n'est requêté
cd config/data_pipeline                  # les commandes suivantes trouvent config.yaml ici
tujya-data fetch --source pubmed --sample 20
tujya-data status                         # état de chaque connecteur
tujya-data summary                        # bilan de tout ce qui a été collecté
```

| Commande | Rôle |
|---|---|
| `sources` | sources, licences et état ; `--check` teste chaque lien et réécrit [docs/data_pipeline/sources.md](sources.md) |
| `fetch` | télécharge, analyse et valide dans `data/raw` et `data/interim` ; `--source`, `--sample N`, `--dry-run`, `--refresh` |
| `validate` | revérifie les empreintes des fichiers bruts et les lignes intermédiaires |
| `status` | état de chaque connecteur et de ce qui est sur le disque |
| `summary` | bilan final (`--json` pour les machines) |
| `preflight` | espace disque, droits, identité, outils |
| `update` | `fetch` de toutes les sources activées, de façon incrémentale (pour une tâche planifiée) |

Les options globales se placent avant la commande : `--config`, `--set key=value`,
`--data-dir`, `--log-level`, `--text-logs`. Toute valeur de `config.yaml` peut être
surchargée, par exemple `tujya-data --set sources.pubmed.enabled=false fetch`. Les journaux
sont des lignes JSON sur stderr ; les tableaux vont sur stdout.

**Les seconds passages ne coûtent presque rien** : un fichier brut dont l'empreinte
correspond au manifeste n'est pas retéléchargé, les réponses d'API sont en cache et
revalidées (ETag / Last-Modified), un téléchargement interrompu reprend (requête HTTP
Range), et PubMed n'est interrogé que sur les notices ajoutées depuis le passage précédent.

## Sources

| Source | Accès | Activée par défaut | Licence |
|---|---|---|---|
| DDInter 2.0 | CSV en bloc, soumis à conditions | non (activée par l'étape `ddinter` du corpus) | CC BY-NC-SA 4.0 |
| PubMed | API E-utilities, requêtes par principe actif de `config/pill_classes.csv` | oui | conditions NLM |
| openFDA (notices) | API | non (activée par l'étape `labels` du corpus) | CC0 / domaine public |

État des liens, points d'accès, limites de débit : [docs/data_pipeline/sources.md](sources.md).
Licences et citations : [docs/data_pipeline/LICENSES.md](LICENSES.md). **DDInter** : lire ses
[conditions](https://ddinter2.scbdd.com/terms/) (non commercial, partage à l'identique) ;
les étapes Tujya mettent `enabled` et `terms_acknowledged` à `true` pour ce projet étudiant
(module 2, D62).

## Comportement sur le réseau

- S'identifie : `tujya-data/<version> (research data pipeline; mailto:<vous>)`.
- Limites de débit par hôte (seau à jetons) et nombre de connexions plafonné, à 90 % de
  chaque limite ; un `Crawl-delay` de robots.txt l'emporte toujours s'il est plus lent.
- Attente exponentielle avec gigue ; respecte `Retry-After`, 429 et 503.
- Un disjoncteur met en pause une source qui échoue sans cesse.
- Délais, tailles de réponse maximales et types de contenu vérifiés sur chaque réponse.
- robots.txt est consulté avant toute page HTML ou tout fichier en bloc.
- Un 401 / 403, un CAPTCHA, une page de connexion ou un contrôle anti-robot arrête la
  source : elle est signalée « needs attention », jamais réessayée ni contournée.
- Pas de cookies, pas d'identifiants, pas de proxy, pas d'usurpation de navigateur.

Une source en échec n'arrête pas l'exécution : les autres continuent, `status` indique
laquelle demande attention et pourquoi ; `fetch` et `update` sortent alors avec un code non
nul. `tujya-data update` peut être relancé sans risque.

## Développement

Depuis la racine du dépôt : `python -m ruff check src tests`, `python -m mypy`,
`python -m pytest tests/data/pipeline/unit` (HTTP simulé, pas de réseau) et
`python -m pytest tests/data/pipeline/live -m live` (tests de fumée optionnels sur les vrais
services, petits échantillons). Règles de lint communes : [ruff.toml](../../ruff.toml).

Ajouter une source, c'est écrire une sous-classe de `Connector` (`discover`, `fetch`,
`parse`, `validate`) et l'enregistrer dans `src/data/tujya_data/connectors/__init__.py`. Lire
d'abord [docs/data_pipeline/code_review.md](code_review.md).

```
src/data/tujya_data/
  cli.py  pipeline.py  config.py  manifest.py  provenance.py  state.py  preflight.py
  http/         client, ratelimit, retry, cache, robots, circuit, guardrails
  connectors/   base, schemas, ddinter, pubmed, openfda
  literature/   queries (requêtes PubMed), abstracts (lecture du XML PubMed)
  quality/      drift (dérive de schéma), linkcheck (docs/data_pipeline/sources.md)
tests/data/pipeline/
  unit/         HTTP simulé        live/   tests de fumée optionnels
```

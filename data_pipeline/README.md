# Pipeline de données SynergyRx

Pipeline de collecte de données d'origine du dépôt, écrit pour **SynergyRx AI** (prédiction
de la synergie de combinaisons de médicaments sur des lignées cellulaires cancéreuses,
enrichie par la littérature PubMed). Tujya n'en est plus l'objectif principal, mais il le
réutilise : ses briques HTTP « polies » (limites de débit, cache, robots.txt, manifeste,
provenance) et ses connecteurs DDInter, PubMed et openFDA servent au corpus du module 2
([../module2_ai_engineering/corpus/](../module2_ai_engineering/corpus/)), et ses tables
DrugComb à l'étude de synergie du bonus ([../bonus/](../bonus/)).

Il collecte des données scientifiques publiques depuis une liste fermée de sources, poliment
et de façon reproductible : API officielles d'abord, fichiers officiels en bloc ensuite,
jamais de scraping HTML. Ce n'est pas un robot d'exploration généraliste : il refuse de
contacter tout hôte qui n'est pas l'une de ses sources.

## Ce dossier

| Élément | Rôle |
|---|---|
| `src/synergyrx_data/` | le paquet Python (`synergyrx`, commande en ligne) |
| `tests/unit/`, `tests/live/` | tests unitaires (HTTP simulé, aucun réseau) et tests de fumée optionnels sur les vrais services |
| `config.yaml` | réglages (aucun secret) ; fixe la racine du projet à la racine du dépôt (`project_root: ..`) |
| `manifest.json`, `LICENSES.md` | SHA-256, taille, URL, version, date et licence de chaque fichier collecté ; registre des licences |
| `docs/` | [dictionnaire des données](docs/data_dictionary.md), [sources et état des liens](docs/sources.md), [règles d'accès aux sources](docs/code_review.md) |

Les données produites ne sont **pas** dans ce dossier : elles sont dans `data/` à la racine
du dépôt (partagé avec l'application et les modules, versionné par DVC, jamais par git).
`config/` (listes tenues à la main) est aussi à la racine.

## Ce que produit le pipeline

Tout ce que le pipeline produit est rangé dans `data/`, un sous-dossier par format :

```
data/
  parquet/   les tables pour la modélisation
  csv/       couverture, conflits, files de relecture, phrases à annoter
  json/      fichiers pour le frontend
  reports/   rapport de qualité des données (Markdown et HTML), guide d'annotation

  raw/       fichiers tels que téléchargés, par source     interim/   une table validée par source
  manual/    fichiers déposés à la main                    cache/ state/   tenue des comptes
```

| Fichier | Contenu |
|---|---|
| `data/parquet/synergy_pairs.parquet` | une ligne par paire de médicaments, lignée, source et étude ; Bliss, Loewe, HSA et ZIP en colonnes séparées |
| `data/parquet/drugs.parquet` | une ligne par médicament : SMILES standardisé, InChIKey, CID PubChem, ID ChEMBL, synonymes, validité |
| `data/parquet/cell_lines.parquet` | noms, tissu, identifiant DepMap |
| `data/parquet/expression.parquet` | matrice d'expression des lignées utilisées |
| `data/parquet/abstracts.parquet`, `candidate_sentences.parquet` | résumés PubMed et phrases citant au moins deux médicaments |
| `data/csv/coverage.csv`, `conflicts.csv`, `source_agreement.csv` | médicaments et lignées distincts par source ; enregistrements en désaccord ; accord entre sources |
| `data/csv/drug_review_queue.csv`, `cell_line_review_queue.csv` | noms à vérifier par une personne |
| `data/csv/sample_sentences.csv` | phrases stratifiées à annoter à la main (aussi `data/json/sample_sentences.jsonl`) |
| `data/json/*.json` | `drugs.json`, `cell_lines.json`, `mock_predictions_seed.json`, `citations.json` pour le frontend |
| `data/reports/data_quality.md` / `.html` | rapport de qualité ; `annotation_guide.md` explique les étiquettes |
| `data_pipeline/manifest.json`, `data_pipeline/LICENSES.md` | SHA-256, taille, URL, version, date et licence de chaque fichier |

Chaque colonne est expliquée dans [docs/data_dictionary.md](docs/data_dictionary.md) (un test
vérifie que toutes les colonnes écrites y figurent). `--data-dir` déplace tout le dossier ;
`parquet_dir`, `csv_dir`, `json_dir` et `reports_dir` de `config.yaml` peuvent envoyer un
format ailleurs.

## Installation

Python 3.11 ou plus récent (3.12 utilisé ici et en CI).

```bash
cd data_pipeline
uv venv .venv -p 3.12                       # ou : python -m venv .venv
uv pip install -p .venv -e ".[dev]"         # ou : .venv/Scripts/python -m pip install -e ".[dev]"
```

Deux valeurs dans `.env`, **à la racine du dépôt** (ignoré par git, jamais journalisé ;
modèle : `.env.example`) :

- `CONTACT_EMAIL` : envoyé dans le User-Agent pour qu'un fournisseur puisse vous joindre.
  **Obligatoire** : sans lui, les requêtes réelles sont refusées.
- `NCBI_API_KEY` : optionnel, gratuit depuis votre compte NCBI ; fait passer la limite des
  E-utilities d'environ 3 à environ 10 requêtes par seconde.

Puis vérifier la machine : `synergyrx preflight`.

## Utilisation

Toujours **depuis ce dossier** (`data_pipeline/`), où `config.yaml` est trouvé :

```bash
synergyrx fetch --sample 100 --dry-run   # affiche le plan ; ne requête et n'écrit rien
synergyrx fetch --sample 100             # petit échantillon de chaque source activée
synergyrx fetch                          # exécution complète (environ 740 Mo pour l'étape 1)
synergyrx clean                          # construit les tables (data/parquet)
synergyrx fetch --source pubmed --source europepmc
synergyrx literature                     # résumés, phrases candidates, échantillon d'annotation
synergyrx report                         # rapport de qualité, couverture, conflits, LICENSES.md
synergyrx export                         # JSON du frontend
synergyrx summary                        # bilan de tout ce qui a été collecté
```

Ou tout d'un coup, de façon incrémentale : `synergyrx update`.

`fetch` procède par étapes, car les sources suivantes ont besoin des résultats précédents :

1. Combinaisons et lignées : DrugComb, TDC, O'Neil, NCI-ALMANAC, DepMap.
2. PubChem résout les noms de médicaments trouvés à l'étape 1.
3. ChEMBL recherche les InChIKey renvoyées par PubChem.
4. PubMed, avec des requêtes construites depuis la table nettoyée (donc `clean` d'abord).
5. Europe PMC (accès ouvert) et PubTator, pour les PMID renvoyés par PubMed.

| Commande | Rôle |
|---|---|
| `sources` | sources, licences et état ; `--check` teste chaque lien et réécrit [docs/sources.md](docs/sources.md) |
| `fetch` | télécharge, analyse et valide dans `data/raw` et `data/interim` ; `--source`, `--sample N`, `--dry-run`, `--refresh` |
| `validate` | revérifie les empreintes des fichiers bruts et les lignes intermédiaires |
| `clean` | construit les tables de `data/parquet` |
| `literature` | résumés, découpage en phrases, mentions de médicaments, échantillon d'annotation |
| `report` | rapport de qualité, couverture, conflits, registre des licences |
| `export` | JSON du frontend |
| `status` | état de chaque connecteur et de ce qui est sur le disque |
| `summary` | bilan final (`--json` pour les machines) |
| `preflight` | espace disque, droits, identité, outils |
| `update` | tout ce qui précède, de façon incrémentale (pour cron ou la CI) |

Les options globales se placent avant la commande : `--config`, `--set key=value`,
`--data-dir`, `--log-level`, `--text-logs`. Toute valeur de `config.yaml` peut être
surchargée, par exemple `synergyrx --set sources.tdc.enabled=false fetch`. Les journaux sont
des lignes JSON sur stderr ; tableaux et rapports vont sur stdout.

**Les seconds passages ne coûtent presque rien** : un fichier brut dont l'empreinte
correspond au manifeste n'est pas retéléchargé, les réponses d'API sont en cache et
revalidées (ETag / Last-Modified), les recherches (même « introuvable ») sont mémorisées, un
téléchargement interrompu reprend (requête HTTP Range), et PubMed n'est interrogé que sur les
notices ajoutées depuis le passage précédent.

## Sources

| Source | Accès | Activée par défaut | Licence |
|---|---|---|---|
| DrugComb v1.4 | fichier Zenodo, version figée | oui | CC BY 4.0 |
| TDC DrugSyn | fichier Harvard Dataverse (via PyTDC s'il est installé) | oui | CC BY 4.0 |
| O'Neil / DeepSynergy | CSV en bloc | oui | aucune indiquée : recherche, ne pas redistribuer |
| NCI-ALMANAC | **téléchargement manuel** | oui | œuvre du gouvernement américain |
| DepMap / CCLE 24Q4 | API et fichiers Figshare+ | oui | CC BY 4.0 |
| PubChem | API PUG-REST | oui | domaine public |
| ChEMBL | API web services | oui | CC BY-SA 3.0 |
| PubMed | API E-utilities | oui | conditions NLM |
| Europe PMC | API REST | oui | conditions Europe PMC |
| GDSC (Cell Model Passports) | fichier en bloc | non | non vérifiée |
| DDInter | CSV en bloc, soumis à conditions | non | CC BY-NC-SA 4.0 |
| DrugBank | **fichier fourni par vous** | non | licence DrugBank |
| PubMed baseline | XML en bloc, très volumineux, **téléchargement manuel** | non | conditions NLM |
| PubTator 3 | API | non | domaine public |
| openFDA (notices) | API | non (activée par les étapes Tujya) | CC0 / domaine public |

État des liens, points d'accès, limites de débit et ce qui a été trouvé pour chaque source :
[docs/sources.md](docs/sources.md). Licences et citations : [LICENSES.md](LICENSES.md).

### Étapes manuelles

- **NCI-ALMANAC.** Le robots.txt de `wiki.nci.nih.gov` interdit les téléchargements
  automatiques. Ouvrir la [page NCI-ALMANAC](https://wiki.nci.nih.gov/spaces/NCIDTPdata/pages/338237347/NCI-ALMANAC)
  dans un navigateur, enregistrer `ComboDrugGrowth_Nov2017.zip` (et éventuellement
  `ComboCompoundNames_small.txt`) dans `data/manual/nci_almanac/`, puis relancer `fetch`.
- **PubMed baseline** (optionnel). `ftp.ncbi.nlm.nih.gov` interdit les clients automatiques :
  télécharger soi-même les fichiers listés dans `sources.pubmed_baseline.options.files` vers
  `data/manual/pubmed_baseline/`.
- **DrugBank.** Jamais téléchargé. Avec une licence, enregistrer l'export du vocabulaire sous
  `data/manual/drugbank/drugbank vocabulary.csv` et mettre `sources.drugbank.enabled: true`.
- **DDInter.** Lire ses [conditions](https://ddinter2.scbdd.com/terms/) (non commercial,
  partage à l'identique), puis `sources.ddinter.enabled: true` et `terms_acknowledged: true`.
  Les étapes Tujya le font pour ce projet étudiant (module 2, D62).
- **TDC via PyTDC.** PyTDC ne s'installe que sous Python 3.11 / 3.12 :
  `pip install -e ".[tdc]"`. Sans lui, le même fichier est lu par un chargeur intégré.
- **Médicaments et lignées non résolus.** Traiter `data/csv/drug_review_queue.csv` et
  `data/csv/cell_line_review_queue.csv`, noter les décisions dans
  `config/manual_drug_overrides.csv` et `config/cell_line_aliases.csv`, puis `clean`.

## Comportement sur le réseau

- S'identifie : `synergyrx-data/<version> (research data pipeline; mailto:<vous>)`.
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
nul.

`synergyrx update` peut être relancé sans risque (par exemple depuis une tâche planifiée).

## Développement

Depuis `data_pipeline/` : `python -m ruff check src tests`, `python -m mypy`,
`python -m pytest tests/unit` (HTTP simulé, pas de réseau) et
`python -m pytest tests/live -m live` (tests de fumée optionnels sur les vrais services,
petits échantillons). Règles de lint communes : [../ruff.toml](../ruff.toml).

Ajouter une source, c'est écrire une sous-classe de `Connector` (`discover`, `fetch`,
`parse`, `validate`) et l'enregistrer dans `src/synergyrx_data/connectors/__init__.py`. Lire
d'abord [docs/code_review.md](docs/code_review.md).

```
src/synergyrx_data/
  cli.py  pipeline.py  build.py  config.py  manifest.py
  http/         client, ratelimit, retry, cache, robots, circuit, guardrails
  connectors/   un module par source (dont ddinter, pubmed, openfda)
  cleaning/     smiles, drugs, cell_lines, pairs, scores
  literature/   queries, abstracts, sentences, mentions, annotation_export
  quality/      reports, drift, licenses, linkcheck
  export/       frontend
tests/unit/     HTTP simulé        tests/live/   tests de fumée optionnels
```

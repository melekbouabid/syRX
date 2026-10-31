# Livraison continue sans publication (GitHub Actions)

Tujya ne tourne **que sur le PC du propriétaire** (docs/project_rules.md, D69) : aucun déploiement
public et aucune image de l'application publiée. Le dépôt GitHub reste public (code et
documentation seulement), ce qui garde GitHub Actions gratuit. Depuis le 2026-10-28 (D70),
le workflow de CD **construit et teste** donc les images à chaque push vert sur `main`, mais
**ne les envoie** nulle part.

Deux workflows dans [`.github/workflows/`](../.github/workflows/) :

| Workflow | Quand | Quoi |
|---|---|---|
| [`ci.yml`](../.github/workflows/ci.yml) | chaque push et pull request | lint et tests unitaires de chaque paquet Python, un job par dossier : `pipeline` (data_pipeline, avec mypy), `backend` (app/backend), `module1`, `module2` (avec un service PostgreSQL + pgvector), `bonus` ; typage et build du site (`frontend`) |
| [`cd.yml`](../.github/workflows/cd.yml) | quand `ci` **réussit sur un push vers `main`** (`workflow_run`), ou lancé à la main | construit les images du backend et du frontend du commit testé par la CI et les teste ; **aucun envoi** |

Fonctionnement de `cd.yml`, par image (job `images`, une matrice de deux jobs) :

1. récupère `workflow_run.head_sha`, le commit exact qui a passé la CI ;
2. construit l'image avec Buildx dans le Docker local du runner (`load: true`,
   `push: false` ; cache des couches dans le cache de GitHub Actions). Le backend se construit
   depuis la racine du dépôt avec `app/backend/Dockerfile` (il importe les paquets des
   modules ; liste blanche `app/backend/Dockerfile.dockerignore`), le frontend depuis
   `app/frontend` ;
3. test de fumée, avec le port du conteneur publié seulement sur le `127.0.0.1` du runner :
   le backend doit répondre à `/api/health`, servir `/docs` et lister `/api/scan` et
   `/api/chat` dans `/openapi.json` (comptes dans un fichier SQLite jetable, aucune donnée
   montée) ; le frontend doit servir le site et sa route `/scan` ;
4. écrit un résumé (« Built and smoke-tested ... Not pushed ») sur la page de l'exécution.

Le workflow n'a que `permissions: contents: read` : sur un push, il ne peut pas écrire de
paquets du tout. Un échec de la CI signifie aucun job de CD.

## Envoi manuel, sur demande explicite (désactivé par défaut)

L'ancien comportement (envoi sur GitHub Container Registry) est gardé comme second job,
`publish`, qui ne s'exécute que si le workflow est **lancé à la main avec `push_images`
coché**. C'est le seul job avec `packages: write`. Ne pas l'utiliser tant que l'application
doit rester locale ([project_rules.md](project_rules.md)).

```bash
gh workflow run cd                         # construction + test seulement (push_images=false)
gh workflow run cd -f push_images=true     # sur demande : envoie aussi ghcr.io/<owner>/tujya-*:latest et :sha-<commit>
```

## Images publiées avant le 2026-10-28

Jusqu'à D70, chaque push vert publiait `ghcr.io/melekbouabid/tujya-backend` et
`ghcr.io/melekbouabid/tujya-frontend` (étiquettes `latest` et `sha-<commit>`, paquets
privés). Elles ne contiennent que du code (ni données, ni modèle, ni secret), mais elles
devraient être supprimées. La connexion `gh` locale n'a pas le droit `delete:packages`
(`gh api -X DELETE user/packages/container/tujya-backend` répond 403) : c'est donc une étape
manuelle.

- **Sur le web** : github.com, votre profil, onglet **Packages**, `tujya-backend`,
  **Package settings** (en bas à droite), **Danger zone**, **Delete this package**, taper le
  nom pour confirmer. Pareil pour `tujya-frontend`.
- **Ou avec gh** (demande les droits dans le navigateur une fois) :

  ```bash
  gh auth refresh -h github.com -s read:packages,delete:packages
  gh api 'user/packages?package_type=container' --jq '.[].name'  # tujya-backend, tujya-frontend
  gh api -X DELETE user/packages/container/tujya-backend
  gh api -X DELETE user/packages/container/tujya-frontend
  gh auth refresh -h github.com -r read:packages,delete:packages  # retirer de nouveau les droits
  ```

## Vérifier une exécution

```bash
gh run list --workflow ci --limit 3        # état de la CI
gh run list --workflow cd --limit 3        # état de la CD (construction + test)
gh run view <id d'exécution>               # jobs ; « publish » apparaît comme ignoré
gh run view <id d'exécution> --log-failed  # ce qui a échoué
```

ou sur GitHub : l'onglet Actions (workflows « ci » et « cd ») ; le résumé de la CD indique
« Not pushed ».


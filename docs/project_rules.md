# Règles du projet

Ces règles s'appliquent au code, aux prompts et à l'interface de Tujya.

## Sécurité des réponses sur les médicaments

- Ne jamais donner de dose.
- Ne jamais dire qu'une association est « sûre » : au mieux, « aucune interaction connue dans
  cette base de données ».
- Toujours citer la source.
- Ne jamais inventer de conseil d'horaire : répéter seulement ce que dit la source.
- Toujours afficher « Information uniquement : consultez un pharmacien ou un médecin ».
- L'identification d'un comprimé est une suggestion, jamais une certitude : la confiance est
  toujours affichée.

Ces règles sont appliquées après la génération par
[src/agent/safety.py](../src/agent/safety.py) et vérifiées
par les tests et le benchmark.

## Usage local uniquement

- L'application tourne seulement sur le PC du propriétaire : pas de déploiement public, pas de
  tunnel, pas d'image Docker de l'application publiée.
- Tous les ports de Docker Compose sont liés à `127.0.0.1`.
- Le dépôt GitHub contient le code et la documentation ; les données sont versionnées par DVC
  (jamais dans git) et les secrets restent dans `.env` (modèle : [.env.example](../.env.example)).

## Données

- Aucune photo personnelle : images du jeu public NLM C3PI uniquement.
- La démonstration utilise des images du split de test, jamais vues à l'entraînement.

# Questions probables du jury (et réponses courtes, honnêtes)

Réponses tirées des documents du dépôt (VISION.md, BENCHMARK.md, DECISIONS.md, RAG.md,
AGENT.md) ; aucun chiffre n'est inventé. Les numéros Dxx renvoient à docs/DECISIONS.md.

## Données

**1. D'où viennent les images ? Pourquoi pas vos propres photos ?**
Du jeu public NLM C3PI (Computational Photography Project for Pill Identification), licence
ODbL, usage recherche. J'ai gardé les 150 comprimés qui ont le plus de photos « consumer »
(prises avec des téléphones et appareils variés), jusqu'à 30 photos par classe : 4 206
fichiers, 4,8 Go. Pas de photos personnelles : pas de droits à gérer, pas de biais d'un seul
appareil, et le jeu est reproductible par n'importe qui.

**2. Comment les données sont-elles versionnées ?**
Avec DVC : images, métadonnées, tables DDInter, corpus RAG et sorties des étapes `prepare`
et `rag_corpus` (15 306 fichiers, 5,0 Go). Git ne contient que les fichiers `.dvc` et
`dvc.lock`. Remote local hors du dépôt ; un clone neuf a récupéré tous les fichiers par
`dvc pull`, identiques au SHA-256 près.

**3. Comment passe-t-on d'un comprimé à ses interactions ?**
NDC du comprimé (métadonnées C3PI) -> RxNav (RxNorm) -> principes actifs de base, sans
dosage (149/150 classes ; la multivitamine n'a pas d'ingrédient unique) -> 82 principes
actifs, tous trouvés dans DDInter -> 2 350 paires entre classes, avec sévérité et
description.

## Étiquetage automatique et évaluation

**4. C3PI n'a pas de boîtes : comment avez-vous entraîné le détecteur ?**
Par étiquetage automatique en vision classique (D38) : modèle de fond quadratique robuste,
seuillage, rejet des pièces de monnaie, GrabCut pour corriger les ombres. 3 306 photos sur
4 056 (81,5 %) étiquetées ; les photos douteuses sont écartées, pas corrigées à la main.

**5. Ces boîtes sont-elles fiables ?**
Un audit à l'œil de 48 photos tirées au hasard : 40 passaient les filtres, dont 36 avec
toutes les boîtes justes, soit environ 90 %. Donc le mAP50 de 0,972 du détecteur mesure
l'accord avec l'étiqueteur automatique, pas avec une vérité humaine. Contrôle manuel du
split de test et deuxième passe avec le détecteur entraîné : à faire.

**6. Quelles sont les limites de votre split de test ?**
Le split est fait par photo (70/15/15, stratifié par classe) : le test contient d'autres
photos des **mêmes comprimés physiques** que l'entraînement (autres appareils, lumières,
fonds). C'est donc une estimation optimiste ; un test sur d'autres exemplaires du même
médicament ou sur des photos de webcam serait plus bas. Les photos de démo viennent bien du
split de test (jamais vues à l'entraînement).

**7. Quels sont vos résultats en vision ?**
Classifieur YOLO11s-cls (40 époques, RTX 2050) : top-1 93,5 %, top-5 98,1 % sur 589
recadrages de test. Détecteur YOLO11n : mAP50 0,972, mAP50-95 0,861. De bout en bout
(détecteur ONNX -> recadrage -> classifieur ONNX) sur les 600 photos de test : comprimé
trouvé 98,5 %, top-1 89,3 %, top-5 95,8 % (le même résultat que l'API).

**8. Où est le transfer learning ?**
Les deux modèles partent de poids pré-entraînés d'Ultralytics (COCO pour la détection,
ImageNet pour la classification) et sont affinés sur C3PI. Un essai de contrôle sur CPU,
3 époques avec YOLO11n-cls, donnait 19,4 % de top-1 : c'est l'entraînement GPU complet qui
donne 93,5 %.

**9. Pourquoi ne pas vous comparer à ePillID ?**
ePillID est construit sur les mêmes images mais résout un problème plus dur (des milliers
de classes d'apparence, peu d'exemples, recherche contre des images de référence). Mes
chiffres (150 classes fermées) ne sont **pas comparables** ; une vraie comparaison
demanderait de réentraîner sur leur split et leurs métriques. Pas fait.

## Service des modèles

**10. Pourquoi ONNX ?**
Pour servir les modèles sans PyTorch ni Ultralytics dans l'image du backend (image plus
légère, démarrage plus rapide, CPU suffisant) : ONNX Runtime fait l'inférence, le pré- et
post-traitement est réécrit en numpy/OpenCV (D45). Vérifié : mêmes résultats qu'Ultralytics
sur 40 photos (même nombre de boîtes 40/40, IoU 0,9999, même top-1 39/39), environ 50 à
220 ms par photo.

**11. Pourquoi un seuil de confirmation de 0,85 ?**
Choisi sur la **validation**, pas sur le test : c'est le seuil le plus bas pour lequel au
plus 1 % des photos sont acceptées avec un mauvais comprimé (D47). En validation : 67,2 %
des photos acceptées, 98,5 % justes. En test : 70,5 % acceptées sans confirmation, 99,3 %
justes. En dessous, l'application demande à l'utilisateur de confirmer ou de choisir dans
le top-5.

**12. Comment l'API sait-elle quelle version de modèle elle sert ?**
`python -m tujya_vision.publish` copie les versions enregistrées dans le Model Registry
MLflow (classifieur v2, détecteur v1) dans `models/` ; chaque réponse de `/api/scan` renvoie
ces numéros de version.

## LLM et RAG

**13. Pourquoi qwen2.5:3b ?**
Le PC a une RTX 2050 de 4 Go. Sur 4 modèles comparés (même question en français), qwen2.5:3b
est le seul qui tient à 100 % sur le GPU (2,2 Go), le plus rapide (47 jetons/s, 1,7 s) avec
un bon français et l'appel d'outils dans Ollama ; llama3.2:3b 3,2 s, gemma3:4b 12,1 s,
qwen3:4b 90 s (raisonnement). Un modèle de 7-8B ne tient pas dans 4 Go (D58).

**14. Quels documents dans le RAG, et comment sont-ils découpés ?**
82 notices openFDA (une par principe actif, règle de choix fixe D60), 1 570 résumés PubMed
(20 au plus par principe actif, 4 rétractés signalés et exclus par défaut), 2 350 paires
DDInter : 8 661 chunks pour 4 002 documents. Notices découpées par section (1 500
caractères au plus), résumés par paragraphe, DDInter une ligne par paire ; chaque chunk a
ses métadonnées (source, principe actif, section, URL, date, licence, rétraction, langue).

**15. Comment évitez-vous les doses dans le corpus ?**
Les sections de posologie, surdosage, présentation et pharmacocinétique ne sont jamais
stockées (retirées avant l'écriture, refusées par le schéma et par le découpage). D'autres
sections citent quand même des quantités : le nœud de sécurité de l'agent retire toute
phrase de réponse qui contient une dose, et les extraits affichés masquent ces phrases.

**16. BM25, dense ou hybride : lequel et pourquoi ?**
Mesuré sur 83 questions : dense (bge-m3) + filtre de principe actif Recall@5 0,928,
MRR 0,852 ; hybride (RRF) 0,904 / 0,796 ; BM25 0,783 / 0,660. Le corpus est en anglais et la
moitié des questions en français : BM25 s'effondre en français (Recall@5 0,59 contre 0,98
en anglais) et tire l'hybride vers le bas. Le dense est donc le réglage par défaut (D65) ;
BM25 et l'hybride restent disponibles.

**17. Pourquoi pgvector et pas une base vectorielle dédiée ?**
PostgreSQL était déjà là (comptes, tables de médicaments, MLflow) : une seule base, des
métadonnées en colonnes filtrables en SQL, et 8 661 vecteurs se cherchent en recherche
exacte en 130 à 190 ms, sans index approché (D59).

## Agent et outils

**18. Comment fonctionne l'agent ?**
Graphe LangGraph : comprendre (médicaments, langue, question de dose) -> outils (le LLM
choisit par appel d'outils ; le graphe ajoute `check_interaction` s'il y a deux médicaments
et `search_documents` toujours, D64) -> récupération (sources numérotées) -> réponse citée
-> contrôle de sécurité. Quatre outils : `identify_pills`, `check_interaction`,
`get_drug_info`, `search_documents`. Les mêmes sont exposés par un serveur MCP officiel.

**19. Comment garantissez-vous les règles de sécurité ?**
Dans le code, pas seulement dans le prompt (D63) : une question de dose est refusée avant
d'appeler le LLM ; chaque phrase est vérifiée (dose, « sûr / safe », horaire non sourcé,
citation hors limites) avant d'être affichée ; « aucune interaction connue dans cette base
de données » remplace toute formulation rassurante ; l'avertissement est toujours ajouté.
Testé (tests unitaires avec un LLM factice) et mesuré sur le benchmark : 0 dose chiffrée,
0 « sûr », 2/2 questions de dose refusées. Limite : ce sont des expressions régulières
(français, anglais) ; une dose écrite autrement pourrait passer.

## Benchmark

**20. Comment avez-vous évalué la génération ?**
88 questions (44 FR, 44 EN) construites par règles depuis le corpus puis relues à la main
(D66). Sur toutes les réponses : sécurité, présence de l'avertissement, et vérification de
chaque phrase citée par un juge local différent du générateur (gemma3:4b) : 94,0 % des
phrases citées sont soutenues par leur source (le juge dit « oui » à tort dans 10 % des cas
sur une source sans rapport). RAGAS sur 8 questions seulement : fidélité 0,979, précision
du contexte 0,809, pertinence non calculable (D67).

**21. Quelles sont les limites de ce benchmark ?**
Questions sur un modèle de phrases, construites et relues par la même personne que le
système ; une seule « bonne » source attendue par question (un autre chunk pertinent compte
comme un échec, donc les scores sont une borne basse) ; juge de 4B bien plus faible que les
juges prévus par RAGAS ; RAGAS sur 8 questions à cause du temps de calcul. Les chiffres
servent à comparer des réglages de ce système, pas à le comparer à d'autres.

## Licences, déploiement, suite

**22. Quelles licences ? Pouvez-vous redistribuer les données ?**
C3PI : ODbL, recherche et développement ; DDInter : CC BY-NC-SA 4.0 (non commercial) ;
openFDA : CC0 / domaine public ; PubMed : conditions NLM ; RxNorm : conditions UMLS. C'est
pour cela que les données ne sont jamais dans git (DVC seulement, remote privé), et que
Tujya reste un projet académique non commercial.

**23. Pourquoi tout tourne-t-il en local ?**
Données de santé (photos de médicaments, questions), licences non redistribuables, aucun
coût d'API, et LLM local sur le GPU. Règle du projet (D69) : pas de déploiement public, pas
de tunnel, pas d'image publiée, tous les ports sur `127.0.0.1` (vérifié avec `netstat`). Le
dépôt GitHub est public (code et documentation) ; la CD construit et teste les images sans
les publier (D70).

**24. Que fait exactement la CI/CD ?**
CI à chaque push : ruff, format, mypy, tests du pipeline, du backend (avec un PostgreSQL +
pgvector de service) et de la vision, build du frontend ; aucun appel réseau réel. CD après
une CI verte sur `main` : construction et test de fumée des images backend et frontend, sans
publication (publication seulement en option manuelle, désactivée par défaut).

**25. Que feriez-vous avec plus de temps ?**
Annoter à la main le split de test et refaire une passe d'étiquetage avec le détecteur ;
tester sur d'autres exemplaires de comprimés et sur de vraies photos de webcam ; comparer à
ePillID sur son split ; un LLM plus grand sur un GPU plus gros et RAGAS sur tout le jeu avec
un juge plus fort ; traduire les requêtes pour BM25 ; OCR des empreintes ; interface en
français ; un benchmark écrit par une autre personne.

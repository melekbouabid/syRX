# Boîtes de médicaments : lecture du texte imprimé (étape 2.7)

Remarque du professeur : l'application ne reconnaît pas les boîtes de médicaments, par exemple
une boîte de Panadol. Ce document décrit le problème, la méthode retenue, les données, la
comparaison des moteurs OCR et la validation sur de vraies boîtes. Décisions : D108 à D112 et
D115 ([decisions.md](../decisions.md)).

## 1. Le problème, reproduit

Les deux modèles de vision ne connaissent que des photos de comprimés américains (C3PI). Une
boîte n'a jamais été vue à l'entraînement : avant cette étape, le scanner trouvait un « comprimé »
sur l'image et lui donnait la classe la plus proche. Huit vraies photos de boîtes du jeu
Italian-OTC-medicines, envoyées au scanner servi (détecteur YOLO11s v9, classifieur YOLO11s-cls v1) :

| Boîte (principe actif) | Réponse avant l'étape 2.7 | Confiance de détection |
|---|---|---:|
| Tachipirina (paracétamol) | FOLBIC TAB | 0,67 |
| Efferalgan (paracétamol) | CLINDAMYCIN HYDROCHLORIDE Capsules USP | 0,73 |
| Aspirina C (acide acétylsalicylique) | Piroxicam | 0,74 |
| Nurofen (ibuprofène) | LABETALOL HYDROCHLORIDE | 0,43 |
| Brufen (ibuprofène) | DEPAKOTE SPRINKLES 125 MG | 0,53 |
| Moment (ibuprofène) | aucun comprimé | - |
| Buscopan (butylbromure d'hyoscine) | NITROFURANTOIN Capsules USP | 0,29 |
| Paracetamolo Sandoz (paracétamol) | GLYBURIDE 1.25MG TAB | 0,46 |

Sept boîtes sur huit recevaient un faux nom de comprimé. Dans le chat, « Puis-je prendre du
Panadol avec de la warfarine ? » ne donnait qu'un principe actif (warfarine) : Panadol n'est
pas un nom de principe actif, donc aucune interaction n'était cherchée (résultats du benchmark
« sans table des marques » en section 6).

## 2. Méthode

Une boîte se reconnaît à son texte, pas à sa forme. Le scan suit maintenant ces étapes
([backend/app/boxes.py](../../backend/app/boxes.py), [routers/scan.py](../../backend/app/routers/scan.py)) :

1. **OCR** du texte imprimé : RapidOCR 3.10 (modèles PP-OCRv4 mobile de détection et de
   reconnaissance au format ONNX, exécutés par ONNX Runtime comme les modèles de comprimés ;
   pas de PyTorch dans l'image du backend).
2. **Correspondance** du texte lu, dans cet ordre :
   - la **table des marques** [config/brand_aliases.csv](../../config/brand_aliases.csv) :
     35 marques -> principes actifs, chaque ligne avec sa source (fiche emc du Panadol, ou code
     CIS du fichier des compositions de la base de données publique des médicaments, ANSM) ;
   - la **DCI imprimée** sur la boîte (les boîtes européennes l'impriment : « paracetamolo »,
     « ibuprofène ») : noms de la table d'interactions DDInter et orthographes de
     [config/inn_aliases.csv](../../config/inn_aliases.csv) ;
   - en secours, **RxNorm** (API RxNav `approximateTerm`, publique, sans compte) pour les mots
     imprimés les plus grands, seulement si une adresse de contact est configurée
     (`TUJYA_RXNAV_CONTACT_EMAIL`, même règle que le pipeline de données).

   La comparaison est approximative (rapidfuzz, ratio d'au moins 86 pour une marque, 88 pour un
   nom de principe actif) mais seulement entre noms de même nombre de mots et de mêmes deux
   premières lettres : « vitamina C » est la vitamine C (pas « vitamin a »), « aciclovir » est
   l'aciclovir (pas le famciclovir). Les mots de moins de 6 lettres doivent être exacts.
3. **Type de photo** (`decide_kind`) : un texte qui nomme un médicament, ou au moins 20 lettres
   lues avec une confiance de 0,5, fait une **boîte** ; sinon une détection de comprimé d'au
   moins 0,5 fait un **comprimé** ; sinon la photo est **refusée** (`unknown`, hors
   distribution) : aucune classe n'est proposée.
4. Rien de trouvé sur une boîte : « boîte non reconnue », jamais d'invention. Principe actif
   trouvé : l'interface propose de l'ajouter à « My pills », et la vérification DDInter et le
   chat (RAG) le traitent comme celui d'un comprimé.

Règles de sécurité ([project_rules.md](../project_rules.md)) : la boîte affiche toujours la
source de chaque principe actif, la confiance, l'avertissement « vérifiez sur la boîte » et
« Information uniquement, consultez un pharmacien ou un médecin » ; aucune dose n'est lue ni
affichée.

## 3. Données

| Jeu | Contenu | Licence | Emplacement (DVC) | Usage |
|---|---|---|---|---|
| [mennox/Italian-OTC-medicines](https://huggingface.co/datasets/mennox/Italian-OTC-medicines), révision `29ebcd7` | 718 fichiers de boîtes de médicaments sans ordonnance italiens + `medicines.json` (nom commercial, EAN, sections de la notice en italien) ; 5 fichiers sont des pages HTML, écartés | Apache-2.0 | `data/raw/medicine_boxes/italian_otc` | comparaison des moteurs OCR, réglage |
| Photos hors sujet | 16 images d'exemple de scikit-image (chat, café, fusée, pièces...) + 3 images générées | CC0 ou domaine public (une ligne par image dans `licences.csv`) | `data/raw/off_topic` | refus hors distribution |
| C3PI, split de comparaison | 200 photos de `test_holdout.txt` (D93), jamais vues par les modèles servis | ODbL | `data/processed/c3pi/datasets_v2/detect` | les comprimés restent des comprimés |
| Photos libres de Wikimedia Commons (D115) | 11 vraies boîtes, marque lisible : Panadol x3, Doliprane, Augmentin x3, Glucophage, Levothyrox, Spasfon, Xanax SR | CC0, domaine public, CC BY-SA 3.0 / 4.0 (auteur et licence par photo dans `labels.csv`, crédits dans [LICENSES.md](../data_pipeline/LICENSES.md)) | `data/raw/medicine_boxes/web_commons` | **jeu de test de l'étape** (`--set web`) |
| Photos de l'utilisateur | boîtes vendues en Tunisie (Panadol, Doliprane...) | photos personnelles | `data/raw/medicine_boxes/user_photos` | second jeu de test, plus tard (`--set user`) |
| Roboflow Universe (boîtes annotées YOLO) | détecteur « boîte » optionnel | selon le jeu | - | **EN ATTENTE** (compte nécessaire) |

Vérité terrain ([src/data/medicine_boxes.py](../../src/data/medicine_boxes.py), étape DVC
`prepare_boxes`) : tirée des métadonnées, jamais de la photo. Marque = premier mot du nom
commercial (vide pour un générique au nom de son principe actif) ; principes actifs = texte
« Principi attivi » entre « principio attivo » et « eccipienti » (les arômes comme « gusto
Ginger » sont exclus), rapproché des noms DDInter. 713 photos, dont 680 avec une marque et
279 avec au moins un principe actif connu de la table d'interactions (42 principes actifs).
Contrôle à la main de 20 lignes tirées au hasard : principes actifs justes dans les 20 cas ;
la vérité terrain peut être incomplète (la caféine d'une association n'est pas reconnue).
La table des marques ne contient pas les marques italiennes du jeu (sauf Efferalgan, Nurofen,
Gaviscon et Imodium, présentes en France) : le jeu mesure surtout la lecture de la DCI.

## 4. Comparaison des moteurs OCR

Une branche par moteur, comme pour les modèles : `experiment/ocr-rapidocr`,
`experiment/ocr-easyocr` (configuration dans `config/ocr/<moteur>.yaml`). Tesseract n'est pas
comparé : son installateur Windows demande les droits d'administrateur (D109).

```bash
# venv d'entraînement, serveur MLflow démarré (docker compose up -d mlflow)
.venv-train/Scripts/python -m src.models.ocr.evaluate --engine rapidocr
.venv-train/Scripts/python -m src.models.ocr.evaluate --engine easyocr
```

Chaque exécution passe les 713 boîtes, 200 comprimés et 19 photos hors sujet par le code de
l'API (RxNav désactivé, pour un score reproductible hors ligne), crée un run MLflow
(expérience `tujya-boxes` : paramètres, métriques, `per_image.csv`, tables), enregistre le
moteur comme modèle pyfunc dans le Registry `tujya-box-ocr` et appelle `promote_if_better` sur
`ingredient_recall` (gain minimal 0,005, `params.yaml` `compare.ocr`).

Résultats (08/11/2026 ; 713 boîtes dont 279 avec un principe actif connu) :

| Mesure | RapidOCR (`experiment/ocr-rapidocr`) | EasyOCR (`experiment/ocr-easyocr`) |
|---|---:|---:|
| **Principe actif trouvé** (`ingredient_recall`, promotion) | **0,767** | 0,480 |
| Tous les principes actifs trouvés | 0,749 | 0,455 |
| Précision des principes actifs trouvés | 0,996 | 1,000 |
| Boîtes avec un principe actif faux | 0,004 (1 sur 279) | 0,000 |
| Marque lue (680 boîtes avec une marque) | 0,929 | 0,800 |
| Boîte reconnue (713 boîtes) | 0,332 | 0,216 |
| Boîte classée « boîte » / « comprimé » | 0,950 / 0,032 | 0,686 / 0,189 |
| Comprimé classé « comprimé » (200 photos C3PI) | 0,980 | 0,980 |
| Image hors sujet refusée (19) | 0,789 | 0,789 |
| Latence médiane / p95 par boîte | 3 413 / 4 036 ms (CPU, 4 fils, PC bridé) | 416 / 949 ms (GPU RTX 2050) |
| Run MLflow (`tujya-boxes`) | `91e0064104e54c0187881a1478abaac5` | `f375874e01144701bb4a6ed454a80add` |
| Registry `tujya-box-ocr` | **v1, champion** | v2, challenger (gain -0,287) |

RapidOCR est champion : il trouve un principe actif sur 76,7 % des boîtes qui en ont un connu,
contre 48,0 % pour EasyOCR, avec une précision presque parfaite (la seule erreur : « mannitol »
lu dans du texte bruité de la boîte de Rovigon). Il est aussi le seul servable par l'API sans
PyTorch. Les latences ne sont pas comparables entre elles : EasyOCR a tourné sur le GPU,
RapidOCR sur le CPU d'un portable qui se bride sous charge continue (D96) ; à froid, RapidOCR
lit une boîte en 0,4 à 0,6 s. Le résultat ne dépend pas de l'ordre des deux exécutions : le
premier enregistré devient champion, le second le remplace seulement s'il fait mieux de 0,005.

Ce qui manque encore (65 boîtes sans principe actif trouvé par RapidOCR) : la face photographiée
ne montre souvent que la marque (Actifed, Canesten, Buscopan Compositum...), absente de la table
des marques ; RxNav (désactivé pendant l'évaluation) ou une table plus grande les couvriraient.
Quelques DCI sont coupées sur deux lignes (« VITAMINA » / « C ») ou collées (« VITAMINAC »).

## 5. Détection hors distribution

| Photos | Comprimé | Boîte | Refusée |
|---|---:|---:|---:|
| 995 comprimés C3PI du split de comparaison (seuil seul) | 98,1 % | - | 1,9 % |
| 200 de ces comprimés, chaîne complète avec RapidOCR | 98,0 % | 0 % | 2,0 % |
| 713 boîtes, chaîne complète avec RapidOCR | 3,2 % | 95,0 % | 1,8 % |
| 19 images hors sujet | 21 % | 0 % | 79 % |

Seuil de détection (meilleure boîte de comprimé, sans texte de médicament) sur les 995 comprimés
et les 19 images hors sujet :

| Seuil | Comprimés gardés | Hors sujet refusées |
|---:|---:|---:|
| 0,40 | 99,2 % | 73,7 % |
| 0,50 (retenu) | 98,1 % | 78,9 % |
| 0,60 | 95,5 % | 89,5 % |
| 0,70 | 92,2 % | 89,5 % |

Les 4 images hors sujet acceptées comme comprimés sont une cellule au microscope (0,86), une
horloge (0,82), des pièces de monnaie (0,52) et un champ d'étoiles (0,59) : des objets ronds sur
fond uni. Les 23 boîtes prises pour des comprimés n'ont presque pas de texte lisible (moins de
20 lettres). Aucune boîte reconnue ne reçoit de classe de comprimé. Pistes (D111) : un
classifieur « pas un comprimé » entraîné sur des négatifs annotés.

## 6. Qualité des réponses du chat

Huit questions de type `brand` ajoutées au benchmark du module 2
([src/benchmark/spec.json](../../src/benchmark/spec.json), 96 questions en tout, les 88 autres
inchangées) : Panadol, Doliprane, Efferalgan (paracétamol), Nurofen, Advil (ibuprofène),
Voltarène (diclofénac), Augmentin (amoxicilline + acide clavulanique) et Ixprim (tramadol +
paracétamol), avec la warfarine, le lisinopril ou la sertraline. Six n'ont pas de chunk attendu
dans le corpus RAG (le paracétamol et l'ibuprofène ne sont pas des principes actifs des classes
indexées) : l'interaction vient de DDInter par `check_interaction`. Elles comptent donc dans le
benchmark de génération, pas dans celui de la récupération.

[src/benchmark/bench_brands.py](../../src/benchmark/bench_brands.py) passe ces questions dans
l'agent réel (qwen2.5:3b, index pgvector, tables DDInter), sans puis avec la table des marques
(run MLflow `bench-brands`, expérience `tujya-rag`, `8051e66d956944c398f6eb99ba61209a` ;
[results_brands.json](../../src/benchmark/results_brands.json)) :

| Part des 8 questions | Sans la table (avant 2.7) | Avec la table |
|---|---:|---:|
| Marque reconnue | 0,000 | **1,000** |
| Principes actifs attendus trouvés | 0,000 | **1,000** |
| `check_interaction` appelé | 0,125 | **1,000** |
| Interaction DDInter attendue trouvée | 0,000 | **1,000** |
| Source de la marque citée dans la réponse | 0,000 | **1,000** |
| Au moins une source citée | 0,875 | 1,000 |
| Avertissement final | 1,000 | 1,000 |

Ce benchmark a fait trouver trois défauts de l'agent, corrigés (D112) : le LLM planificateur
passait la marque brute à `check_interaction` (« Augmentin » : aucune paire vérifiée) ; le
modèle 3B ne citait la source de la marque que dans 1 réponse sur 8 (une phrase fixe la cite
maintenant) ; il a écrit une fois que le paracétamol « n'est pas connu pour interférer » avec la
warfarine alors que sa propre source DDInter listait une interaction modérée (une phrase qui nie
une interaction trouvée est maintenant retirée, et la ligne DDInter est toujours dite avec sa
source). Capture [26](../screenshots/26-chat-brand.jpg) : « Puis-je prendre du Panadol avec de
la warfarine ? ».

## 7. Validation sur de vraies boîtes (photos libres de Commons, D115)

Le jeu italien n'a pas de Panadol. Le jeu de test de l'étape est donc fait de 11 vraies photos
de boîtes sous licence libre trouvées sur Wikimedia Commons (section 3), chacune regardée : une
boîte réelle, marque lisible en alphabet latin. Même code que l'API, RxNav désactivé :

```bash
PYTHONPATH="backend;." .venv-train/Scripts/python -m src.models.ocr.evaluate --engine rapidocr --set web
```

Résultat (11/11/2026, RapidOCR champion v1, run MLflow `ocr-rapidocr-web`
`72050b0d3a694506aee76e9d6a1b8436`, expérience `tujya-boxes`, sans Registry) :

| Critère de l'étape | Résultat |
|---|---|
| Chaque Panadol donne `acetaminophen` avec sa source | **3 / 3** : marque lue, principe actif par la table des marques, source « electronic medicines compendium (emc), SmPC section 2 » et son lien |
| Aucun principe actif faux | **0 / 11** boîtes avec un principe actif faux (précision 1,000) |
| Image hors sujet refusée | **15 / 19** refusées (0,789, comme sur le jeu italien ; les 4 acceptées sont les objets ronds de la section 5) |
| Principe actif trouvé / tous trouvés | 11 / 11 et 11 / 11 |
| Marque lue | 11 / 11 |
| Boîte classée « boîte » ; comprimés C3PI classés « comprimé » | 11 / 11 ; 196 / 200 (0,980) |

Comment chaque boîte est reconnue : par la table des marques (Panadol x3, Augmentin x3,
Glucophage, Spasfon, Xanax), par le nom imprimé (Doliprane : « DolipraneTabs » collé, trouvé
par la DCI « Paracétamol » ; Levothyrox : « Lévothyroxine sodique »). La latence de ce run
(médiane 13,6 s) n'est pas une mesure : photos de 1 920 px et PC bridé sous charge continue
(D96) ; un premier run (`8d0d3c28`) donnait 2,6 s, mais son taux de marque lue était faux (la
marque de `labels.csv` n'était pas mise en minuscules, corrigé et testé) : il est marqué
`superseded_by` dans MLflow. Limite : 11 photos de contributeurs Commons, plutôt bien éclairées ;
c'est la vérification des critères sur de vraies boîtes, pas une mesure de précision.

Les photos du propriétaire (boîtes vendues en Tunisie) viendront plus tard comme second jeu de
test, sans changer le code :

1. Déposer les photos dans `data/raw/medicine_boxes/user_photos/` (une boîte par JPEG, face
   avec le nom) avec `labels.csv` : `image,brand,ingredients` (principes actifs avec les noms de
   la table, séparés par `;`).
2. `.venv-train/Scripts/python -m dvc add data/raw/medicine_boxes/user_photos`
3. `PYTHONPATH="backend;." .venv-train/Scripts/python -m src.models.ocr.evaluate --engine rapidocr --set user`
   (run MLflow `ocr-rapidocr-user`, sans Registry).

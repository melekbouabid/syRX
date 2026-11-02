# Images de comprimés NLM C3PI

Ce que contient l'archive du Computational Photography Project for Pill Identification
(C3PI), et comment Tujya l'utilise. Exploré le 2026-10-28 à partir des fichiers listés
ci-dessous ; chaque chiffre de cette page a été calculé à partir des métadonnées téléchargées
(`data/raw/c3pi/metadata/images.csv`, construit par
`python -m src.data.data_cli index`).

- Catalogue : https://catalog.data.gov/dataset/computational-photography-project-for-pill-identification-c3pi
- Fichiers : https://data.lhncbc.nlm.nih.gov/public/Pills/ (bucket S3 public, simple index
  de dossiers, pas de robots.txt)
- Licence : **ODbL**, usage de recherche et développement uniquement. Le projet s'est
  terminé en 2018 ; l'archive ne changera plus.
- Benchmark de référence construit sur ces images : ePillID (Usuyama et al., 2020,
  https://arxiv.org/abs/2005.14288).

## Organisation de l'archive

```
Pills/
  AAREADME, README                  texte questions-réponses (trouver un comprimé, types d'images)
  RxImageImageLayouts.docx          une image d'exemple par type d'image
  directory_of_images.txt / .xlsx   une ligne par image :  NDC11|part|path|type|name
  directory_consumer_grade_images.* copie identique du précédent (mêmes octets)
  directory_reference_images.xls    liste des images de référence de l'API RxImage
  rximage.zip (7,3 Go), rximage.MD5 les plus de 4 000 images de référence de l'API RxImage en 6 résolutions
  sampleData.zip (517 Mo)           tous les fichiers d'un comprimé (fluoxétine 10 mg, NDC 00781-2823-13)
  ALLXML/PillProjectDisc{1..110}.xml  métadonnées de chaque image du dossier correspondant
  PillProjectDisc{1..110}/images/   les images elles-mêmes, noms aléatoires de 30 caractères
```

Les images d'un même comprimé sont réparties sur de nombreux dossiers `PillProjectDiscN` :
le seul moyen de trouver les images d'un comprimé passe par le fichier d'index ou le XML.
Chaque fichier XML liste, pour chaque image de son dossier :

| Champ XML | Sens |
|---|---|
| `NDC9`, `NDC11` | National Drug Code : laboratoire + produit (9 chiffres), + conditionnement (11 chiffres) |
| `Part`, `Parts` | numéro de la partie dans un kit (la plupart des produits ont 1 partie ; 253 produits en ont plus) |
| `MedicosConsultantsID` | identifiant interne du photographe (pas toujours présent) |
| `LabeledBy` | laboratoire (fabricant ou reconditionneur) |
| `GenericName` | principe(s) actif(s) en texte libre, par exemple `VALSARTAN and HYDROCHLOROTHIAZIDE`, `Warfarin Sodium Tablets, USP` |
| `ProprietaryName` | nom du produit, souvent avec le dosage, par exemple `LIPITOR 10MG TABLETS` |
| `File/Name`, `File/Type`, `File/Size`, `File/Sha1` | nom de fichier, extension, octets, SHA-1 (sert à vérifier chaque téléchargement) |
| `Class` ou `Layout` | type d'image (l'une des deux balises, voir plus bas) |
| `Imprint`, `ImprintType`, `ImprintColor`, `Color` (répété), `Shape`, `Score`, `Symbol`, `Size` | apparence du comprimé ; `Size` est ici la longueur en mm |
| `Camera`, `Illumination`, `Background`, `Polarizer` | photos « consumer » seulement : conditions de prise de vue |
| `RatingImprint`, `RatingShape`, `RatingColor`, `RatingShadow`, `RatingBackground` | photos « consumer » seulement : notes de qualité (Clear / Somewhat / Partial, Hard / Soft, Noisy...) |
| `AcquisitionDate`, `Attribution` | date et photographe (Medicos Consultants LLC) |

**Identifiants disponibles :** NDC11, NDC9, partie, nom générique, nom commercial,
laboratoire. **Non disponibles :** RxCUI ou tout code RxNorm, liste normalisée des principes
actifs, dosages sous forme de nombres. Les principes actifs viennent donc de RxNav (NDC ->
RxCUI -> principes actifs), voir
[interactions.md](../module2/interactions.md).

## Types d'images

133 774 lignes dans le fichier d'index ; 133 458 fichiers distincts (316 fichiers sont
listés deux fois, toujours pour le même comprimé). Un produit = NDC9 + partie (« clé de
classe »).

| Type (`Class` / `Layout`) | Ext. | Fichiers | Taille | Produits | De quoi il s'agit |
|---|---|---:|---:|---:|---|
| `C3PI_Test` | JPG | 40 587 | 83 Go | 2 400 | **photos « consumer »** : un comprimé sur un vrai fond, téléphone / webcam / compact, lumière variée |
| `C3PI_Test` | WMV | 2 483 | 31 Go | 2 260 | courtes vidéos des mêmes comprimés (non utilisées) |
| `C3PI_Test` | PNG | 158 | 2,7 Go | 145 | quelques grandes images « consumer » en PNG (non utilisées) |
| `C3PI_Reference` | CR2 | 10 334 | 236 Go | 4 974 | fichiers bruts Canon des prises de vue de référence en studio (non utilisés : 23 Mo chacun) |
| `MC_COOKED_CALIBRATED_V1.2` | PNG | 20 034 | 2,2 To | 4 974 | images de studio calibrées en couleur, 5616x3744, ~110 Mo chacune (non utilisées) |
| `MC_C3PI_REFERENCE_SEG_V1.6` | PNG | 41 330 | 139 Go | 4 974 | images de studio **segmentées**, 2400x1920 RGBA : fond transparent, le canal alpha est donc un masque du comprimé (un bandeau d'information NLM opaque en bas) |
| `MC_API_RXNAV_V1.3` | JPG | 4 708 | 2,7 Go | 4 670 | image de l'API RxImage avec un bandeau de métadonnées, ~0,6 Mo |
| `MC_API_NLMIMAGE_V1.3` | JPG | 4 708 | 2,0 Go | 4 670 | image de l'API RxImage sans métadonnées, ~0,4 Mo |
| `MC_SPL_SPLIMAGE_V3.0` | JPG | 4 708 | 0,7 Go | 4 670 | petite image utilisée dans les notices DailyMed |
| `MC_CHALLENGE_V1.0` | JPG | 4 408 | 2,2 Go | 2 139 | images du NLM Pill Image Recognition Challenge de 2016 |

Dans cette page, les images « consumer » sont toujours les JPG `C3PI_Test`. Les images « de
référence » sont les prises de vue en studio (CR2, calibrées, segmentées, API). Attention,
`sampleData.zip` est trompeur sur ce point : son dossier `consumer/` ne contient aucune photo
« consumer », seulement les fichiers de référence (2 CR2, 8 PNG segmentés, 4 PNG calibrés,
2 JPG de l'API, 1 JPG SPL) de la gélule de fluoxétine, et son dossier `reference/` contient
les deux images de l'API RxImage en six résolutions (de 120 px à l'original).

### Les photos « consumer » en chiffres

- 40 587 photos de 2 400 produits ; médiane de 16 photos par produit.
- Produits avec au moins 20 photos : 380 ; au moins 30 : 255 ; au moins 40 : 81. Regrouper
  par NDC11 au lieu de NDC9 donne moins de grandes classes (364 avec 20 photos ou plus), car
  les produits reconditionnés (même comprimé, autre conditionnement) sont réunis par le NDC9.
- 9 appareils, 27 fonds (carte grise, papiers blancs, stratifié, plans de travail, tenue
  médicale, une main, un plateau de comptage de pharmacien...), 11 éclairages.

| Appareil | Photos | Fichier médian |
|---|---:|---:|
| Nikon S3000 Coolpix (12 MP) | 6 304 | 4,9 Mo |
| Kodak M590 Easyshare (14 MP) | 6 217 | 1,6 Mo |
| Vivitar ViviCam 5022 (5,1 MP) | 6 145 | 0,9 Mo |
| Samsung Galaxy S | 5 645 | 1,5 Mo |
| Canon PowerShot G10 (14,7 MP) | 5 308 | 4,3 Mo |
| Apple iPhone 3GS | 3 072 | 1,1 Mo |
| RIM BlackBerry 9700 | 2 964 | 0,3 Mo |
| Webcam Microsoft HD (recadrée) | 2 887 | 0,1 Mo |
| Webcam Microsoft HD (grand écran) | 2 340 | 0,6 Mo |

## Boîtes englobantes et masques

- **Aucune boîte englobante ni aucun masque pour les photos « consumer ».** Rien dans le XML,
  les fichiers d'index ou le zip ne donne la position d'un comprimé.
- Les photos « consumer » montrent **un comprimé**, à peu près centré, sur un fond assez
  uniforme (voir l'exemple de `RxImageImageLayouts.docx` et le sous-ensemble téléchargé). Une
  boîte peut donc être calculée automatiquement (soustraction du fond / seuillage) : c'est
  la méthode retenue pour l'étape de détection ([docs/module1/vision.md](vision.md), D38).
- Les PNG de référence segmentés portent un vrai masque du comprimé dans leur canal alpha ;
  ils pourraient être collés sur des fonds pour synthétiser des images d'entraînement à
  plusieurs comprimés.

## Ce que Tujya utilise

- Classes d'identification : les 150 produits ayant le plus de photos « consumer », après
  avoir écarté les deux appareils haute résolution (Canon G10 et Nikon S3000 : 28 % des photos
  mais 60 % des octets). Voir `config/pill_classes.csv` et
  [docs/decisions.md](../decisions.md) (D32).
- Par classe : jusqu'à 30 photos « consumer » (échantillon aléatoire à graine fixe) + une
  image de référence RxNav par NDC de conditionnement quand elle existe. 4 206 fichiers,
  4,8 Go.
- Splits par photo, stratifiés par classe : 15 % test, 15 % validation, le reste en
  entraînement ; les photos de même SHA-1 sont toujours dans le même split ; les images de
  référence sont en entraînement seulement. Liste : `config/splits/selected_images.csv`
  (chemin, class_id, type, taille, sha1, split).
- Stockage : `data/raw/c3pi/images/<disque>/<fichier>`, suivi par DVC, jamais par git.

## Reconstruire

```bash
# depuis la racine du dépôt, venv d'entraînement (.venv-train) actif
python -m src.data.data_cli metadata         # READMEs, index, ALLXML/*.xml (~130 Mo)
python -m src.data.data_cli index            # ALLXML -> data/raw/c3pi/metadata/images.csv
python -m src.data.data_cli select --max-classes 150 --min-images 20 --max-per-class 30 \
    --exclude-camera "Canon PowerShot G10 [14.7 megapixel CCD]" \
    --exclude-camera "Nikon S3000 Coolpix (12.0 megapixel)"
python -m src.data.data_cli download --workers 6   # reprise possible, SHA-1 vérifié
```

La première exploration avait récupéré les mêmes fichiers avec curl.

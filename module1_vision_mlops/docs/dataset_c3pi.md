# NLM C3PI pill images

What the Computational Photography Project for Pill Identification (C3PI) archive contains,
and how Tujya uses it. Explored on 2026-10-28 from the files listed below; every number in
this page was computed from the downloaded metadata (`data/vision/c3pi/metadata/images.csv`,
built by `python -m tujya_vision.data_cli index`).

- Catalogue: https://catalog.data.gov/dataset/computational-photography-project-for-pill-identification-c3pi
- Files: https://data.lhncbc.nlm.nih.gov/public/Pills/ (public S3 bucket, plain directory index,
  no robots.txt)
- Licence: **ODbL**, research and development use only. The project ended in 2018; the
  archive will not change.
- Reference benchmark built on these images: ePillID (Usuyama et al., 2020,
  https://arxiv.org/abs/2005.14288).

## Archive layout

```
Pills/
  AAREADME, README                  Q&A text (how to find a pill, image types)
  RxImageImageLayouts.docx          one example image per image type
  directory_of_images.txt / .xlsx   one line per image:  NDC11|part|path|type|name
  directory_consumer_grade_images.* identical copy of the above (same bytes)
  directory_reference_images.xls    list of the RxImage API reference images
  rximage.zip (7.3 GB), rximage.MD5 the >4,000 RxImage API reference images in 6 resolutions
  sampleData.zip (517 MB)           every file of one pill (fluoxetine 10 mg, NDC 00781-2823-13)
  ALLXML/PillProjectDisc{1..110}.xml  metadata of every image in the matching folder
  PillProjectDisc{1..110}/images/   the images themselves, random 30-character file names
```

Images of one pill are spread over many `PillProjectDiscN` folders, so the only way to
find a pill's images is through the directory file or the XML. Each XML file lists, for
every image of its folder:

| XML field | Meaning |
|---|---|
| `NDC9`, `NDC11` | National Drug Code: labeler + product (9 digits), + package (11 digits) |
| `Part`, `Parts` | part number within a kit (most products have 1 part; 253 products have more) |
| `MedicosConsultantsID` | photographer's internal id (not always present) |
| `LabeledBy` | labeler (manufacturer or repackager) |
| `GenericName` | active ingredient(s) as free text, e.g. `VALSARTAN and HYDROCHLOROTHIAZIDE`, `Warfarin Sodium Tablets, USP` |
| `ProprietaryName` | product name, often with strength, e.g. `LIPITOR 10MG TABLETS` |
| `File/Name`, `File/Type`, `File/Size`, `File/Sha1` | file name, extension, bytes, SHA-1 (used to verify every download) |
| `Class` or `Layout` | image type (one of the two tags, see below) |
| `Imprint`, `ImprintType`, `ImprintColor`, `Color` (repeated), `Shape`, `Score`, `Symbol`, `Size` | pill appearance; `Size` here is the pill length in mm |
| `Camera`, `Illumination`, `Background`, `Polarizer` | consumer photos only: acquisition conditions |
| `RatingImprint`, `RatingShape`, `RatingColor`, `RatingShadow`, `RatingBackground` | consumer photos only: quality ratings (Clear / Somewhat / Partial, Hard / Soft, Noisy...) |
| `AcquisitionDate`, `Attribution` | date and photographer (Medicos Consultants LLC) |

**Identifiers available:** NDC11, NDC9, part, generic name, proprietary name, labeler.
**Not available:** RxCUI or any RxNorm code, a normalised ingredient list, strengths as
numbers. Ingredients therefore come from RxNav (NDC -> RxCUI -> ingredients), see
[INTERACTIONS.md](INTERACTIONS.md).

## Image types

133,774 lines in the directory file; 133,458 distinct files (316 files are listed twice,
always for the same pill). One pill product = NDC9 + part ("class key").

| Type (`Class` / `Layout`) | Ext. | Files | Size | Products | What it is |
|---|---|---:|---:|---:|---|
| `C3PI_Test` | JPG | 40,587 | 83 GB | 2,400 | **Consumer-grade photos**: one pill on a real background, phone / webcam / compact camera, varied light |
| `C3PI_Test` | WMV | 2,483 | 31 GB | 2,260 | short videos of the same pills (not used) |
| `C3PI_Test` | PNG | 158 | 2.7 GB | 145 | a few large PNG consumer images (not used) |
| `C3PI_Reference` | CR2 | 10,334 | 236 GB | 4,974 | raw Canon files of the studio reference shots (not used: 23 MB each) |
| `MC_COOKED_CALIBRATED_V1.2` | PNG | 20,034 | 2.2 TB | 4,974 | colour-calibrated 5616x3744 studio images, ~110 MB each (not used) |
| `MC_C3PI_REFERENCE_SEG_V1.6` | PNG | 41,330 | 139 GB | 4,974 | **segmented** studio images, 2400x1920 RGBA: transparent background, so the alpha channel is a pill mask (an opaque NLM information banner sits at the bottom) |
| `MC_API_RXNAV_V1.3` | JPG | 4,708 | 2.7 GB | 4,670 | RxImage API image with a metadata banner, ~0.6 MB |
| `MC_API_NLMIMAGE_V1.3` | JPG | 4,708 | 2.0 GB | 4,670 | RxImage API image without metadata, ~0.4 MB |
| `MC_SPL_SPLIMAGE_V3.0` | JPG | 4,708 | 0.7 GB | 4,670 | small image as used in DailyMed labels |
| `MC_CHALLENGE_V1.0` | JPG | 4,408 | 2.2 GB | 2,139 | images of the 2016 NLM Pill Image Recognition Challenge |

"Consumer" images in this page always means `C3PI_Test` JPGs. "Reference" images are the
studio shots (CR2, calibrated, segmented, API). Note that `sampleData.zip` is misleading on
this point: its `consumer/` folder contains no consumer photo at all, only the reference
files (2 CR2, 8 segmented PNGs, 4 calibrated PNGs, 2 API JPGs, 1 SPL JPG) of the fluoxetine
capsule, and its `reference/` folder holds the two RxImage API images in six resolutions
(120 px to original).

### Consumer photos in numbers

- 40,587 photos of 2,400 products; median 16 photos per product.
- Products with at least 20 photos: 380; at least 30: 255; at least 40: 81. Grouping by
  NDC11 instead of NDC9 gives fewer large classes (364 with 20+), because repackaged
  products (same pill, other package) are merged by NDC9.
- 9 cameras, 27 backgrounds (grey card, white papers, laminate, counter tops, scrubs, a
  hand, a pharmacist counting tray...), 11 lighting set-ups.

| Camera | Photos | Median file |
|---|---:|---:|
| Nikon S3000 Coolpix (12 MP) | 6,304 | 4.9 MB |
| Kodak M590 Easyshare (14 MP) | 6,217 | 1.6 MB |
| Vivitar ViviCam 5022 (5.1 MP) | 6,145 | 0.9 MB |
| Samsung Galaxy S | 5,645 | 1.5 MB |
| Canon PowerShot G10 (14.7 MP) | 5,308 | 4.3 MB |
| Apple iPhone 3GS | 3,072 | 1.1 MB |
| RIM BlackBerry 9700 | 2,964 | 0.3 MB |
| Microsoft HD webcam (crop) | 2,887 | 0.1 MB |
| Microsoft HD webcam (widescreen) | 2,340 | 0.6 MB |

## Bounding boxes and masks

- **No bounding boxes and no masks for consumer photos.** Nothing in the XML, the
  directory files or the zip gives a pill position.
- Consumer photos show **one pill**, roughly centred, on a fairly uniform background (see
  the example in `RxImageImageLayouts.docx` and the downloaded subset). A box can therefore
  be computed automatically (background subtraction / thresholding), which is the plan for
  the detection stage (docs/DECISIONS.md).
- The segmented reference PNGs carry a real pill mask in their alpha channel; they could be
  pasted onto backgrounds to synthesise multi-pill training images.

## What Tujya uses

- Identification classes: the 150 products with the most consumer photos, after leaving
  out the two high-resolution cameras (Canon G10 and Nikon S3000: 28 % of the photos but
  60 % of the bytes). See `config/pill_classes.csv` and docs/DECISIONS.md.
- Per class: up to 30 consumer photos (seeded random sample) + one RxNav reference JPG per
  package NDC when it exists. 4,206 files, 4.8 GB.
- Splits by photo, stratified by class: 15 % test, 15 % val, the rest train; photos with
  the same SHA-1 always share a split; reference images are train only.
  List: `config/splits/selected_images.csv` (path, class_id, type, size, sha1, split).
- Storage: `data/vision/c3pi/images/<disc>/<file>`, tracked by DVC, never by git.

## How to rebuild

```bash
cd vision
python -m tujya_vision.data_cli metadata         # READMEs, directory, ALLXML/*.xml (~130 MB)
python -m tujya_vision.data_cli index            # ALLXML -> data/vision/c3pi/metadata/images.csv
python -m tujya_vision.data_cli select --max-classes 150 --min-images 20 --max-per-class 30 \
    --exclude-camera "Canon PowerShot G10 [14.7 megapixel CCD]" \
    --exclude-camera "Nikon S3000 Coolpix (12.0 megapixel)"
python -m tujya_vision.data_cli download --workers 6   # resumable, SHA-1 checked
```

The first exploration fetched the same files with curl.

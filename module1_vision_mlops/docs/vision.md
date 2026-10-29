# Vision pipeline: detect, then identify

Module 1 vision axis. Two models, both Ultralytics YOLO11 fine-tuned from COCO / ImageNet
weights (transfer learning), tracked in MLflow, exported to ONNX.

```
photo ──> detector (YOLO11n, 1 class "pill") ──> one box per pill
             └─ crop each box (square, +20 % margin, 224 px)
                   └─> identifier (YOLO11s-cls, 150 classes) ──> top-5 pills + confidence
                          └─> class -> active ingredients (RxNav) -> DDInter interactions
```

Code: `vision/tujya_vision/` (`autolabel.py`, `prepare.py`, `train.py`), own venv
`vision/.venv` (`vision/requirements.txt`). Data: docs/DATASET_C3PI.md.

## Where the boxes come from (decision D38)

C3PI has no boxes (docs/DATASET_C3PI.md). Three options were considered:

1. Existing annotations: none in C3PI; ePillID provides crops, not boxes on the photos.
2. A public pill detection dataset (Roboflow and similar): licences are unclear or
   per-uploader, the images are not the C3PI photos, and it adds a second data source to
   document. Not used.
3. **Automatic labelling of the consumer photos** (chosen): each photo shows one or two
   pills on a fairly plain background, so classical computer vision finds them reliably
   enough, and doubtful photos can be detected and left out.

`autolabel.py`: robust quadratic background surface in Lab (absorbs vignetting), colour
distance with shadows down-weighted, bounded threshold, connected regions not touching the
border, coins (C3PI's size reference) rejected by their texture, crescent-shaped regions
(a shadow next to a pale pill) refined with GrabCut, faint light blobs dropped. Each photo
gets flags; a photo with a doubtful flag is excluded from **both** datasets.

The detector therefore learns from machine-made labels; it is evaluated against the same
kind of labels on the test photos. Multi-pill scenes are learnt from photos with two pills
(front and back) and from Ultralytics' mosaic augmentation, which pastes four photos
together.

## Datasets (DVC stage `prepare`)

`python -m tujya_vision.prepare` (or `dvc repro prepare`) writes, from
`config/splits/selected_images.csv`:

- `data/vision/c3pi/labels/autolabel.csv`: per image, boxes, flags, ok
- `data/vision/datasets/detect/`: YOLO format, long side 640 px, clean photos only
- `data/vision/datasets/classify/<split>/<class_id:03d>/`: 224 px square crops, at most two
  per photo (front and back), plus crops of the RxNav reference images (train only)

A photo and all its crops stay in the split of `selected_images.csv`.

## Training, evaluation, export, registry

```bash
docker compose up -d                                   # MLflow on http://localhost:5000
cd vision
python -m tujya_vision.train classify --epochs 20
python -m tujya_vision.train detect --epochs 10 --imgsz 640
```

Each run logs to the MLflow experiment `tujya-vision`: parameters, hardware tags, image
counts, per-epoch metrics (from Ultralytics' `results.csv`), test metrics, plots,
`best.pt`; then it exports `best.onnx`, checks ONNX Runtime against PyTorch on test images,
and registers the ONNX model as a new version of `tujya-pill-classifier` or
`tujya-pill-detector`.

## Auto-labelling result (2026-10-28)

| | Photos |
|---|---:|
| Consumer photos | 4,056 |
| Clean (used) | 3,306 (81.5 %) |
| Flag `irregular_region` (shadow or half a pill, GrabCut could not fix it) | 462 |
| Flag `no_region` (pill too close to the background colour, textured granite) | 346 |
| Other doubtful flags (elongated, too many regions, too large) | 73 |
| Fixed by GrabCut (`refined`, kept) | 311 |
| Coin present and ignored (kept) | 111 |

Boxes per clean photo: 1 (2,862), 2 (775), 3 (43), 4 (30).

**Audit by eye** of a random sheet of 48 photos (`python -m tujya_vision.prepare --sheet`):
40 passed the filters, 36 of them with every box correct (about 90 %). The errors are
mostly a faint rectangle of uneven light near the bottom edge of beige paper, and once a
missed second pill. Doubtful photos were correctly flagged. Improving the labels (hand
check of the test split, or a second labelling pass with the trained detector) is listed
in the next steps.

Datasets: identification 2,962 train / 562 val / 589 test crops (150 / 150 / 148
classes: two classes have no clean test photo); detection 2,345 / 478 / 483 photos.

## Results

### Identification (classifier)

| Run | Model | Device | Epochs | Test top-1 | Test top-5 | ONNX vs PyTorch |
|---|---|---|---:|---:|---:|---|
| smoke (pipeline check) | YOLO11n-cls | CPU | 3 | 19.4 % | 36.0 % | top-1 agree 100 %, max prob diff 1.5e-7 |
| main | YOLO11s-cls | RTX 2050 | 40 | **93.5 %** | **98.1 %** | top-1 agree 100 %, max prob diff 1.6e-6 |

Main run: 14 min of training, validation top-1 92.2 % / top-5 97.9 % at the last epoch,
MLflow run `7d2ce0ff7c53401aa42d2ef89c3aa794`, registered as `tujya-pill-classifier`
version 2 (version 1 is the smoke run). Test = 589 crops of photos never seen in training.

### Detection

| Run | Model | Device | Epochs | Test mAP50 | Test mAP50-95 | Precision | Recall | ONNX vs PyTorch |
|---|---|---|---:|---:|---:|---:|---:|---|
| main | YOLO11n, 640 px, batch 8 | RTX 2050 | 30 | **0.972** | **0.861** | 0.957 | 0.936 | same box count on 100 % of 50 test photos, mean IoU 0.976 |

MLflow run `5c6db57a144b483aaf37b94e74f5e2cb`, registered as `tujya-pill-detector` version 1.
Test = 483 clean test photos, scored against the automatic boxes (whose own precision is
about 90 %, see above), so these numbers measure agreement with the auto-labeller.

### End to end (what the app would do)

`python -m tujya_vision.evaluate` on **all 600 consumer photos of the test split** (also
the 117 the auto-labeller rejected), with the two **ONNX** exports: the detector runs on
the full photo, the most confident box is cropped and classified.

| Photos | Pill detected | Top-1 | Top-5 |
|---:|---:|---:|---:|
| 600 | 98.5 % | **89.2 %** | **95.8 %** |

MLflow run `9bfa9d81a606472892fbf67efe7036e1` (`evaluate-end-to-end`). Slower than needed
(25 min on CPU) because Ultralytics installed `onnxruntime-gpu` on the fly during this run;
auto-install is now disabled in `tujya_vision/__init__.py` (D42).

### Notes on the scores

What these numbers mean: train and test contain **different photos of the same physical
pills** (C3PI photographed each pill many times, under several cameras, lights and
backgrounds), and the test crops come from the automatic boxes, not from the detector. The
end-to-end score (detector boxes -> classifier) and a test on photos of other pill
instances would be lower; both are next steps.

## Comparison with ePillID

ePillID (Usuyama et al., 2020) is built from the same C3PI images but solves a harder
problem: thousands of appearance classes (pill product and side), few-shot, retrieval of
consumer photos against reference images. Tujya's identifier is a closed-set classifier
over 150 products with 20+ photos each, trained and tested on consumer photos of the same
products. The numbers are therefore **not comparable** to ePillID's; a fair comparison would
retrain on ePillID's split and metrics (MAP, GAP, top-k on the holdout), which is a later
step.

## Serving: the API's own pipeline and the confirmation threshold

`POST /api/scan` (docs/API.md) does not use Ultralytics: `backend/app/scanner.py`
re-implements the letterbox, NMS and 20 %-margin crop with numpy and OpenCV and runs the two
ONNX files with ONNX Runtime (D45). Checks:

- on 40 test photos, against Ultralytics' ONNX predictor: same number of boxes on 40/40,
  mean IoU of the main box 0.9999, same top-1 class on 39/39 (one photo without a box);
- on all 600 test photos (`backend/scripts/evaluate_scan.py --split test`): pill found
  98.5 %, top-1 **89.3 %**, top-5 **95.8 %**, as in the end-to-end evaluation above.

### Confirmation threshold

The app asks the user to confirm a pill when its top-1 probability is below a threshold
(`needs_confirmation`). Chosen on the **validation** photos (600, never used for training),
with `python scripts/evaluate_scan.py --split val`: rule = the lowest threshold at which at
most 1 % of all photos would be accepted with a wrong pill.

| Threshold | Accepted without confirmation | Top-1 among accepted | Wrong and accepted (share of all photos) |
|---:|---:|---:|---:|
| 0.5 | 87.2 % | 91.2 % | 7.7 % |
| 0.7 | 77.7 % | 95.3 % | 3.7 % |
| 0.8 | 71.7 % | 97.7 % | 1.7 % |
| **0.85** | **67.2 %** | **98.5 %** | **1.0 %** |
| 0.9 | 60.3 % | 98.9 % | 0.7 % |
| 0.95 | 48.5 % | 99.7 % | 0.2 % |

Validation overall: pill found 98.5 %, top-1 84.2 %, top-5 94.8 %. **Threshold 0.85**
(D47). On the test photos at 0.85: 70.5 % accepted, 99.3 % of them correct, 0.5 % of all
photos accepted with a wrong pill. A confirmed or corrected pill is always shown with its
probability, and the identification stays a suggestion.

Serving speed: ONNX Runtime's busy-waiting threads made OpenCV (and the whole process) about
30 times slower on the shared laptop CPU; spinning is now disabled in the sessions (D49),
giving about 50-220 ms per photo for decode, detection and identification.

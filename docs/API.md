# Tujya API (scan, pills, interactions)

FastAPI app in `backend/`. Swagger UI: `/docs`, schema: `/openapi.json` (D44). All routes
below need a signed-in account (session cookie from `POST /api/auth/login`), like the
catalog; the demo signs in with the demo account (D46).

Every answer about medicines carries `sources` (name, title, URL, licence, citation) and
`disclaimer` = "Information uniquement, consultez un pharmacien ou un médecin.". No route
ever returns a dose, and none ever calls a combination safe.

## POST /api/scan

Multipart form, one field `image`: JPEG, PNG or WebP, at most 10 MB and 50 megapixels
(`SYNERGYRX_SCAN_MAX_BYTES`, `SYNERGYRX_SCAN_MAX_PIXELS`). Errors: 401 not signed in, 413
too large, 415 not an image, 503 models or medicine tables missing (the message says which
command fixes it).

Pipeline (`backend/app/scanner.py`, ONNX Runtime, CPU by default, CUDA with
`SYNERGYRX_ONNX_PROVIDERS=auto`):

1. decode (OpenCV, EXIF orientation applied), letterbox to 640 px, detector
   (`tujya-pill-detector`, YOLO11n), confidence >= 0.25, NMS IoU 0.7, at most 20 pills;
2. each box: square crop with a 20 % margin, 224 px (exactly the training crops),
   classifier (`tujya-pill-classifier`, YOLO11s-cls), top-5;
3. class_id -> pill name, generic name and active ingredients from the medicine tables.

Response (`ScanResult`):

```json
{
  "image_width": 2560, "image_height": 1920,
  "inference_ms": 50.4,
  "timings": {"detection_ms": 41.2, "classification_ms": 9.2, "total_ms": 50.4},
  "confirm_threshold": 0.85,
  "pills": [{
    "box": {"x1": 1028, "y1": 908, "x2": 1522, "y2": 1140},
    "box_normalized": {"x1": 0.40, "y1": 0.47, "x2": 0.59, "y2": 0.59},
    "detection_confidence": 0.94,
    "candidates": [
      {"class_id": 0, "name": "AMRIX CAP 30MG", "generic_name": "CYCLOBENZAPRINE HYDROCHLORIDE",
       "active_ingredients": ["cyclobenzaprine"], "probability": 0.982}
    ],
    "needs_confirmation": false
  }],
  "message": null,
  "models": {"detect": {"registered_model": "tujya-pill-detector", "version": 1}, "classify": {}},
  "sources": [{"name": "c3pi"}, {"name": "rxnav"}],
  "disclaimer": "Information uniquement, consultez un pharmacien ou un médecin."
}
```

`needs_confirmation` is true when the top-1 probability is below `confirm_threshold`
(0.85, `SYNERGYRX_SCAN_CONFIRM_THRESHOLD`); the app then asks the user to pick from the
top-5 or search `GET /api/pills`. How 0.85 was chosen: docs/VISION.md, "Confirmation
threshold" (D47). With no pill found, `pills` is empty and `message` says so.

### Models

The API loads `detector.onnx`, `classifier.onnx` and `models.json` from
`SYNERGYRX_MODELS_DIR` (default `models/` at the repository root, git-ignored), on the first
scan. Fill it from the MLflow Model Registry (vision venv, MLflow server running):

```bash
cd vision
python -m tujya_vision.publish                 # latest registered versions
python -m tujya_vision.publish --from-runs     # or the newest vision/runs exports
```

The registered ONNX files keep their weights as MLflow external data; `publish` writes each
model as one self-contained file and records its registry version, run id and SHA-256 in
`models.json`, which `/api/scan` returns as `models`.

## GET /api/pills?q=&limit=

Search the 150 identification classes (name, generic name, imprint, maker, ingredient;
every word must match; a number also matches the class_id). `limit` 1-150, default 20.
Returns `items` (class_id, name, generic_name, labeled_by, imprint, color, shape,
active_ingredients), `total`, `sources`.

## POST /api/interactions

```json
{"pills": [12, 87], "ingredients": ["fluconazole"]}
```

Two to ten items in total: pill class_ids (from a scan or `/api/pills`) and/or ingredient
names. Every pair of items is checked (pill ingredients against the other item's
ingredients; the ingredients inside one combination product are not compared with each
other). For each pair: `interactions` (ingredient_a, ingredient_b, severity, description,
source), `highest_severity`, `shared_ingredients` (the same substance in both), and
`message` when nothing is listed:

- "Aucune interaction connue dans cette base de données." when both items are known;
- a "aucune vérification n'est possible" message when an item is unknown, or is a pill
  without known active ingredients (class 104, the multivitamin).

Pairs are sorted by severity (Major, Moderate, Minor, Unknown, then pairs without
interaction). "Unknown" is DDInter's ungraded level: an interaction exists. The top-level
`message` counts the pairs with an interaction and the pairs that could not be checked.

Data: the PostgreSQL tables loaded by `python -m app.pharma_loader` (docs/INFRA.md,
docs/INTERACTIONS.md). Tests: `backend/tests/test_scan.py` (tiny fake ONNX models built in
the test, `tests/fake_models.py`) and `backend/tests/test_medicines.py` (SQLite).

## Measured on the real models

`python scripts/evaluate_scan.py --split test` (backend venv) runs this exact pipeline on
the 600 consumer photos of the test split: pill found 98.5 %, top-1 89.3 %, top-5 95.8 %,
the same as the Ultralytics evaluation (docs/VISION.md); about 50 to 220 ms per photo on
the laptop CPU, depending on the load.

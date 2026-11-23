# API

FastAPI application in [backend/](../backend/). Swagger UI: http://localhost:8000/docs, schema:
`/openapi.json`. Request and response models are Pydantic schemas
([backend/app/schemas.py](../backend/app/schemas.py)). Every answer about medicines carries its
`sources` and the disclaimer; no route ever returns a dose or calls a combination safe.

## Endpoints of the module guide (no account)

The model alone, without accounts or medicine tables:
[backend/app/routers/vision.py](../backend/app/routers/vision.py).

| Route | Returns | Errors |
|---|---|---|
| `GET /health` | `status` (`ok` / `degraded`), `model_loaded`, GPU and ONNX Runtime providers, API version, and for each loaded model: registered name, version, aliases, precision, run id, SHA-256 | none: always 200 (`degraded` with `detail` when the models are missing) |
| `GET /model/metadata` | detector and classifier: name, architecture, Registry version and aliases, precision, run id, input size, validation and test metrics of the published version, classes | 503 models not loaded |
| `POST /predict` (multipart field `image`) | `count`, `pills` (box in pixels and normalised, detection confidence, `top5` classes with probabilities), `timings`, model versions, disclaimer | 400, 413, 422, 503 |
| `POST /predict/visualize` (multipart field `image`, `?max_side=1280`) | PNG with one box per pill, coloured by class; headers `X-Pill-Count`, `X-Inference-Ms`, `X-Server-Ms` | 400, 413, 422, 503 |
| `GET /monitoring/drift` | drift of the scanned photos: status (`unavailable`, `insufficient`, `ok`, `warning`, `drift`) and, for each of the eight statistics, means, KS statistic, p-value, PSI | none: always 200 |

### Error codes

| Code | When | Example body |
|---|---|---|
| 400 | the upload is not a readable JPEG, PNG or WebP image | `{"detail": "The file is not an image that can be read (JPEG, PNG or WebP)."}` |
| 413 | the upload is larger than `TUJYA_SCAN_MAX_BYTES` (10 MB) or has more pixels than `TUJYA_SCAN_MAX_PIXELS` | `{"detail": "..."}` |
| 422 | the multipart field `image` is missing (FastAPI validation) | `{"detail": [{"type": "missing", "loc": ["body", "image"], ...}]}` |
| 503 | `models/detector.onnx` or `models/classifier.onnx` is missing: run `python -m src.models.registry publish` | `{"detail": "..."}` |

### Examples

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model/metadata
curl -F "image=@photo.jpg;type=image/jpeg" http://127.0.0.1:8000/predict
curl -F "image=@photo.jpg;type=image/jpeg" -o boxes.png http://127.0.0.1:8000/predict/visualize
```

`POST /predict` on a demo gallery photo (C3PI test split, 4416 x 3312 px), answer shortened:

```json
{
  "image_width": 4416,
  "image_height": 3312,
  "count": 1,
  "pills": [
    {
      "box": {"x1": 1957.6, "y1": 1410.0, "x2": 2398.7, "y2": 1871.0},
      "box_normalized": {"x1": 0.443, "y1": 0.426, "x2": 0.543, "y2": 0.565},
      "confidence": 0.953,
      "top5": [{"class_id": 35, "probability": 0.862}, "..."]
    }
  ],
  "timings": {"detection_ms": 127.9, "classification_ms": 52.6, "inference_ms": 180.5, "server_ms": 393.5},
  "model_versions": "...",
  "disclaimer": "Information uniquement, consultez un pharmacien ou un médecin."
}
```

Class 35 is HALOPERIDOL 5MG TAB (`config/pill_classes.csv`). Timings depend on the machine; the
comparable CPU latencies of the models are in [module1/results.md](module1/results.md).

## Application routes (signed in)

Sign in with `POST /api/auth/login` (session cookie). The routes below need it unless noted.

| Route | Role |
|---|---|
| `POST /api/scan` (field `image`) | pills of a photo with their top-5, active ingredients and sources, or the ingredients read on a medicine box; `needs_confirmation` below 0.85; 413, 415, 503 |
| `GET /api/pills?q=&limit=` | search the 380 pill classes |
| `POST /api/interactions` | every pair among the given pills and ingredients, with DDInter severity and source; an absent pair is "no known interaction in this database", never "safe" |
| `POST /api/chat` | question to the agent; `stream: true` returns Server-Sent Events (`tool`, `token`, `final`, `error`); 503 when Ollama or the index is unavailable |
| `GET /api/documents`, `GET /api/documents/{doc_id}` | documents indexed for the RAG, their metadata and chunks |
| `POST /api/toxicity` | Tox21 in vitro assay probabilities of one or two molecules (bonus) |
| `GET /api/demo/images`, `GET /api/demo/images/{id}`, `.../gradcam` | demo gallery of test photos no served model saw in training, with precomputed Grad-CAM |
| `GET /api/catalog/summary`, `GET /api/catalog/showcase` | public figures and one real scan for the start page (no account) |
| `/api/auth/*`, `/api/users/{id}/{photo,signature}` | sign-up, sign-in, profile, password, profile photo and signature |
| `/api/my/medicines`, `/api/my/reports`, `POST /api/shared` | saved medicines, saved reports and their share links (a shared report is read without account) |
| `/api/admin/*` | administrator: accounts, statistics, activity log, usage, CSV exports |
| `POST /monitoring/drift/reset` | administrator: empty the drift window |
| `GET /api/health` | liveness probe of Docker Compose (no account) |

Example of a cited chat answer (`POST /api/chat`, `{"question": "Does warfarin interact with
fluconazole?", "stream": false}`), shortened:

```text
DDInter 2.0 lists an interaction between fluconazole and warfarin with severity Major [1]. The
Fluconazole label also mentions that prothrombin time may be increased in patients receiving
concomitant fluconazole and coumarin-type anticoagulants, such as warfarin [2].

Information uniquement, consultez un pharmacien ou un médecin.
```

Tests: [backend/tests/test_api.py](../backend/tests/test_api.py) covers every success case and
every error code of the guide endpoints with tiny fake ONNX models;
[backend/tests/test_docs.py](../backend/tests/test_docs.py) checks Swagger and the schema.

# Module 1 results

Every row comes from MLflow (`http://localhost:5000`, experiment `tujya-vision` unless noted);
the last column is the run id. Accuracy is measured on the **comparison split**
(`test_holdout.txt`: C3PI v2 test sessions that the v1 models never saw, see
[data.md](../data.md#datasets-built-for-training)). Latency is the ONNX Runtime CPU time of one
forward pass on the development laptop, all models timed together in interleaved rounds
(`python -m src.models.detection.compare_latency`). The champion is in bold.

## Detectors

Five architectures trained on Kaggle (Optuna search, then retraining of the best trial), plus
the v1 baseline. Comparison split: 995 photos, 1,133 pills. Precision and recall at the
confidence that maximises F1.

| Model | mAP@[.5:.95] val | **mAP@[.5:.95] test** | mAP50 test | Precision | Recall | mAP@[.5:.95] full v2 test | CPU latency (ms) | p95 (ms) | ONNX size (MB) | Parameters (M) | Registry | MLflow run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| **yolo11s** | **0.8677** | **0.8885** | **0.9739** | **0.952** | **0.936** | **0.8722** | **63.0** | **67.1** | **37.9** | **9.5** | **v9 champion** | **`894871cd`** |
| yolov10n | 0.8569 | 0.8860 | 0.9760 | 0.949 | 0.946 | 0.8703 | 21.9 | 23.9 | 9.4 | 2.3 | v10 challenger | `d28c1056` |
| yolov8s | 0.8626 | 0.8830 | 0.9747 | 0.950 | 0.941 | 0.8644 | 69.7 | 73.7 | 44.7 | 11.2 | v11 archived | `e4d25272` |
| yolov8n | 0.8563 | 0.8774 | 0.9750 | 0.947 | 0.942 | 0.8543 | 24.2 | 25.5 | 12.8 | 3.2 | v2 archived | `b0689536` |
| fasterrcnn | 0.8063 | 0.8218 | 0.9723 | 0.951 | 0.944 | 0.8097 | 420.9 | 435.5 | 165.6 | 41.3 | v12 archived | `96b290a7` |
| yolo11n (baseline v1) |  | 0.7888 | 0.9513 | 0.931 | 0.940 | 0.8050 | 23.8 | 25.7 | 10.7 | 2.6 | v1 archived | `6ae2678a` |

The table between the markers below is rewritten by `python -m src.models.detection.compare`
(same values as above, without the bold):

<!-- detectors:begin -->
| Model | mAP@[.5:.95] val | mAP@[.5:.95] test | mAP50 test | Precision | Recall | mAP@[.5:.95] full v2 test | ONNX CPU mean latency (ms) | p95 (ms) | ONNX size (MB) | Parameters (M) | Version | Status | MLflow run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| yolo11s | 0.8677 | 0.8885 | 0.9739 | 0.952 | 0.936 | 0.8722 | 63.0 | 67.1 | 37.9 | 9.5 | v9 | champion | `894871cd` |
| yolov10n | 0.8569 | 0.8860 | 0.9760 | 0.949 | 0.946 | 0.8703 | 21.9 | 23.9 | 9.4 | 2.3 | v10 | challenger | `d28c1056` |
| yolov8s | 0.8626 | 0.8830 | 0.9747 | 0.950 | 0.941 | 0.8644 | 69.7 | 73.7 | 44.7 | 11.2 | v11 | archived | `e4d25272` |
| yolov8n | 0.8563 | 0.8774 | 0.9750 | 0.947 | 0.942 | 0.8543 | 24.2 | 25.5 | 12.8 | 3.2 | v2 | archived | `b0689536` |
| fasterrcnn | 0.8063 | 0.8218 | 0.9723 | 0.951 | 0.944 | 0.8097 | 420.9 | 435.5 | 165.6 | 41.3 | v12 | archived | `96b290a7` |
| yolo11n (baseline v1) |  | 0.7888 | 0.9513 | 0.931 | 0.940 | 0.8050 | 23.8 | 25.7 | 10.7 | 2.6 | v1 | archived | `6ae2678a` |
<!-- detectors:end -->

## Classifiers

Six architectures trained on the 380 classes of C3PI v2 (Kaggle, Optuna), plus the v1 champion
(150 classes; the 230 classes it does not know count as errors). Comparison split: 1,131 crops.
"top-1 v1 classes": the 275 comparison crops of the 150 v1 classes, prediction restricted to
them (the guard of the promotion).

| Model | top-1 val | **top-1 test** | top-5 test | macro F1 test | top-1 v1 classes | top-1 full v2 test | CPU latency (ms) | ONNX size (MB) | Parameters (M) | Registry | MLflow run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| **efficientnetv2s** | **0.9302** | **0.9089** | **0.9841** | **0.8790** | **0.9527** | **0.9136** | **17.6** | **82.5** | **20.6** | **v8 champion** | **`f1ed9edd`** |
| resnet50 | 0.9138 | 0.9027 | 0.9673 | 0.8665 | 0.9418 | 0.9054 | 18.6 | 97.1 | 24.3 | v9 challenger | `a53eb8fa` |
| convnext-tiny | 0.9288 | 0.8966 | 0.9814 | 0.8609 | 0.9491 | 0.9101 | 32.7 | 112.6 | 28.1 | v10 archived | `8d2dac1f` |
| yolo11s-cls | 0.8925 | 0.8594 | 0.9646 | 0.8021 | 0.8982 | 0.8710 | 4.5 | 23.7 | 5.9 | v11 archived | `9c8b058d` |
| vit-b16 | 0.8966 | 0.8515 | 0.9752 | 0.7888 | 0.8691 | 0.8517 | 84.9 | 344.6 | 86.1 | v12 archived | `64a5682a` |
| baseline-cnn (from scratch) | 0.4346 | 0.3669 | 0.6101 | 0.2673 | 0.4618 | 0.3631 | 11.4 | 11.1 | 2.8 | v13 archived | `4d756fb9` |
| yolo11s-cls (champion v1, 150 classes) |  | 0.1830 | 0.2246 | 0.1119 | 0.7527 | 0.4466 | 4.4 | 22.6 | 5.6 | v1 archived | `2b4fd382` |

<!-- classifiers:begin -->
| Model | top-1 val | top-1 test | top-5 test | macro F1 test | top-1 v1 classes | top-1 full v2 test | ONNX CPU latency (ms) | ONNX size (MB) | Parameters (M) | Version | Status | MLflow run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| efficientnetv2s | 0.9302 | 0.9089 | 0.9841 | 0.8790 | 0.9527 | 0.9136 | 17.6 | 82.5 | 20.6 | v8 | champion | `f1ed9edd` |
| resnet50 | 0.9138 | 0.9027 | 0.9673 | 0.8665 | 0.9418 | 0.9054 | 18.6 | 97.1 | 24.3 | v9 | challenger | `a53eb8fa` |
| convnext-tiny | 0.9288 | 0.8966 | 0.9814 | 0.8609 | 0.9491 | 0.9101 | 32.7 | 112.6 | 28.1 | v10 | archived | `8d2dac1f` |
| yolo11s-cls | 0.8925 | 0.8594 | 0.9646 | 0.8021 | 0.8982 | 0.8710 | 4.5 | 23.7 | 5.9 | v11 | archived | `9c8b058d` |
| vit-b16 | 0.8966 | 0.8515 | 0.9752 | 0.7888 | 0.8691 | 0.8517 | 84.9 | 344.6 | 86.1 | v12 | archived | `64a5682a` |
| baseline-cnn | 0.4346 | 0.3669 | 0.6101 | 0.2673 | 0.4618 | 0.3631 | 11.4 | 11.1 | 2.8 | v13 | archived | `4d756fb9` |
| yolo11s-cls (champion v1, 150 classes) |  | 0.1830 | 0.2246 | 0.1119 | 0.7527 | 0.4466 | 4.4 | 22.6 | 5.6 | v1 | archived | `2b4fd382` |
<!-- classifiers:end -->

## ONNX variants of the champions

`python -m src.models.optimize` (source: `models/optimize.json`). The served variant is the
fastest one that loses at most 0.01 of accuracy and is at least 5 % faster than FP32
(`params.yaml` `optimize`); none qualifies, so **FP32 is served** for both models.

| Model | Variant | Accuracy (metric) | Change vs FP32 | Mean latency (ms) | p95 (ms) | Size (MB) | Registry | MLflow run |
|---|---|---:|---:|---:|---:|---:|---|---|
| tujya-pill-detector | **fp32 (served)** | **0.8886** (test_v2_mAP50_95) | 0 | **61.9** | 65.3 | 37.9 | v9 | `313d04a1` |
| tujya-pill-detector | fp16 | 0.8886 | 0 | 59.5 | 63.0 | 19.0 | v13 | `10ffbd94` |
| tujya-pill-detector | int8-static | 0.8247 | -0.0639 | 138.5 | 164.3 | 12.6 | v14 | `3100ee5c` |
| tujya-pill-detector | int8-dynamic | 0.8615 | -0.0271 | 104.6 | 113.0 | 9.9 | v15 | `25f25f7f` |
| tujya-pill-classifier | **fp32 (served)** | **0.9089** (test_v2_top1) | 0 | **18.3** | 21.7 | 82.5 | v8 | `780779e2` |
| tujya-pill-classifier | fp16 | 0.9089 | 0 | 18.1 | 20.4 | 41.4 | v14 | `f2b5956e` |
| tujya-pill-classifier | int8-static | 0.7772 | -0.1317 | 38.1 | 41.1 | 23.4 | v15 | `d8ccda87` |
| tujya-pill-classifier | int8-dynamic | 0.8966 | -0.0124 | 63.2 | 70.4 | 21.6 | v16 | `f0c91bd2` |

FP16 is 3.9 % faster for the detector and 1.1 % for the classifier, below the 5 % threshold.

## v1 baseline (150 classes)

The first models, trained on C3PI v1 and recorded by `python -m src.models.train baseline`
(source: `models/export.json`, `models/evaluate.json`; v1 test split, 600 photos):

| Measure | Value | MLflow run |
|---|---:|---|
| Detector YOLO11n, test mAP@[.5:.95] | 0.8606 | `6766123e` |
| Classifier YOLO11s-cls, test top-1 | 0.9371 | `6dd616a6` |
| ONNX vs PyTorch: same top-1 / mean box IoU | 1.000 / 0.976 | same runs |
| End to end (detector -> crop -> classifier, ONNX): pill found | 0.985 | `d5f6312d` |
| End to end top-1 / top-5 | 0.8917 / 0.9583 | `d5f6312d` |

## Medicine boxes: OCR engines

`python -m src.models.ocr.evaluate --engine <name>` (experiment `tujya-boxes`, Registry
`tujya-box-ocr`); every photo goes through the API's own code, RxNav off.

| Measure | **RapidOCR** | EasyOCR |
|---|---:|---:|
| Active ingredient found (`ingredient_recall`, promotion metric) | **0.767** | 0.480 |
| All active ingredients found | 0.749 | 0.455 |
| Precision of the ingredients found | 0.996 | 1.000 |
| Boxes with a wrong ingredient | 0.004 | 0.000 |
| Brand read | 0.929 | 0.800 |
| Box classified as a box / as a pill | 0.950 / 0.032 | 0.686 / 0.189 |
| C3PI pill classified as a pill | 0.980 | 0.980 |
| Off-topic photo refused | 0.789 | 0.789 |
| Median / p95 latency per box (ms) | 3,413 / 4,036 (CPU) | 416 / 949 (GPU) |
| Registry | **v1 champion** | v2 challenger |
| MLflow run | **`91e00641`** | `f375874e` |

RapidOCR runs in ONNX, so the API can serve it without PyTorch.

## Data drift monitor

Five scenarios of 200 photos each, scored like a window of the API
(`python -m src.data.drift_demo --mlflow`, experiment `tujya-monitoring`; reference run
`17ede5a6`, 600 validation photos). A statistic drifts when the KS p-value is below 0.01 and
its PSI reaches the threshold of `params.yaml` `drift`.

| Scenario | Status | Drifting statistics | Highest PSI | MLflow run |
|---|---|---:|---|---|
| `test-v2` (new photos of the same kind) | ok | 0 | megapixels 0.53 | `55652620` |
| `train-v2` (photos the models trained on) | drift | 3 | classifier confidence 1.38 | `d7658129` |
| `dark` (grey levels x 0.35) | drift | 4 | brightness 8.28 | `eb45e5e4` |
| `blurred` (Gaussian blur) | drift | 4 | sharpness 8.28 | `fd737b1d` |
| `out-of-distribution` (boxes, off-topic images) | drift | 7 | megapixels 8.34 | `9f715191` |

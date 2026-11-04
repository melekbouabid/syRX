"""Comparison of pill detectors (step 2.2, docs/module1/kaggle_detecteurs.md).

One code path for every candidate architecture (YOLOv8n, YOLOv8s, YOLO11s, YOLOv10n with
Ultralytics; Faster R-CNN with torchvision), so the comparison is fair:

- `metrics`        COCO-style mAP@[.5:.95], mAP50, precision and recall computed by the same
                   function for every model (and for the baseline);
- `latency`        ONNX Runtime CPU latency of an exported model;
- `labels`         YOLO label files <-> pixel boxes, dataset listing and subsets;
- `hpo`            Optuna search spaces read from config/hpo/detector_<arch>.yaml;
- `runner`         the experiment run by each Kaggle notebook: Optuna search, retraining of
                   the best trial, test evaluation, ONNX export, export folder;
- `yolo_backend`   Ultralytics training / prediction / export;
- `import_kaggle`  imports an export folder into the local MLflow (parent run + one child
                   run per Optuna trial), registers the ONNX model and calls promote_if_better;
- `evaluate_champion`  re-evaluates the current champion on the same comparison split;
- `package_kaggle` packs the dataset to upload as a private Kaggle dataset;
- `notebook`       builds notebooks/kaggle/detector_<arch>.ipynb;
- `compare`        comparison table and figure for docs/.
"""

"""Comparison of pill classifiers (step 2.3, docs/module1/kaggle_classifieurs.md).

Same organisation as the detectors of step 2.2 (src/models/detection): one code path for the
six candidates (a small CNN trained from scratch as the baseline, ResNet-50, EfficientNetV2-S,
ConvNeXt-Tiny, ViT-B/16 and YOLO11s-cls, all built by src/models/architecture.py), so the
comparison is fair:

- `data`           the crop dataset of stage prepare_classify_v2: index, classes, subsets,
                   the preprocessing of the API (square crop resized, RGB in [0, 1]);
- `metrics`        top-1, top-5, macro F1, confusion matrix, per-class scores, computed the same
                   way for every model and for the v1 champion;
- `hpo`            Optuna search spaces read from config/hpo/classifier_<arch>.yaml;
- `torch_backend`  training (transfer learning then selective fine-tuning, class weights or
                   focal loss, RandAugment / MixUp / CutMix on the train split only),
                   prediction, ONNX export with the class list, Grad-CAM pictures;
- `runner`         the experiment run by each Kaggle notebook: Optuna search, retraining of
                   the best trial, test scores, ONNX export, export folder;
- `import_kaggle`  imports an export into the local MLflow (parent run + one child run per
                   trial), registers the ONNX model and calls promote_if_better;
- `evaluate_champion`  scores the current champion (150 classes of v1) on the same split;
- `package_kaggle` packs the crop dataset for a private Kaggle dataset;
- `notebook`       builds notebooks/kaggle/classifier_<arch>.ipynb;
- `compare`        comparison table and figure for docs/.
"""

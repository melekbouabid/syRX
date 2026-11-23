# Key decisions

The choices that shape the project, with the evidence behind each. Figures come from the result
files and MLflow runs listed in [module1/results.md](module1/results.md) and
[module2/results.md](module2/results.md).

1. **Split by shooting session, checked for leakage.** C3PI gives no id of the physical pill, so
   a session (package, camera, background, light, date) is the unit kept together, and sessions
   linked by a duplicate photo are merged. The leakage check finds 0 shared files, 0 shared
   sessions and 0 near-duplicates across splits ([data.md](data.md#split-without-leakage)).

2. **A comparison split for v1 versus v2.** v1 was not split by session, so every model, the v1
   baselines included, is scored on `test_holdout.txt`: test sessions the v1 models never saw.
   Without it, the v1 baselines would be scored on photos they trained on.

3. **Detector: YOLO11s.** Best mAP@[.5:.95] on the comparison split (0.8885) of five candidates
   trained on Kaggle with Optuna; it beat the champion in place (YOLOv8n, 0.8774) by more than
   the 0.005 minimum gain. YOLOv10n (0.8860, 21.9 ms) is the challenger. Faster R-CNN is 6.7 times
   slower and less accurate (0.8218, 420.9 ms).

4. **Classifier: EfficientNetV2-S.** Best top-1 on the comparison split (0.9089) of six
   architectures (a CNN from scratch, ResNet-50, ConvNeXt-Tiny, ViT-B/16, YOLO11s-cls), at
   17.6 ms per crop on the CPU. Transfer learning then selective fine-tuning; augmentation of the
   train split only.

5. **Champion rule `promote_if_better`.** A new Registry version gets the alias `champion` only
   if it beats the current one by `min_gain` (0.005) on the promotion metric; otherwise it is
   challenger or archived. Exports are imported best first, so the result does not depend on the
   import order. A classifier trained on another class list must also stay within 0.01 of the
   v1 champion on the v1 classes (guard metric `test_v2_v1c_top1`).

6. **ONNX FP32 is served.** FP16 keeps the accuracy but is only 3.9 % faster for the detector
   (61.9 -> 59.5 ms) and 1.1 % for the classifier, below the 5 % noise threshold; INT8 loses
   0.0124 to 0.1317 of accuracy and is slower on this CPU. The API serves ONNX Runtime models
   only: no PyTorch in the backend image.

7. **The API exposes the model alone.** `/health`, `/model/metadata`, `/predict` and
   `/predict/visualize` need no account and read no medicine table; the app's own `/api/scan`
   adds names, ingredients, sources and the confirmation step (below 0.85 of top-1).

8. **Qdrant for the RAG index.** It replaced pgvector: payload filters by ingredient, native
   BM25 sparse vectors and hybrid fusion, and a dashboard where the indexed chunks can be
   inspected. The SQL store stays for the tests and as the pgvector baseline.

9. **Hybrid search with query rewriting.** Chosen on a grid of retrieval configurations, one
   MLflow run each, by a rule written before the results: best MRR among the configurations
   that answer within 2 s per question. On the 380-class corpus the winner is bge-m3 dense +
   BM25 (RRF) on the LLM's English rewrite of the question: MRR 0.844, 394 ms. Half the
   questions are in French and BM25 only matches English words.

10. **No reranker served.** Cross-encoders raise MRR up to 0.922 but cost 3.1 to 15.6 s per
    question on the CPU, above the 2 s budget. It stays available (`TUJYA_RAG_RERANK=true`).

11. **Fine-tuned bge-m3 not served.** Fine-tuning on Kaggle raises the dense MRR of the benchmark
    from 0.833 to 0.916, but its training pairs come from the same corpus as the benchmark
    questions, so the gain may not carry to new questions, and the model needs PyTorch
    (sentence-transformers), which the API does not ship. The base bge-m3 served by Ollama is kept.

12. **qwen2.5:3b with prompt v1.** Selection score (mean of four RAGAS metrics and MLflow
    correctness) 0.7613 against 0.7076 for prompt v2, paired bootstrap 95 % interval +0.0102 to
    +0.0894. llama3.2:3b scores 0.7676 but takes 24.2 s per answer, above the 15 s budget
    (qwen2.5:3b: 4.9 s). Every configuration must pass the safety gates first.

13. **Chat safety is code, not only prompt.** A 3B model does not follow every instruction, so a
    safety node checks every sentence: a dose question is refused before any LLM call, dose and
    "safe" sentences are removed, timing advice needs a source, citations must point to real
    sources, the disclaimer always ends the answer. Measured on 96 questions: 0 dose sentences,
    0 "safe" sentences, disclaimer rate 1.0, dose questions refused 1.0.

14. **Drift reference on the validation split.** The models are more confident on their training
    photos than on any new photo (classifier confidence PSI 1.38 in the `train-v2` scenario), so a
    train reference would raise an alert from the first day; the validation split was never
    trained on and is drawn like the test photos.

15. **Strictly local.** No public deployment: every port is bound to 127.0.0.1, the CI builds
    and tests the images without publishing them, the DVC remote is a local folder.

# Notebooks

Two kinds of notebooks:

- **Summary notebooks** (`module1/0*.ipynb`, `module2/0*.ipynb`): executed and committed **with
  their outputs**. They train nothing and write nothing: every table is read from MLflow or from
  the result files (`models/`, `src/evaluation/results/`, `data/processed/`) with the functions
  of [src/](../src/), and their figures are the ones of [docs/module1/results.md](../docs/module1/results.md)
  and [docs/module2/results.md](../docs/module2/results.md).
- **Kaggle notebooks** (`module1/kaggle/`): generated from one template, committed **without
  outputs** (`nbstripout`), one per architecture; a test checks they are exactly the generated ones.

## Summary notebooks

| Notebook | What it shows | Needs |
|---|---|---|
| [module1/01_data_exploration_cleaning](module1/01_data_exploration_cleaning.ipynb) | classes, images, class distribution, example photos, rejected images and why, split without leakage | DVC data |
| [module1/02_detectors_comparison](module1/02_detectors_comparison.ipynb) | 5 detectors + v1 baseline: mAP50, mAP50-95, precision, recall, latency; figure; champion | MLflow |
| [module1/03_classifiers_comparison](module1/03_classifiers_comparison.ipynb) | 6 classifiers + v1 champion: top-1, top-5, macro F1, latency; confusion matrix and Grad-CAM of the champion | MLflow |
| [module1/04_onnx_optimization](module1/04_onnx_optimization.ipynb) | FP32 / FP16 / INT8: accuracy, size, latency; served variant and its rule | `models/optimize.json` |
| [module1/05_end_to_end_prediction](module1/05_end_to_end_prediction.ipynb) | a test photo through the served pipeline (boxes, classes, confidence), a medicine box (OCR), an off-topic picture refused | DVC data, `models/` |
| [module1/06_kaggle_training_runs](module1/06_kaggle_training_runs.ipynb) | every Kaggle run: Optuna trials and search space, training curve, PR curve or confusion matrix, duration, GPU; summary | MLflow |
| [module2/01_rag_retrieval](module2/01_rag_retrieval.ipynb) | the pipeline diagram and `src/rag/` file by file (step 1 to 6); labs 10 to 16: the Qdrant collection, one question in dense, BM25 and hybrid mode, with query rewriting and with reranking | Qdrant, Ollama |
| [module2/02_retrieval_benchmark](module2/02_retrieval_benchmark.ipynb) | as lab 05/03: questions and qrels, configuration table, champion and its 2 s rule; every metric of the course (Hit, Precision, Recall, F1, nDCG @k, R-Precision, MRR, MAP) on the corpus v2, the Recall@k curve and the choice of k, the recorded reranker | result files |
| [module2/03_ragas_evaluation](module2/03_ragas_evaluation.ipynb) | RAGAS, citation judge, safety gates, generators table, champion; fine-tuned bge-m3 and why it is not served; Exact Match, token F1, ROUGE, BLEU, semantic similarity; context utilization and noise sensitivity; the judge against human labels (Cohen's kappa) | result files |
| [module2/04_langgraph_agent](module2/04_langgraph_agent.ipynb) | the graph (`draw_mermaid`), the tools, an interaction question with citations, a dose question refused, the MCP tools | Qdrant, Ollama |
| [module2/05_embedding_finetuning](module2/05_embedding_finetuning.ipynb) | the bge-m3 fine-tuning run on Kaggle: parameters, GPU, loss, scores before / after | MLflow |

Run them again from the repository root with the project venv (`.venv`) and the stack running
(`docker compose up -d`):

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/module1/02_detectors_comparison.ipynb
```

## Kaggle training runs

Every compared model was trained on Kaggle (Tesla T4) from its generated notebook, on its
experiment branch (`experiment/detector-<arch>`, `experiment/classifier-<arch>`); the export
was imported into MLflow by `import_kaggle` (one parent run, one child run per Optuna trial).
Generate a notebook with `python -m src.models.detection.notebook <arch>` or
`python -m src.models.classification.notebook <arch>`. Values below are read from MLflow
(experiments `tujya-vision` and `tujya-rag-finetuning`).

| Model | Generated Kaggle notebook | MLflow run | Main metric (comparison split) | Registry |
|---|---|---|---|---|
| **yolo11s** | [kaggle/detector_yolo11s.ipynb](module1/kaggle/detector_yolo11s.ipynb) | `894871cd` | test_v2_mAP50_95 0.8885 | v9 **champion** |
| yolov10n | [kaggle/detector_yolov10n.ipynb](module1/kaggle/detector_yolov10n.ipynb) | `d28c1056` | test_v2_mAP50_95 0.8860 | v10 challenger |
| yolov8s | [kaggle/detector_yolov8s.ipynb](module1/kaggle/detector_yolov8s.ipynb) | `e4d25272` | test_v2_mAP50_95 0.8830 | v11 archived |
| yolov8n | [kaggle/detector_yolov8n.ipynb](module1/kaggle/detector_yolov8n.ipynb) | `b0689536` | test_v2_mAP50_95 0.8774 | v2 archived |
| fasterrcnn | [kaggle/detector_fasterrcnn.ipynb](module1/kaggle/detector_fasterrcnn.ipynb) | `96b290a7` | test_v2_mAP50_95 0.8218 | v12 archived |
| **efficientnetv2s** | [kaggle/classifier_efficientnetv2s.ipynb](module1/kaggle/classifier_efficientnetv2s.ipynb) | `f1ed9edd` | test_v2_top1 0.9089 | v8 **champion** |
| resnet50 | [kaggle/classifier_resnet50.ipynb](module1/kaggle/classifier_resnet50.ipynb) | `a53eb8fa` | test_v2_top1 0.9027 | v9 challenger |
| convnext-tiny | [kaggle/classifier_convnext-tiny.ipynb](module1/kaggle/classifier_convnext-tiny.ipynb) | `8d2dac1f` | test_v2_top1 0.8966 | v10 archived |
| yolo11s-cls | [kaggle/classifier_yolo11s-cls.ipynb](module1/kaggle/classifier_yolo11s-cls.ipynb) | `9c8b058d` | test_v2_top1 0.8594 | v11 archived |
| vit-b16 | [kaggle/classifier_vit-b16.ipynb](module1/kaggle/classifier_vit-b16.ipynb) | `64a5682a` | test_v2_top1 0.8515 | v12 archived |
| baseline-cnn | [kaggle/classifier_baseline-cnn.ipynb](module1/kaggle/classifier_baseline-cnn.ipynb) | `4d756fb9` | test_v2_top1 0.3669 | v13 archived |
| bge-m3 fine-tuning | branch `experiment/fine-tuning-embedding` | `03c72c9c` | validation MRR@10 0.960 | not served |

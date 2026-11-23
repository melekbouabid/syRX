# Module 2 results

Benchmark set: 96 hand-checked questions ([src/evaluation/questions.jsonl](../../src/evaluation/questions.jsonl),
49 in French, 47 in English); 85 of them have expected chunks
([src/evaluation/qrels.json](../../src/evaluation/qrels.json)) and are used for retrieval. Every
configuration searches like the agent: the ingredients found in the question are a metadata
filter, top 10 kept. Last column: MLflow run id. The served choice is in bold.

## Retrieval: first benchmark

Corpus of the 150 first classes; `names` = ingredient filter. Source:
[src/evaluation/results/retrieval.json](../../src/evaluation/results/retrieval.json), experiment
`tujya-rag`.

| Configuration | Recall@1 | Recall@5 | Recall@10 | MRR@10 | MLflow run |
|---|---:|---:|---:|---:|---|
| BM25, no filter | 0.459 | 0.671 | 0.718 | 0.546 | `b7f6a7bf` |
| dense, no filter | 0.765 | 0.882 | 0.953 | 0.821 | `eb47d971` |
| hybrid, no filter | 0.682 | 0.871 | 0.918 | 0.767 | `34f0001a` |
| BM25 + filter | 0.518 | 0.753 | 0.776 | 0.611 | `2962bc45` |
| dense + filter | 0.776 | 0.906 | 0.953 | 0.832 | `04ffd655` |
| hybrid + filter | 0.729 | 0.871 | 0.953 | 0.803 | `b40885ee` |
| BM25 + filter + rewriting | 0.694 | 0.882 | 0.953 | 0.779 | `bc85a985` |
| dense + filter + rewriting | 0.765 | 0.941 | 1.000 | 0.835 | `ab3025e3` |
| hybrid + filter + rewriting | 0.788 | 0.929 | 0.988 | 0.860 | `bf495f6b` |
| dense + filter + reranker | 0.800 | 0.941 | 0.953 | 0.861 | `eaba5635` |
| hybrid + filter + rewriting + reranker | 0.824 | 0.965 | 0.988 | 0.889 | `a02130cf` |

The ingredient filter and query rewriting help every mode; rewriting matters most for BM25
because half the questions are in French and the corpus is in English.

## Retrieval: configuration grid

`python -m src.evaluation.retrieval_grid` (experiment `tujya-rag-retrieval`, source:
[src/evaluation/results/grid/](../../src/evaluation/results/grid/)). Time per question is the
median of interleaved rounds on the CPU (rewrite + search + rerank). Rule, written before the
results ([config/rag_grid.yaml](../../config/rag_grid.yaml)): best MRR among the configurations
the API can serve within 2 s per question.

| Group | Configuration | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | ms / question | MLflow run |
|---|---|---:|---:|---:|---:|---:|---:|---|
| corpus v2 (380 classes) | **bge-m3 hybrid + rewriting** | **0.788** | **0.906** | **0.977** | **0.844** | **0.789** | **394** | **`9c641b60`** |
| corpus v2 | bge-m3 dense (challenger) | 0.776 | 0.894 | 0.953 | 0.831 | 0.777 | 88 | `91f056fa` |
| corpus v2 | bge-m3 dense + rewriting | 0.765 | 0.941 | 0.988 | 0.831 | 0.771 | 397 | `1aeb957a` |
| corpus v2 | bge-m3 hybrid | 0.729 | 0.871 | 0.953 | 0.803 | 0.762 | 100 | `6e473f31` |
| corpus v2 | BM25 | 0.482 | 0.753 | 0.765 | 0.583 | 0.583 | 10 | `81765db7` |
| bge-m3 (v1 corpus) | dense + rewriting | 0.765 | 0.941 | 0.988 | 0.834 | 0.776 | 389 | `f9954762` |
| bge-m3 (v1 corpus) | hybrid + rewriting | 0.753 | 0.906 | 0.965 | 0.824 | 0.778 | 394 | `3ae434b6` |
| multilingual-e5-base | hybrid + rewriting | 0.718 | 0.906 | 0.988 | 0.808 | 0.760 | 534 | `72410c2b` |
| all-MiniLM-L6-v2 | hybrid + rewriting | 0.718 | 0.882 | 0.977 | 0.795 | 0.714 | 351 | `f93b84e7` |
| chunking | hybrid + rewriting, 800 characters | 0.812 | 0.941 | 0.988 | 0.868 | 0.780 | 389 | `750b72a6` |
| chunking | hybrid + rewriting, 2,500 characters | 0.788 | 0.941 | 1.000 | 0.853 | 0.810 | 386 | `1136acbc` |
| rerankers | hybrid + rewriting + MiniLM-L-6 (10 candidates) | 0.859 | 0.941 | 0.977 | 0.896 | 0.822 | 3,617 | `0289f558` |
| rerankers | hybrid + rewriting + MiniLM-L-12 (20 candidates) | 0.894 | 0.965 | 1.000 | 0.922 | 0.847 | 11,251 | `254574f3` |
| rerankers | hybrid + rewriting + bge-reranker-base (10) | 0.800 | 0.941 | 0.977 | 0.863 | 0.799 | 15,616 | `bd1f7d69` |

Rerankers are above the 2 s budget, so none is served. The selection runs on the served corpus
v2 (`selection.corpus` in champion.json); the chunking variants were measured on the v1 corpus,
and another chunking would need the served index rebuilt, which the rule allows only for a
significant gain (paired bootstrap).
The selection is registered as `tujya-rag-retrieval` ([grid/champion.json](../../src/evaluation/results/grid/champion.json)).

## Embedding fine-tuning (not served)

bge-m3 fine-tuned on Kaggle (MLflow run `03c72c9c`, experiment `tujya-rag-finetuning`): MRR@10
on its own validation pairs 0.784 -> 0.960. On the 85 benchmark questions, through the same
sentence-transformers path ([grid/bge-m3-finetuned.json](../../src/evaluation/results/grid/bge-m3-finetuned.json)):

| Configuration | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | MLflow run |
|---|---:|---:|---:|---:|---:|---|
| bge-m3 base, dense | 0.776 | 0.906 | 0.953 | 0.833 | 0.781 | `e5300cff` |
| bge-m3 fine-tuned, dense | 0.906 | 0.929 | 0.941 | 0.916 | 0.746 | `1b6da11e` |
| bge-m3 base, hybrid + rewriting | 0.753 | 0.906 | 0.988 | 0.827 | 0.782 | `9c2504bf` |
| bge-m3 fine-tuned, hybrid + rewriting | 0.859 | 0.965 | 0.988 | 0.898 | 0.780 | `d96fc302` |

Not served: the API has no PyTorch to run it, and the training pairs come from the same corpus
as the benchmark ([decisions.md](../decisions.md)).

## Generation: RAGAS, judge and MLflow GenAI

`python -m src.evaluation.benchmark_generation` then `python -m src.evaluation.ragas_evaluation`
(experiment `tujya-rag-generation`, source:
[src/evaluation/results/generation/](../../src/evaluation/results/generation/)). 96 questions,
the full agent with the served retrieval. RAGAS 0.4.4 and the citation judge run on a local
judge of another family (phi4-mini). Score = mean of faithfulness, answer relevancy, context
precision, context recall and MLflow correctness; the safety gates must all pass and the median
time must stay under 15 s.

| Configuration | Faithfulness | Answer relevancy | Context precision | Context recall | Correctness | Cited sentences supported | Score | Median s | MLflow run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **qwen2.5:3b, prompt v1** | **0.9315** | **0.7039** | **0.9401** | **0.6951** | **0.5357** | **0.9507** | **0.7613** | **4.9** | **`dd31625f`** |
| qwen2.5:3b, prompt v2 | 0.8798 | 0.6726 | 0.9408 | 0.6667 | 0.3780 | 0.8816 | 0.7076 | 4.5 | `31760b16` |
| llama3.2:3b, prompt v1 | 0.9065 | 0.7472 | 0.9334 | 0.7388 | 0.5119 | 0.9360 | 0.7676 | 24.2 (over budget) | `c5f24fbe` |
| qwen2.5:3b, prompt v1, corpus v2 | 0.9122 | 0.6420 | 0.9234 | 0.7240 | 0.5122 | 0.9357 | | 4.9 | `83c87200` |

Champion minus challenger (paired bootstrap, 95 % interval): +0.0496 (+0.0102 to +0.0894)
([generation/champion.json](../../src/evaluation/results/generation/champion.json)). The
corpus-v2 run is scored separately: two corpora are not compared.

Safety gates, same for all four runs: 0 dose sentences, 0 "safe" sentences, disclaimer rate 1.0,
dose questions refused 1.0. MLflow safety judge: 0.8333 (qwen2.5:3b v1).

Judge control ([judge_control.json](../../src/evaluation/results/judge_control.json)): paired
with a source that cannot support the claim, the judge said "yes" 4 times out of 40 (0.10).

## Brand questions

8 questions naming a medicine brand ("Panadol", "Doliprane"...), qwen2.5:3b, full agent
([brands.json](../../src/evaluation/results/brands.json)):

| Measure | Without brand table | With brand table |
|---|---:|---:|
| Brand recognised | 0.000 | 1.000 |
| Expected ingredients found | 0.000 | 1.000 |
| Interaction found | 0.000 | 1.000 |
| Brand source cited | 0.000 | 1.000 |
| Disclaimer | 1.000 | 1.000 |

## Bonus: Tox21 toxicity model

12 LightGBM models on Morgan fingerprints, scaffold split 80/10/10 (`models/tox21/model.json`,
MLflow run `6d3e6790`, Registry `tujya-tox21` v2). Mean test ROC-AUC **0.7252**; per assay from
0.6468 (SR-ARE) to 0.8217 (NR-AhR).

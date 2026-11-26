# Module 2 results

Benchmark set: 96 hand-checked questions ([src/evaluation/questions.jsonl](../../src/evaluation/questions.jsonl),
49 in French, 47 in English); 85 of them have expected chunks
([src/evaluation/qrels.json](../../src/evaluation/qrels.json)) and are used for retrieval. Every
configuration searches like the agent: the ingredients found in the question are a metadata
filter, top 10 kept. Last column: MLflow run id. The served choice is in bold. In the first
two tables "Recall@k" is the share of questions with at least one relevant chunk in the top k
(Hit@k in the course); the [metrics of the course](#retrieval-metrics-of-the-course) below
separate Hit@k from Recall@k (share of the relevant chunks found).

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

Qdrant ordered the tied RRF scores of the hybrid search differently from one call to the next,
so the hybrid rows above come from one order of the ties. The order is now fixed (score, then
`chunk_id`): on the same 85 questions the served configuration gets MRR 0.839 (top 20, run
`104ad4c3`) and dense 0.833 (run `8cbb255f`), so the champion is unchanged (next section).

## Retrieval metrics of the course

`python -m src.evaluation.retrieval_evaluation run` (source:
[retrieval_metrics.json](../../src/evaluation/results/retrieval_metrics.json), rankings in
[retrieval_rankings.jsonl](../../src/evaluation/results/retrieval_rankings.jsonl), experiment
`tujya-rag-retrieval`). Served corpus v2 (`tujya_chunks_v2`, read only), the 85 questions with
qrels, binary relevance (the qrels mapped to the v2 chunks as in the grid), top 20 per question,
deterministic order of ties: two runs give the same rankings. Set metrics: Hit@k, Precision@k,
Recall@k, F1@k, R-Precision (= Precision@|relevant|); rank metrics: MRR, MAP, nDCG@k (linear
gains). Formulas and the worked example of the course:
[retrieval_metrics.py](../../src/evaluation/retrieval_metrics.py) and its tests.

| Configuration | Hit@5 | P@5 | R@5 | F1@5 | nDCG@5 | R@10 | R@20 | R-Precision | MRR | MAP | MLflow run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BM25 | 0.753 | 0.306 | 0.580 | 0.344 | 0.556 | 0.664 | 0.737 | 0.497 | 0.587 | 0.537 | `66804baf` |
| dense | 0.894 | 0.377 | 0.694 | 0.409 | 0.749 | 0.807 | 0.883 | 0.690 | 0.833 | 0.719 | `8cbb255f` |
| hybrid | 0.871 | 0.395 | 0.694 | 0.429 | 0.734 | 0.817 | 0.871 | 0.665 | 0.803 | 0.706 | `86f03f5f` |
| **hybrid + rewriting (served)** | **0.906** | **0.386** | **0.720** | **0.426** | **0.763** | **0.810** | **0.884** | **0.686** | **0.839** | **0.730** | **`104ad4c3`** |

The served configuration is the best on 6 of the 10 columns (Hit@5, R@5, nDCG@5, R@20, MRR,
MAP); plain hybrid has the best P@5, F1@5 and R@10, dense the best R-Precision. The qrels hold
every chunk of the expected section, so most questions have several relevant chunks and
Precision@5 stays low even when the answer is found (Hit@5 0.906).

**Recall@k curve of the served configuration and the choice of k.** The agent gives the LLM the
5 best search passages (`search_documents`, `k = 5` in
[src/agent/tools.py](../../src/agent/tools.py); at most `rag_top_k = 6` of them and 8 sources in
all in [rag_chain.py](../../src/rag/rag_chain.py)).

| k | Hit@k | Precision@k | Recall@k | nDCG@k |
|---:|---:|---:|---:|---:|
| 1 | 0.776 | 0.776 | 0.464 | 0.776 |
| 2 | 0.835 | 0.565 | 0.550 | 0.757 |
| 3 | 0.882 | 0.498 | 0.641 | 0.770 |
| 4 | 0.906 | 0.441 | 0.689 | 0.768 |
| **5 (sent to the LLM)** | **0.906** | **0.386** | **0.720** | **0.763** |
| 6 | 0.941 | 0.345 | 0.751 | 0.769 |
| 7 | 0.965 | 0.309 | 0.771 | 0.772 |
| 8 | 0.977 | 0.281 | 0.782 | 0.773 |
| 9 | 0.977 | 0.261 | 0.796 | 0.777 |
| 10 | 0.977 | 0.246 | 0.810 | 0.783 |
| 11 | 0.977 | 0.232 | 0.820 | 0.787 |
| 12 | 0.988 | 0.220 | 0.836 | 0.792 |
| 13 (flattens) | 0.988 | 0.209 | 0.851 | 0.797 |
| 14 | 0.988 | 0.199 | 0.858 | 0.801 |
| 15 | 0.988 | 0.190 | 0.864 | 0.803 |
| 16 | 0.988 | 0.179 | 0.866 | 0.803 |
| 17 | 0.988 | 0.172 | 0.871 | 0.806 |
| 18 | 0.988 | 0.164 | 0.872 | 0.806 |
| 19 | 0.988 | 0.159 | 0.877 | 0.808 |
| 20 | 0.988 | 0.153 | 0.884 | 0.810 |

The curve does not flatten at k = 5: it gains less than 0.01 Recall twice in a row only from
k = 13. Most of the gain comes early: from 5 to 8 passages, Hit goes from 0.906 to 0.977 and
Recall from 0.720 to 0.782, while Precision falls from 0.386 to 0.281. k = 5 is kept in this
step; **k = 8 is the proposed next experiment**, to be validated by a generation run (answer
quality, latency budget of 15 s, context of a 3B model), since more passages only help if the
small LLM still cites the right one.

**Reranker** (recorded rankings, not run again). The reranker grid ran on the corpus v1 only;
its top-10 rankings, logged by the grid runs, are scored with the same metrics next to the same
configuration without reranker:

| Corpus v1, recorded top 10 | Hit@1 | Hit@5 | R@5 | nDCG@5 | R@10 | MRR | MAP | ms / question | MLflow run (grid run) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| hybrid + rewriting | 0.753 | 0.906 | 0.722 | 0.756 | 0.819 | 0.824 | 0.699 | 394 | `070d7b4b` (`3ae434b6`) |
| hybrid + rewriting + MiniLM-L-12 reranker (20 candidates) | 0.894 | 0.965 | 0.762 | 0.834 | 0.848 | 0.922 | 0.766 | 11,251 | `458a2cee` (`254574f3`) |

The reranker gives the best ranking measured (MRR 0.922 against 0.824, nDCG@5 0.834 against
0.756) but takes 11,251 ms per question against 394, far above the 2 s budget: it stays off.
These v1 rankings predate the fixed order of ties.

## Document updates

`python -m src.rag.vector_store add | update | delete` on a trial copy of 300 points of
`tujya_chunks_v2` (`sample`), with the example documents of [examples/](examples/); transcript:
[document_updates_demo.txt](../../src/evaluation/results/document_updates_demo.txt).

- `add` embeds the new document (1 chunk) and it comes first in BM25, dense and hybrid search;
  a second `add` of the same document is refused ("use update").
- `update` with the same text embeds nothing (1 chunk unchanged); with version 2 it embeds the
  new text under the same chunk id (301 points), and search returns the new text.
- `delete` removes it (300 points); `add` on `tujya_chunks_v2` is refused without
  `--allow-served`.
- `tujya_chunks_v2` keeps 27302 points and the same `indexed_at` before and after.

## Embedding fine-tuning (not served)

bge-m3 fine-tuned on Kaggle (MLflow run `03c72c9c`, experiment `tujya-rag-finetuning`, Tesla T4,
151.7 min of training, final loss 0.0917). On its own 1,007 validation questions:

| Validation pairs | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| bge-m3 base | 0.655 | 0.930 | 0.967 | 0.784 | 0.830 |
| bge-m3 fine-tuned | 0.930 | 0.999 | 1.000 | 0.960 | 0.970 |

On the 85 benchmark questions, through the same sentence-transformers path
([grid/bge-m3-finetuned.json](../../src/evaluation/results/grid/bge-m3-finetuned.json)):

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

## Generation: reference-based metrics

`python -m src.evaluation.generation_evaluation` (source:
[generation_metrics.json](../../src/evaluation/results/generation_metrics.json), per question in
[generation_metrics_per_question.jsonl](../../src/evaluation/results/generation_metrics_per_question.jsonl),
experiment `tujya-rag-generation`). The recorded answers of the table above, never generated
again, cleaned as for RAGAS (citation marks and fixed safety texts removed), against the
reference of each question (the label or abstract text that answers it); answered questions
with a reference only. Tokens are normalised as in SQuAD (lower case, punctuation and articles
removed, accents folded). BLEU is sentence BLEU with smoothing: a long free-form answer rarely
shares a 4-gram with the reference, and without smoothing its BLEU is 0 whatever its 1- and
2-gram overlap, so most answers would tie at 0; the smoothing of Chen & Cherry (2014, `exp` in
sacrebleu) gives an empty order a small precision instead. Semantic similarity: cosine of the
bge-m3 embeddings of answer and reference, used instead of BERTScore because half the answers
are in French and the references in English (BERTScore matches tokens and needs PyTorch, which
the project venv does not have; bge-m3 is multilingual and already served). The metrics agree
with sacrebleu 2.6.0 and rouge-score 0.1.2 on the same tokens (85 pairs, largest difference
1e-16).

| Configuration | Answers | Language | EM | Token F1 | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | Similarity | MLflow run |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| **qwen2.5:3b, prompt v1, corpus v2 (served)** | 83 | all | 0.000 | 0.263 | 0.263 | 0.157 | 0.222 | 0.123 | 0.790 | `2a15d4e1` |
| | 42 | en | 0.000 | 0.434 | 0.434 | 0.298 | 0.367 | 0.234 | 0.828 | |
| | 41 | fr | 0.000 | 0.087 | 0.087 | 0.013 | 0.072 | 0.011 | 0.751 | |
| qwen2.5:3b, prompt v1 (corpus v1) | 84 | all | 0.000 | 0.240 | 0.240 | 0.135 | 0.198 | 0.102 | 0.773 | `e3ceaeb2` |
| | 42 | en | 0.000 | 0.382 | 0.382 | 0.252 | 0.318 | 0.186 | 0.799 | |
| | 42 | fr | 0.000 | 0.099 | 0.099 | 0.017 | 0.079 | 0.019 | 0.747 | |
| qwen2.5:3b, prompt v2 (corpus v1) | 84 | all | 0.000 | 0.245 | 0.245 | 0.143 | 0.209 | 0.107 | 0.770 | `3245d4a6` |
| | 42 | en | 0.000 | 0.401 | 0.401 | 0.267 | 0.341 | 0.196 | 0.808 | |
| | 42 | fr | 0.000 | 0.089 | 0.089 | 0.019 | 0.077 | 0.019 | 0.732 | |
| llama3.2:3b, prompt v1 (corpus v1) | 83 | all | 0.000 | 0.246 | 0.246 | 0.152 | 0.209 | 0.114 | 0.794 | `3d4d47a2` |
| | 42 | en | 0.000 | 0.392 | 0.392 | 0.272 | 0.332 | 0.200 | 0.820 | |
| | 41 | fr | 0.000 | 0.096 | 0.096 | 0.029 | 0.082 | 0.025 | 0.768 | |

Exact Match is 0 everywhere: it suits short extractive answers, not cited paragraphs. ROUGE-1
equals token F1 here (both are the F1 of the common words, counted with their multiplicity).
On the English answers the served champion has the best lexical scores; on the French answers
the lexical metrics mostly measure the language gap with the English references, while the
multilingual similarity stays between 0.73 and 0.77. The corpus-v1 rows compare the prompt and
the LLM on the same corpus; the corpus-v2 row is not compared with them.

## RAGAS on the passages and the judge check

`python -m src.evaluation.judge_check ragas` (source:
[ragas_context.json](../../src/evaluation/results/ragas_context.json), per question inside,
experiment `tujya-rag-generation`, run `1eb8100d`). The recorded answers keep no passages, so
the served setting (qwen2.5:3b, prompt v1, corpus v2) was generated again with its sources
(`python -m src.evaluation.benchmark_generation --corpus v2`): 96 questions, 94 answered, 17
answers identical to the recorded ones (temperature 0.1). Judge phi4-mini (RAGAS 0.4.4), the
first 4 passages given to the LLM, 900 characters each, as in the table of
[Generation: RAGAS, judge and MLflow GenAI](#generation-ragas-judge-and-mlflow-genai).

- **Context utilization** (higher is better): for each passage, does the judge find it useful to
  reach the answer; the mean precision of those verdicts weighted by rank. It is context
  precision with the answer in place of a reference.
- **Noise sensitivity** (lower is better): share of the claims of the answer that are wrong
  against the reference and come from a passage. It makes many more judge calls (about 100 s
  per answer here), so it runs on the first 20 answered questions only.
- **Faithfulness**, computed again on this generation for the judge check below.

| Metric | Mean | Answers scored | Not parsed | Time | s / answer |
|---|---:|---:|---:|---:|---:|
| Faithfulness | 0.9469 | 84 of 94 | 10 | 2259 s | 26.9 |
| Context utilization | 0.9512 | 94 of 94 | 0 | 765 s | 8.1 |
| Noise sensitivity (lower is better) | 0.1760 | 17 of 20 | 3 | 1748 s | 102.8 |

"Not parsed": the judge's output did not fit the RAGAS format even after its retries; those
answers have no score and are left out of the mean. The whole run took 4772 s (about 80 min) on the
development laptop. This faithfulness is not compared with the 0.9122 of the corpus-v2 row
above: it is another generation of the same setting.

### The judge against human labels

`python -m src.evaluation.judge_check sheet` drew 30 of these answers (seed 0) for a person to
label: faithful (1) when every claim is supported by the passages the judge read, else 0. The
judge's verdict was not on the sheet. It aims at half judged faithful and half not, but only 8
answers judged not faithful cite passages within the 4 the judge read, so the sheet holds 22
judged faithful and 8 not. `python -m src.evaluation.judge_check kappa` compares the labels
([human_labels.csv](../../src/evaluation/human_labels.csv)) with the judge (1 when
faithfulness = 1). Source: [judge_kappa.json](../../src/evaluation/results/judge_kappa.json),
experiment `tujya-rag-generation`, run `3d66bbed`.

| | Judge 1 | Judge 0 |
|---|---:|---:|
| **Human 1** | 14 | 2 |
| **Human 0** | 8 | 6 |

Observed agreement 0.667 (20 of 30), agreement expected by chance 0.516, **Cohen's kappa
0.312**: "fair" on the Landis & Koch scale. The labels find 16 answers faithful, the judge 22.
8 of the 10 disagreements are answers the judge scored 1 and the label marks unfaithful: the
judge is more often too lenient than too strict, so the faithfulness means above are likely
optimistic.

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

## Limitations

- **Size of the question set.** 96 questions, 85 with qrels and 84 with a reference: one
  question moves a mean such as Hit@k or MRR by up to 1/85 = 0.012, so differences of one or two
  hundredths between configurations are within noise (for that reason the grid asks for a
  paired bootstrap before rebuilding the index).
- **How the questions were built.** From hand-written templates per question type
  (contraindications, boxed warning, mechanism, adverse reactions, warnings, interactions,
  abstracts, brands, safety) over the corpus itself, each with an extractive reference taken
  from the source text. They are closer to the corpus than free questions of users.
- **Binary qrels at section level.** Every chunk of the expected section is relevant, every
  other chunk is not, even when it also answers: no graded relevance (nDCG uses gains of 1) and
  a low Precision@k by construction. The qrels were built on the corpus v1 and mapped to the v2
  chunks by shared text.
- **Recorded query rewrites.** The served configuration replays the rewrites recorded for the
  grid (`data/processed/rag/bench/queries.json`); a new LLM call could rewrite a question
  differently.
- **Reranker on the corpus v1 only**, scored from its recorded top-10 rankings, made before the
  order of tied scores was fixed.
- **Lexical generation metrics against English references.** Half the answers are in French:
  there, Exact Match, token F1, ROUGE and BLEU measure the language gap. The semantic similarity
  uses bge-m3, the embedding model of the retriever itself.
- **A local judge of 3.8 B parameters** (phi4-mini) scores RAGAS. Its errors are measured only
  by Cohen's kappa (0.312, "fair") on 30 answers labelled by one person, the author: 22 judged
  faithful and 8 not, so their mix differs from the whole set, and with 30 items the kappa is
  itself uncertain (the citation-judge control of `judge_control.json` used an earlier judge,
  gemma3:4b).
- **RAGAS on a new generation and on subsets.** The metrics that read the passages ran on a new
  generation of the served setting (17 of its 96 answers are identical to the recorded ones:
  temperature 0.1), not on the recorded answers; noise sensitivity on the first 20 answered questions only (17
  scored); 10 faithfulness and 3 noise-sensitivity outputs of the judge could not be parsed.
- **Corpus mostly in English**, from three sources (openFDA labels, PubMed abstracts, DDInter):
  French questions rely on the multilingual embeddings and on the query rewriting.
- **One machine.** Every time comes from the development laptop (Ollama on a 4 GB GPU); a
  reranker or a larger k could fit the budget on other hardware.
- **Embedding fine-tuning** evaluated on pairs taken from the same corpus as the benchmark; not
  served.

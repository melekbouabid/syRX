# RAG benchmark: retrieval and generation

Measured on 2026-10-28 on the laptop (RTX 2050 4 GB, Docker Desktop), with the real corpus
(8,661 chunks, docs/RAG.md), bge-m3 embeddings and qwen2.5:3b (docs/AGENT.md). Every run
is logged in MLflow, experiment **`tujya-rag`** (http://localhost:5000): parameters,
metrics, per-question results as artefacts. Decisions: D65 to D68.

## The benchmark set (`benchmark/`)

**88 questions, 44 in French and 44 in English**, about the 82 class ingredients:

| Type | Questions | Expected chunks (rule) |
|---|---:|---|
| `interaction` (A, B) | 24 | A's label chunks in interaction / warning / precaution / contraindication sections that mention B, B's that mention A, and the DDInter row of the pair |
| `contraindications` | 12 | the contraindications chunks of the ingredient's label |
| `boxed_warning` | 10 | the boxed-warning chunks |
| `mechanism` | 10 | the mechanism-of-action chunks |
| `adverse_reactions` | 10 | the adverse-reactions chunks |
| `warnings` | 8 | the warnings-and-precautions (or warnings) chunks |
| `abstract` | 9 | the chunks of one PubMed article (case reports, a systematic review, a meta-analysis) |
| `safety` | 5 | none: two dose requests, two "is it safe" questions, one timing question; scored on the answer only |

How it was built (`backend/scripts/bench_build.py`, spec `benchmark/spec.json` written by
`benchmark/make_spec.py`):

1. Candidates from the corpus: interaction pairs where one label mentions the other
   ingredient in an interaction or warning section and DDInter grades the pair (178 such
   pairs; 24 picked to cover many drug classes); ingredients that have the section asked
   about (e.g. 34 labels have a boxed warning; 10 picked); PubMed case reports and reviews
   with one clear finding (9 picked from a random sample of 50 titles).
2. Questions from fixed French and English templates (French INN names and articles, e.g.
   "Quelles sont les contre-indications de la metformine ?"), alternating languages.
3. Expected chunks by the rules above; reference answers extracted from the first expected
   chunk (sentences with a dose, a dosing instruction or a reporting phone line are skipped,
   label cross-references removed).
4. **Checked by hand**: every candidate was read. Fixed: the French grammar of the
   templates; one empty boxed warning (gemfibrozil: the section is only a heading) replaced
   by carbamazepine; one ingredient without an adverse-reactions section (fexofenadine)
   replaced; **14 interaction references rewritten by hand** from the label sentences of the
   pair, because the extracted sentence was off-topic (e.g. "propranolol does not affect the
   pharmacokinetics of verapamil" instead of the bradycardia warning) or quoted a dose
   ("do not exceed simvastatin 20 mg"); the 9 abstract references are paraphrases written
   after reading each abstract.

**Limits.** Expected chunks are found by rules on one label per ingredient: another chunk
can also answer the question and is then counted as a miss (the scores are a lower bound).
The questions are templated, so they are cleaner than real users' questions. The set was
built by the same person who built the system. The abstract questions have a single
expected article although other abstracts may be relevant. References are extractive
(label sentences) for 60 of the 83 questions; 23 were written by hand.

## Retrieval

`backend/scripts/bench_retrieval.py`: the 83 questions with expected chunks, top 10, six
configurations (three modes x two filter settings). **Recall@k** = share of questions with
at least one expected chunk in the top k (hit rate); **Coverage@10** = mean share of a
question's expected chunks in the top 10; **MRR** = mean of 1 / rank of the first expected
chunk (top 10). Filter `names` = the ingredients the agent finds in the question
(French or English names) used as a metadata filter, as the agent does; `none` = the raw
question only.

| Mode | Filter | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Coverage@10 | ms / query |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | none | 0.554 | 0.711 | 0.735 | 0.783 | 0.637 | 0.671 | 5 |
| dense (bge-m3) | none | 0.783 | 0.867 | 0.904 | 0.976 | 0.841 | 0.823 | 295 |
| hybrid (RRF) | none | 0.699 | 0.855 | 0.892 | 0.940 | 0.779 | 0.804 | 211 |
| BM25 | names | 0.566 | 0.747 | 0.783 | 0.819 | 0.660 | 0.710 | 26 |
| **dense (bge-m3)** | **names** | **0.795** | **0.880** | **0.928** | **0.976** | **0.852** | **0.829** | 180 |
| hybrid (RRF) | names | 0.711 | 0.892 | 0.904 | 0.928 | 0.796 | 0.791 | 193 |

By language (Recall@5 / MRR):

| Configuration | French | English |
|---|---|---|
| BM25, names | 0.585 / 0.449 | 0.976 / 0.865 |
| dense, names | 0.878 / 0.765 | 0.976 / 0.937 |
| hybrid, names | 0.805 / 0.643 | **1.000 / 0.944** |

By question type (Recall@5, filter `names`): dense is at 1.00 for contraindications,
mechanism and adverse reactions, 0.92 for interactions, 0.89 for abstracts, 0.90 for boxed
warnings, 0.75 for warnings; hybrid is better for boxed warnings (1.00) and interactions
(0.96) but much worse for abstracts (0.67).

**Choice (D65): dense retrieval with the ingredient filter** is the agent's default
(`SYNERGYRX_RAG_SEARCH_MODE=dense`): best Recall@1, Recall@5 and MRR overall. BM25 only
matches English words, so it fails on French questions (Recall@5 0.59) and drags the
hybrid down; on English questions alone, hybrid is slightly better (MRR 0.944 vs 0.937).
A language-aware choice (hybrid for English) or a translated BM25 query would be the next
step; it was not tuned here, to avoid fitting the system to its own test set. The
ingredient filter helps every mode a little (+0.01 to +0.05 Recall@5).

## Generation

`backend/scripts/bench_answers.py` runs the real agent (qwen2.5:3b, dense retrieval,
safety node) on the 88 questions; `bench_generation.py` scores the answers. Judge:
**gemma3:4b**, a different model from the generator (D67).

**Answer time** (warm models, RTX 2050): mean **4.7 s**, median
4.5 s, maximum 9.8 s per question (planner ~1 s, answer 1-7 s). The
first question after a start also loads the models (10-20 s).

**Tools.** On all 24 interaction questions the planner model itself called
`check_interaction`; it called `get_drug_info` on 59 questions; `search_documents` was
added by rule on 85 (the model rarely calls it on its own).

### Safety (all 88 answers)

| Check | Result |
|---|---:|
| Answers ending with the disclaimer | 100.0 % |
| Sentences with a numeric dose in the final answers | 0 |
| Sentences with dosing advice without a number (rule added after the run) | 4 |
| Sentences calling a combination "safe" / "sûr" | 0 |
| Timing advice without a citation | 0 |
| Dose questions refused (before any LLM call) | 100.0 % (2 / 2) |
| Answers where the safety node removed a sentence | 6 |
| Answers refused (dose question or no source) | 3 |

The two "is it safe?" questions got no "safe": one answered with DDInter's Major grade (but
as a raw list of its sources, a 3B-model failure), the other with "no known interaction in
this database". The timing question (levothyroxine + calcitriol) got DDInter's "Unknown"
grade, its uncited spacing sentence was removed. The safety counts are recomputed with
the final rules: **no answer gives a numeric dose**, but 4 answers kept a dose-adjustment
sentence without a number, paraphrased from the labels ("dose adjustment of warfarin may
be necessary", "carefully consider dose reduction"). The rules of that run only looked
for amounts; the rule now also removes dosing advice without numbers (found on the
containers just after the run; tests in `test_safety.py`), so these 4 would be removed
today.

### Citations (82 answered questions, 265 answer sentences)

| Check | Result |
|---|---:|
| Answer sentences with a citation [n] | 69.8 % |
| Answers with at least one citation | 92.7 % |
| **Cited sentences supported by the cited source** (judge) | **94.0 %** |
| ... French answers / English answers | 90.0 % / 96.5 % |
| Judge control: "supported" for a sentence paired with another question's source | 10.0 % (4 / 40) |

The control shows the judge discriminates (94 % vs 10 % "yes"), but it is lenient: a
sentence citing the right label passage while adding a wrong detail can still pass. Many
citations were added after generation: 93 by meaning (bge-m3) and 43 by word overlap (D68);
the model wrote the others itself. 30 % of the sentences remain uncited (typically
introductions and French paraphrases with no clear closest source).

### RAGAS 0.4.3 (stratified subset of 8 answered questions, one per question type, judge gemma3:4b)

| Metric | Mean | Scored |
|---|---:|---:|
| Faithfulness (claims of the answer supported by the contexts) | 0.979 | 6 / 8 |
| Answer relevancy (questions regenerated from the answer vs the question, bge-m3) | n/a | 0 / 8 |
| Context precision with reference (useful contexts ranked first) | 0.809 | 8 / 8 |

Read these as a comparison point for later runs, not as absolute values: a 4B judge with
half of its layers on the CPU, 8 questions (one per type; the subset took the first
question of each type, so 7 of the 8 are French), unparseable judge outputs counted as
missing ("Scored"). **Answer relevancy could not be computed**: on all 8 answers the
judge's generated questions were not parseable by RAGAS.

### What the answers show (read by hand)

- Good: contraindications, mechanisms, boxed warnings and DDInter grades are mostly
  right and cited; no dose and no "safe" reached an answer; dose questions are refused.
- Weak: the 3B model often stays close to the label text and adds off-topic sentences
  from the same passage (e.g. fluconazole + citalopram: tolvaptan and triazolam, which are
  in the fluconazole interaction section); it sometimes adds a detail that is not in the
  source (levothyroxine: "insuffisance rénale sévère"); once it answered a boxed-warning
  question with the "no known interaction" phrase; once it printed its sources instead of an
  answer. A larger model (7-8B) would need more than 4 GB of VRAM.

All of this is in MLflow (`generation-qwen2.5:3b-dense`, `citation-judge-control`) with
the per-question scores (`benchmark/generation_per_question.jsonl`, no source text).

## Reproduce

```bash
cd backend                                # backend venv; stack running (postgres, ollama, mlflow)
export SYNERGYRX_PHARMA_DATABASE_URL=postgresql+psycopg://tujya:tujya@localhost:5432/tujya
python scripts/bench_build.py             # benchmark/rag_questions.jsonl from the corpus
python scripts/bench_retrieval.py         # six retrieval runs -> MLflow, results_retrieval.json
python scripts/bench_answers.py           # the agent on the 88 questions -> data/rag/bench/
python scripts/bench_generation.py        # safety, citations, RAGAS -> MLflow, results_generation.json
```

RAGAS and the MLflow client are not in `requirements.txt` (benchmark only):
`pip install ragas==0.4.3 --no-deps` plus its dependencies except `scikit-network` (no
wheel for Python 3.14; it is only used by RAGAS' test-set generator), and `mlflow-skinny`.

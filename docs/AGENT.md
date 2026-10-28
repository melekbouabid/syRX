# Tujya agent (LangGraph) and POST /api/chat

The agent answers questions about the confirmed pills and their active ingredients, only
from retrieved documents and tool results, with numbered citations. Code:
`backend/app/rag/` (agent.py, tools.py, safety.py, names.py, runtime.py), route
`backend/app/routers/chat.py`. Corpus and retrieval: docs/RAG.md. Decisions D57-D59, D63.

## Graph

```
START -> understand --(dose question)--------------------------> safety -> END
             \--> tools -> retrieve -> answer -> safety -> END
```

| Node | What it does | LLM |
|---|---|---|
| `understand` | drug names in the question (French or English: "warfarine" -> warfarin, "paracétamol" -> acetaminophen) and the confirmed pills' ingredients; language (fr / en); dose question? | no |
| `tools` | the LLM chooses tool calls (Ollama tool calling with the three chat tools); the graph adds the calls a question always needs if the model forgot them (`check_interaction` for two or more drugs, `search_documents` always), runs every call, and records it (`chosen_by`: `llm` or `rule`) | yes (planner) |
| `retrieve` | numbered sources [1], [2] ...: DDInter results of `check_interaction` first, then label excerpts of `get_drug_info`, then the `search_documents` hits (dense with the ingredient filter by default, D65; retracted excluded); at most 8 | no |
| `answer` | the LLM answers from the numbered sources only, with the rules in the system prompt (French or English); streamed sentence by sentence through the safety filter | yes |
| `safety` | the rules below on the whole answer: rewrite, or refuse | no |

If the planner call fails (model down, bad JSON), the rule-based calls still run.

## Tools (`app/rag/tools.py`, shared with the MCP server)

| Tool | Arguments | Returns |
|---|---|---|
| `identify_pills` | `image_path` or `image_base64` | the `/api/scan` result: boxes, top-5, `needs_confirmation` (not given to the chat LLM: a chat question has no image; MCP clients have it) |
| `check_interaction` | `drugs` (names, French or English), `pills` (class_ids) | the `/api/interactions` result: every pair, DDInter severity, description, source; "aucune interaction connue dans cette base de données" when nothing is listed |
| `get_drug_info` | `drug`, optional `sections` | pill classes containing the ingredient, and the first chunk of each label section (boxed warning, indications, contraindications, warnings, interactions, mechanism) |
| `search_documents` | `query`, `ingredients`, `doc_types`, `sections`, `k`, `exclude_retracted` | hybrid search hits with title, section, URL, source, date, licence, text |

**Extension point.** A tool is one async method on `Toolbox` and one entry in
`TOOL_SPECS`; it is then available to the agent (`CHAT_TOOLS`) and the MCP server. Two
tools are planned, not implemented (docs/ROADMAP.md, Bonus): `predict_toxicity(smiles)`
(single-molecule toxicity, a model trained on Tox21 / ClinTox) and
`predict_synergy(drug_a, drug_b)` (cancer synergy from the DrugComb data already in the
pipeline).

## Safety rules (`app/rag/safety.py`, D63)

Applied to every answer, sentence by sentence:

1. **No dose.** A question asking for a dose ("quelle dose", "posologie", "how much") is
   refused before any LLM call. A sentence with an amount of medicine ("50 mg", "2
   comprimés", "deux fois par jour", "mg/kg") is removed; lab values ("eGFR 30 mL/min",
   "200 mg/dL") are not doses.
2. **Never "safe".** A sentence with "sûr", "sans danger", "sans risque", "safe" is removed;
   if no interaction was found, "Aucune interaction connue dans cette base de données." is
   written instead.
3. **No invented timing advice.** A sentence about spacing or meals ("2 heures
   d'intervalle", "avant le repas", "hours apart") is kept only if it cites a source.
4. **Citations.** `[n]` must point to a numbered source (others are removed). A sentence the
   model forgot to cite gets the source containing at least 60 % of its content words (it
   is then really in that source); an answer that still cites nothing gets "Sources
   consultées : [1] ...". No source at all: the answer says so and does not answer.
   Label cross-references ("[see Contraindications (4)]") are removed from the sources
   given to the model and from the answer.
5. **Disclaimer.** Every answer ends with "Information uniquement, consultez un pharmacien
   ou un médecin."

While streaming, a sentence that breaks rules 1-3 is held back, so it never reaches the
screen; the `final` event carries the checked answer, which the app displays.

## POST /api/chat

Signed-in users only. Body:

```json
{"question": "La warfarine interagit-elle avec le fluconazole ?", "pills": [9], "stream": false}
```

Response (`ChatResult`): `answer` (checked, ends with the disclaimer), `language`,
`ingredients`, `citations` (n, title, url, section, snippet, source_name, doc_type, date,
`cited`), `tool_calls` (name, args, chosen_by, ok, summary), `safety` (action `ok` /
`rewritten` / `refused`, violations, cited numbers), `model`, `timings` (per node and
total), `disclaimer`.

With `"stream": true` the answer is Server-Sent Events: `tool` (each call, as it happens),
`token` (answer text, one checked sentence at a time), then `final` (the `ChatResult`) or
`error`. nginx passes the stream through unbuffered (`X-Accel-Buffering: no`).

Errors: 401 not signed in; 503 when the index is empty ("run `python -m app.rag.index`") or
Ollama is not reachable.

Configuration (`SYNERGYRX_*`): `OLLAMA_URL` (compose: `http://ollama:11434`), `LLM_MODEL`
(`qwen2.5:3b`), `LLM_TEMPERATURE` (0.1), `LLM_NUM_CTX` (4096, D68), `LLM_MAX_TOKENS` (450), `EMBEDDING_MODEL`
(`bge-m3`), `RAG_TOP_K` (6).

From the command line: `python scripts/ask.py "question" [--pills 9 120]` (backend venv).

## Example (real models, 2026-10-28)

`What are the contraindications of metformin?` -> planner: `get_drug_info` and
`search_documents`; answer: "Metformin hydrochloride tablets are contraindicated in
patients with severe renal impairment (eGFR below 30 mL/min/1.73 m²), hypersensitivity to
metformin, and acute or chronic metabolic acidosis, including diabetic ketoacidosis with or
without coma [1]." + disclaimer; about 15 s in total on the RTX 2050 (planner ~7 s,
answer ~6 s).

## Tests (no Ollama, no real embeddings)

`backend/tests/test_chat.py`: a scripted chat model (`tests/rag_fakes.py`, tool calls and
streamed answers) and hashed fake embeddings over a six-chunk corpus. Covers sign-in, tool
choice + rule-added calls, unknown tools ignored, citations mapping, retracted excluded,
unsafe answers rewritten (dose, "sans danger", bad citation), dose question refused without
any LLM call, planner failure fallback, pills -> ingredients, SSE order (tool, token,
final) with unsafe sentences held back, 503 without index. `backend/tests/test_safety.py`:
the rules and the name matcher.

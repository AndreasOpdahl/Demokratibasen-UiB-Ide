# G-Eval results summary

**Judges:** `google/gemma-3-4b`
**Dimensions:** `completeness`, `correctness`, `faithfulness`
**Documents in subset:** 601 distinct `doc_id`.
**Datapoints:** 7212 pairwise judgments total (2404 rows per G-Eval table × 3 table(s), one per judge × dimension).
Equivalent to 2404 pair comparisons × 3 dimensions × 1 judge.

Bradley–Terry: `GPT4o-mini` labels gold summaries (JSONL `reference`). Exported θ use mean-centered β (geom. mean θ = 1); odds vs any other model match the fitted BT model.

---

## 1. Pairwise win rates

### Faithfulness

| model | google/gemma-3-4b_win_rate |
| --- | --- |
| GPT4o-mini | 0.602 |
| eurollm-9B-Instruct-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.419 |
| gemma-2-9b-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.528 |
| gemma-2b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.587 |
| gemma-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.410 |
| llama-2-13b-chat-norwegian-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.543 |
| llama-3.1-8b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.567 |
| nb-gpt-j-6b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.424 |
| normistral-11b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.545 |
| normistral-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.437 |
| normistral-7b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.498 |
| norskgpt-llama3-8b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.446 |

### Correctness

| model | google/gemma-3-4b_win_rate |
| --- | --- |
| GPT4o-mini | 0.694 |
| eurollm-9B-Instruct-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.388 |
| gemma-2-9b-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.653 |
| gemma-2b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.563 |
| gemma-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.352 |
| llama-2-13b-chat-norwegian-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.573 |
| llama-3.1-8b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.623 |
| nb-gpt-j-6b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.327 |
| normistral-11b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.554 |
| normistral-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.453 |
| normistral-7b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.426 |
| norskgpt-llama3-8b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.392 |

### Completeness

| model | google/gemma-3-4b_win_rate |
| --- | --- |
| GPT4o-mini | 0.662 |
| eurollm-9B-Instruct-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.348 |
| gemma-2-9b-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.596 |
| gemma-2b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.472 |
| gemma-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.350 |
| llama-2-13b-chat-norwegian-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.582 |
| llama-3.1-8b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.646 |
| nb-gpt-j-6b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.294 |
| normistral-11b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.669 |
| normistral-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.503 |
| normistral-7b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.477 |
| norskgpt-llama3-8b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.419 |

---

## 2. Bradley–Terry strengths (θ)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Faithfulness

| model | google/gemma-3-4b_theta |
| --- | --- |
| GPT4o-mini | 1.4890 |
| eurollm-9B-Instruct-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.7305 |
| gemma-2-9b-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.1162 |
| gemma-2b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.3874 |
| gemma-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.7128 |
| llama-2-13b-chat-norwegian-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.1782 |
| llama-3.1-8b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.2883 |
| nb-gpt-j-6b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.7479 |
| normistral-11b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.2139 |
| normistral-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.7770 |
| normistral-7b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.9818 |
| norskgpt-llama3-8b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.7923 |

### Correctness

| model | google/gemma-3-4b_theta |
| --- | --- |
| GPT4o-mini | 2.2514 |
| eurollm-9B-Instruct-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.6310 |
| gemma-2-9b-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.8341 |
| gemma-2b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.2919 |
| gemma-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.5580 |
| llama-2-13b-chat-norwegian-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.3214 |
| llama-3.1-8b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.6453 |
| nb-gpt-j-6b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.4991 |
| normistral-11b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.3226 |
| normistral-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.8025 |
| normistral-7b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.7497 |
| norskgpt-llama3-8b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.6166 |

### Completeness

| model | google/gemma-3-4b_theta |
| --- | --- |
| GPT4o-mini | 1.9771 |
| eurollm-9B-Instruct-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.5352 |
| gemma-2-9b-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.4650 |
| gemma-2b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.9031 |
| gemma-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.5459 |
| llama-2-13b-chat-norwegian-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.3512 |
| llama-3.1-8b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.8288 |
| nb-gpt-j-6b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.4285 |
| normistral-11b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 2.0695 |
| normistral-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.9744 |
| normistral-7b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.9076 |
| norskgpt-llama3-8b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.6753 |

---

## Export layout

- `json/` — pairwise rows and per-judge G-Eval tables (JSON)
- `tables/` — CSV summaries (win rates, Bradley–Terry)
- `reports/` — Markdown / LaTeX for reading and papers

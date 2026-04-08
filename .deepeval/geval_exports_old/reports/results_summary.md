# G-Eval results summary

**Judges:** `google/gemma-3-4b`
**Dimensions:** `completeness`, `correctness`, `faithfulness`
**Documents in subset:** 603 distinct `doc_id`.
**Datapoints:** 7236 pairwise judgments total (2412 rows per G-Eval table × 3 table(s), one per judge × dimension).
Equivalent to 2412 pair comparisons × 3 dimensions × 1 judge.

Bradley–Terry: `GPT4o-mini` labels gold summaries (JSONL `reference`). Exported θ use mean-centered β (geom. mean θ = 1); odds vs any other model match the fitted BT model.

---

## 1. Pairwise win rates

### Faithfulness

| model | google/gemma-3-4b_win_rate |
| --- | --- |
| GPT4o-mini | 0.603 |
| eurollm-9B-Instruct-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.418 |
| gemma-2-9b-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.530 |
| gemma-2b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.588 |
| gemma-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.410 |
| llama-2-13b-chat-norwegian-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.541 |
| llama-3.1-8b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.567 |
| nb-gpt-j-6b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.422 |
| normistral-11b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.545 |
| normistral-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.439 |
| normistral-7b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.499 |
| norskgpt-llama3-8b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.442 |

### Correctness

| model | google/gemma-3-4b_win_rate |
| --- | --- |
| GPT4o-mini | 0.695 |
| eurollm-9B-Instruct-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.387 |
| gemma-2-9b-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.652 |
| gemma-2b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.564 |
| gemma-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.352 |
| llama-2-13b-chat-norwegian-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.572 |
| llama-3.1-8b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.624 |
| nb-gpt-j-6b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.326 |
| normistral-11b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.554 |
| normistral-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.455 |
| normistral-7b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.427 |
| norskgpt-llama3-8b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.389 |

### Completeness

| model | google/gemma-3-4b_win_rate |
| --- | --- |
| GPT4o-mini | 0.663 |
| eurollm-9B-Instruct-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.347 |
| gemma-2-9b-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.596 |
| gemma-2b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.473 |
| gemma-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.350 |
| llama-2-13b-chat-norwegian-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.580 |
| llama-3.1-8b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.648 |
| nb-gpt-j-6b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.294 |
| normistral-11b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.669 |
| normistral-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.504 |
| normistral-7b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.478 |
| norskgpt-llama3-8b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.416 |

---

## 2. Bradley–Terry strengths (θ)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Faithfulness

| model | google/gemma-3-4b_theta |
| --- | --- |
| GPT4o-mini | 1.4958 |
| eurollm-9B-Instruct-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.7272 |
| gemma-2-9b-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.1257 |
| gemma-2b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.3910 |
| gemma-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.7130 |
| llama-2-13b-chat-norwegian-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.1716 |
| llama-3.1-8b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.2870 |
| nb-gpt-j-6b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.7454 |
| normistral-11b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.2147 |
| normistral-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.7818 |
| normistral-7b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.9855 |
| norskgpt-llama3-8b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.7828 |

### Correctness

| model | google/gemma-3-4b_theta |
| --- | --- |
| GPT4o-mini | 2.2608 |
| eurollm-9B-Instruct-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.6279 |
| gemma-2-9b-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.8305 |
| gemma-2b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.2950 |
| gemma-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.5580 |
| llama-2-13b-chat-norwegian-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.3137 |
| llama-3.1-8b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.6593 |
| nb-gpt-j-6b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.4973 |
| normistral-11b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.3231 |
| normistral-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.8078 |
| normistral-7b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.7528 |
| norskgpt-llama3-8b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.6107 |

### Completeness

| model | google/gemma-3-4b_theta |
| --- | --- |
| GPT4o-mini | 1.9843 |
| eurollm-9B-Instruct-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.5327 |
| gemma-2-9b-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.4642 |
| gemma-2b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.9069 |
| gemma-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.5458 |
| llama-2-13b-chat-norwegian-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.3439 |
| llama-3.1-8b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.8434 |
| nb-gpt-j-6b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.4266 |
| normistral-11b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 2.0713 |
| normistral-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.9806 |
| normistral-7b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.9107 |
| norskgpt-llama3-8b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.6677 |

---

## Export layout

- `json/` — pairwise rows and per-judge G-Eval tables (JSON)
- `tables/` — CSV summaries (win rates, Bradley–Terry)
- `reports/` — Markdown / LaTeX for reading and papers

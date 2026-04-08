# G-Eval results summary

**Judges:** `google/gemma-3-4b`, `gpt-3.5-turbo`, `google/gemini-2.5-flash-preview-05-20`
**Dimensions:** `faithfulness`, `correctness`, `completeness`, `newsworthiness`
**Documents in subset:** 51 distinct `doc_id`.
**Datapoints:** 4896 pairwise judgments total (408 rows per G-Eval table × 12 table(s), one per judge × dimension).
Equivalent to 408 pair comparisons × 4 dimensions × 3 judges.

Bradley–Terry: `GPT4o-mini` labels gold summaries (JSONL `reference`). Exported θ use mean-centered β (geom. mean θ = 1); odds vs any other model match the fitted BT model.

---

## 1. Pairwise win rates

### Faithfulness

| model | google/gemma-3-4b_win_rate | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate |
| --- | --- | --- | --- |
| GPT4o-mini | 0.650 | 0.742 | 0.850 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 0.526 | 0.704 | 0.579 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.481 | 0.396 | 0.273 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.603 | 0.626 | 0.540 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 0.511 | 0.517 | 0.574 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.463 | 0.329 | 0.299 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.478 | 0.376 | 0.446 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.436 | 0.430 | 0.419 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.326 | 0.399 | 0.539 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 0.583 | 0.577 | 0.571 |

### Correctness

| model | google/gemma-3-4b_win_rate | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate |
| --- | --- | --- | --- |
| GPT4o-mini | 0.633 | 0.758 | 0.525 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 0.539 | 0.632 | 0.493 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.455 | 0.383 | 0.481 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.644 | 0.644 | 0.500 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 0.551 | 0.494 | 0.506 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.409 | 0.335 | 0.494 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.376 | 0.452 | 0.516 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.471 | 0.395 | 0.488 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.365 | 0.404 | 0.478 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 0.615 | 0.590 | 0.526 |

### Completeness

| model | google/gemma-3-4b_win_rate | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate |
| --- | --- | --- | --- |
| GPT4o-mini | 0.525 | 0.733 | 0.767 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 0.513 | 0.658 | 0.605 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.545 | 0.422 | 0.357 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.678 | 0.701 | 0.644 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 0.540 | 0.517 | 0.472 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.384 | 0.274 | 0.293 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.414 | 0.478 | 0.570 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.442 | 0.413 | 0.407 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.376 | 0.371 | 0.472 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 0.609 | 0.506 | 0.474 |

### Newsworthiness

| model | google/gemma-3-4b_win_rate | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate |
| --- | --- | --- | --- |
| GPT4o-mini | 0.492 | 0.658 | 0.808 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 0.572 | 0.684 | 0.664 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.396 | 0.409 | 0.390 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.546 | 0.626 | 0.621 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 0.523 | 0.500 | 0.477 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.439 | 0.293 | 0.317 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.511 | 0.511 | 0.538 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.576 | 0.459 | 0.378 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.489 | 0.433 | 0.455 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 0.442 | 0.474 | 0.436 |

---

## 2. Bradley–Terry strengths (θ)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Faithfulness

| model | google/gemma-3-4b_theta | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta |
| --- | --- | --- | --- |
| GPT4o-mini | 1.7531 | 2.6109 | 5.0107 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 1.0536 | 2.1460 | 1.2774 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.9030 | 0.6409 | 0.3861 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 1.3866 | 1.4467 | 1.0188 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 0.9845 | 0.9233 | 1.1234 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.8387 | 0.4906 | 0.4133 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.9297 | 0.6181 | 0.8026 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.7970 | 0.7278 | 0.6681 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.5093 | 0.6646 | 1.1074 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 1.3879 | 1.4213 | 1.4404 |

### Correctness

| model | google/gemma-3-4b_theta | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta |
| --- | --- | --- | --- |
| GPT4o-mini | 1.6152 | 2.8401 | 1.1003 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 1.0997 | 1.5892 | 0.9708 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.8210 | 0.6034 | 0.9307 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 1.6212 | 1.5596 | 0.9969 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 1.1351 | 0.8525 | 1.0109 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.6909 | 0.5001 | 0.9681 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.6372 | 0.8368 | 1.0606 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.9055 | 0.6411 | 0.9568 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.5916 | 0.6793 | 0.9210 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 1.5799 | 1.5153 | 1.1031 |

### Completeness

| model | google/gemma-3-4b_theta | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta |
| --- | --- | --- | --- |
| GPT4o-mini | 1.0744 | 2.4682 | 2.9436 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 1.0114 | 1.7549 | 1.4490 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 1.1402 | 0.7127 | 0.5518 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 1.8861 | 1.9984 | 1.5916 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 1.1321 | 0.9454 | 0.7931 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.6387 | 0.3922 | 0.4274 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.7436 | 0.9380 | 1.3251 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.8271 | 0.7077 | 0.6779 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.6274 | 0.5940 | 0.8772 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 1.5338 | 1.1089 | 0.9994 |

### Newsworthiness

| model | google/gemma-3-4b_theta | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta |
| --- | --- | --- | --- |
| GPT4o-mini | 0.9428 | 1.7875 | 3.6755 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 1.2843 | 1.9734 | 1.8236 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.6868 | 0.6952 | 0.6251 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 1.2226 | 1.5173 | 1.3889 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 1.0646 | 0.9093 | 0.7897 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.8330 | 0.4436 | 0.4661 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 1.0504 | 1.0539 | 1.1444 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 1.3312 | 0.8413 | 0.5927 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.9625 | 0.7728 | 0.8147 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 0.8240 | 0.9725 | 0.8449 |

---

## Export layout

- `json/` — pairwise rows and per-judge G-Eval tables (JSON)
- `tables/` — CSV summaries (win rates, Bradley–Terry)
- `reports/` — Markdown / LaTeX for reading and papers

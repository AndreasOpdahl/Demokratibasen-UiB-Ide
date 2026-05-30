# G-Eval results summary

**Judges:** `gpt-5-mini`, `google/gemini-2.5-flash-preview-05-20`, `anthropic/claude-3-5-haiku-20241022`, `mistral-medium-latest`
**Dimensions:** `relevance`, `consistency`, `newsworthiness`, `hygiene`
**Documents in subset:** 150 distinct `doc_id`.
**Datapoints:** 19200 pairwise judgments total (1200 rows per G-Eval table × 16 table(s), one per judge × dimension).
Equivalent to 1200 pair comparisons × 4 dimensions × 4 judges.

Bradley–Terry: `GPT4o-mini` labels gold summaries (JSONL `reference`). Exported θ use mean-centered β (geom. mean θ = 1); odds vs any other model match the fitted BT model.

---

## 1. Pairwise win rates

### Relevance

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.857 | 0.860 | 0.912 | 0.892 |
| gemma-2-9b__checkpoint-2500-inputs-refs-preds-1000-examples | 0.701 | 0.727 | 0.730 | 0.723 |
| llama2-13b-chat-norwegian__checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.480 | 0.445 | 0.427 | 0.427 |
| nb-gpt-j-6b__checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.327 | 0.310 | 0.293 | 0.302 |
| norskgpt-llama3-8b__checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.226 | 0.249 | 0.207 | 0.223 |
| norwai-mistral-7b__checkpoint-9000-inputs-refs-preds-1000-examples | 0.329 | 0.341 | 0.309 | 0.328 |
| viking-13b__checkpoint-3500-inputs-refs-preds-1000-examples | 0.582 | 0.569 | 0.622 | 0.606 |

### Consistency

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.762 | 0.760 | 0.842 | 0.830 |
| gemma-2-9b__checkpoint-2500-inputs-refs-preds-1000-examples | 0.640 | 0.620 | 0.690 | 0.656 |
| llama2-13b-chat-norwegian__checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.478 | 0.472 | 0.475 | 0.485 |
| nb-gpt-j-6b__checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.296 | 0.245 | 0.258 | 0.281 |
| norskgpt-llama3-8b__checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.362 | 0.421 | 0.273 | 0.297 |
| norwai-mistral-7b__checkpoint-9000-inputs-refs-preds-1000-examples | 0.427 | 0.452 | 0.359 | 0.373 |
| viking-13b__checkpoint-3500-inputs-refs-preds-1000-examples | 0.536 | 0.531 | 0.605 | 0.577 |

### Newsworthiness

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.743 | 0.762 | 0.800 | 0.756 |
| gemma-2-9b__checkpoint-2500-inputs-refs-preds-1000-examples | 0.621 | 0.630 | 0.650 | 0.587 |
| llama2-13b-chat-norwegian__checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.554 | 0.506 | 0.483 | 0.512 |
| nb-gpt-j-6b__checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.449 | 0.453 | 0.431 | 0.466 |
| norskgpt-llama3-8b__checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.208 | 0.233 | 0.208 | 0.261 |
| norwai-mistral-7b__checkpoint-9000-inputs-refs-preds-1000-examples | 0.402 | 0.398 | 0.335 | 0.395 |
| viking-13b__checkpoint-3500-inputs-refs-preds-1000-examples | 0.523 | 0.519 | 0.593 | 0.523 |

### Hygiene

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.810 | 0.785 | 0.846 | 0.836 |
| gemma-2-9b__checkpoint-2500-inputs-refs-preds-1000-examples | 0.685 | 0.673 | 0.727 | 0.732 |
| llama2-13b-chat-norwegian__checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.410 | 0.405 | 0.405 | 0.452 |
| nb-gpt-j-6b__checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.389 | 0.363 | 0.319 | 0.337 |
| norskgpt-llama3-8b__checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.171 | 0.219 | 0.175 | 0.192 |
| norwai-mistral-7b__checkpoint-9000-inputs-refs-preds-1000-examples | 0.299 | 0.325 | 0.321 | 0.294 |
| viking-13b__checkpoint-3500-inputs-refs-preds-1000-examples | 0.738 | 0.730 | 0.707 | 0.657 |

---

## 2. Bradley–Terry strengths (θ)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Relevance

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 5.3846 | 5.5396 | 9.9440 | 7.6070 |
| gemma-2-9b__checkpoint-2500-inputs-refs-preds-1000-examples | 2.3269 | 2.6389 | 2.9110 | 2.6875 |
| llama2-13b-chat-norwegian__checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.8951 | 0.7659 | 0.6657 | 0.6892 |
| nb-gpt-j-6b__checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.4698 | 0.4333 | 0.3568 | 0.3941 |
| norskgpt-llama3-8b__checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.2930 | 0.3257 | 0.2293 | 0.2677 |
| norwai-mistral-7b__checkpoint-9000-inputs-refs-preds-1000-examples | 0.4731 | 0.4928 | 0.3827 | 0.4417 |
| viking-13b__checkpoint-3500-inputs-refs-preds-1000-examples | 1.3693 | 1.2842 | 1.6578 | 1.5230 |

### Consistency

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 2.8188 | 2.8081 | 4.8190 | 4.2687 |
| gemma-2-9b__checkpoint-2500-inputs-refs-preds-1000-examples | 1.6839 | 1.5615 | 2.1898 | 1.8401 |
| llama2-13b-chat-norwegian__checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.9164 | 0.9013 | 0.8856 | 0.9275 |
| nb-gpt-j-6b__checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.4536 | 0.3644 | 0.3492 | 0.4033 |
| norskgpt-llama3-8b__checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.5893 | 0.7442 | 0.3724 | 0.4314 |
| norwai-mistral-7b__checkpoint-9000-inputs-refs-preds-1000-examples | 0.7564 | 0.8344 | 0.5437 | 0.5919 |
| viking-13b__checkpoint-3500-inputs-refs-preds-1000-examples | 1.1371 | 1.1183 | 1.5134 | 1.3329 |

### Newsworthiness

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 2.6316 | 2.8498 | 3.6002 | 2.7309 |
| gemma-2-9b__checkpoint-2500-inputs-refs-preds-1000-examples | 1.5934 | 1.6358 | 1.8174 | 1.3799 |
| llama2-13b-chat-norwegian__checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 1.2342 | 1.0224 | 0.9336 | 1.0422 |
| nb-gpt-j-6b__checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.8329 | 0.8410 | 0.7643 | 0.8830 |
| norskgpt-llama3-8b__checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.3037 | 0.3424 | 0.2887 | 0.3933 |
| norwai-mistral-7b__checkpoint-9000-inputs-refs-preds-1000-examples | 0.6940 | 0.6787 | 0.5143 | 0.6741 |
| viking-13b__checkpoint-3500-inputs-refs-preds-1000-examples | 1.1005 | 1.0737 | 1.4423 | 1.0878 |

### Hygiene

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 4.3247 | 3.5385 | 5.7312 | 5.0498 |
| gemma-2-9b__checkpoint-2500-inputs-refs-preds-1000-examples | 2.2818 | 2.0695 | 2.9008 | 2.8448 |
| llama2-13b-chat-norwegian__checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.6728 | 0.6700 | 0.6363 | 0.8000 |
| nb-gpt-j-6b__checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.6142 | 0.5597 | 0.4257 | 0.4796 |
| norskgpt-llama3-8b__checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.2067 | 0.2867 | 0.1998 | 0.2332 |
| norwai-mistral-7b__checkpoint-9000-inputs-refs-preds-1000-examples | 0.4027 | 0.4719 | 0.4254 | 0.3911 |
| viking-13b__checkpoint-3500-inputs-refs-preds-1000-examples | 2.9456 | 2.6921 | 2.6135 | 1.9889 |

---

## Export layout

- `json/` — pairwise rows and per-judge G-Eval tables (JSON)
- `tables/` — CSV summaries (win rates, Bradley–Terry)
- `reports/` — Markdown / LaTeX for reading and papers

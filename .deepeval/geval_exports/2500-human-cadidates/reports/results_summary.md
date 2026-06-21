# G-Eval results summary

**Judges:** `gpt-5-mini`, `google/gemini-2.5-flash-preview-05-20`, `anthropic/claude-3-5-haiku-20241022`, `mistral-medium-latest`
**Dimensions:** `relevance`, `consistency`, `newsworthiness`, `hygiene`
**Documents in subset:** 500 distinct `doc_id`.
**Datapoints:** 48000 pairwise judgments total (3000 rows per G-Eval table × 16 table(s), one per judge × dimension).
Equivalent to 3000 pair comparisons × 4 dimensions × 4 judges.

Bradley–Terry: `GPT4o-mini` labels gold summaries (JSONL `reference`). Exported θ use mean-centered β (geom. mean θ = 1); odds vs any other model match the fitted BT model.

---

## 1. Pairwise win rates

### Relevance

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.654 | 0.687 | 0.703 | 0.674 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.462 | 0.458 | 0.441 | 0.447 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 0.548 | 0.494 | 0.498 | 0.501 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.336 | 0.361 | 0.358 | 0.378 |

### Consistency

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.658 | 0.671 | 0.676 | 0.681 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.502 | 0.504 | 0.470 | 0.473 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 0.426 | 0.411 | 0.478 | 0.461 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.414 | 0.414 | 0.376 | 0.385 |

### Newsworthiness

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.510 | 0.500 | 0.517 | 0.518 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.453 | 0.428 | 0.417 | 0.418 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 0.737 | 0.781 | 0.760 | 0.758 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.299 | 0.291 | 0.306 | 0.306 |

### Hygiene

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.575 | 0.562 | 0.620 | 0.630 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.428 | 0.432 | 0.466 | 0.461 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 0.561 | 0.558 | 0.523 | 0.541 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.437 | 0.448 | 0.392 | 0.368 |

---

## 2. Bradley–Terry strengths (θ)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Relevance

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.6265 | 1.8115 | 1.9203 | 1.7317 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.8901 | 0.8754 | 0.8284 | 0.8461 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 1.1613 | 0.9801 | 0.9891 | 0.9999 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.5948 | 0.6434 | 0.6356 | 0.6826 |

### Consistency

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.6382 | 1.7085 | 1.7430 | 1.7718 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 1.0053 | 1.0100 | 0.9085 | 0.9183 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 0.7939 | 0.7573 | 0.9322 | 0.8830 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.7648 | 0.7652 | 0.6775 | 0.6960 |

### Newsworthiness

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.0284 | 0.9892 | 1.0483 | 1.0520 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.8577 | 0.7810 | 0.7588 | 0.7600 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 2.2025 | 2.6462 | 2.4118 | 2.3919 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.5147 | 0.4892 | 0.5212 | 0.5229 |

### Hygiene

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.2553 | 1.2082 | 1.4457 | 1.4976 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.8024 | 0.8143 | 0.9003 | 0.8881 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 1.2038 | 1.1911 | 1.0724 | 1.1331 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.8248 | 0.8533 | 0.7164 | 0.6635 |

---

## Export layout

- `json/` — pairwise rows and per-judge G-Eval tables (JSON)
- `tables/` — CSV summaries (win rates, Bradley–Terry)
- `reports/` — Markdown / LaTeX for reading and papers

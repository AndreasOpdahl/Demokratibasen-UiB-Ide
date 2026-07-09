# G-Eval results summary

**Judges:** `gpt-5-mini`, `google/gemini-2.5-flash-preview-05-20`, `anthropic/claude-3-5-haiku-20241022`, `mistral-medium-latest`
**Dimensions:** `relevance`, `consistency`, `newsworthiness`, `hygiene`
**Documents in subset:** 2500 distinct `doc_id`.
**Datapoints:** 240000 pairwise judgments total (15000 rows per G-Eval table × 16 table(s), one per judge × dimension).
Equivalent to 15000 pair comparisons × 4 dimensions × 4 judges.

Bradley–Terry: `GPT4o-mini` labels gold summaries (JSONL `reference`). Exported θ use mean-centered β (geom. mean θ = 1); odds vs any other model match the fitted BT model.

---

## 1. Pairwise win rates

### Relevance

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.654 | 0.680 | 0.700 | 0.675 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.463 | 0.473 | 0.438 | 0.450 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 0.557 | 0.508 | 0.519 | 0.521 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.326 | 0.339 | 0.344 | 0.354 |

### Consistency

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.662 | 0.685 | 0.691 | 0.698 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.480 | 0.471 | 0.462 | 0.463 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 0.452 | 0.441 | 0.481 | 0.470 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.406 | 0.403 | 0.366 | 0.370 |

### Newsworthiness

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.525 | 0.511 | 0.519 | 0.519 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.453 | 0.426 | 0.425 | 0.430 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 0.726 | 0.764 | 0.750 | 0.743 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.296 | 0.299 | 0.306 | 0.309 |

### Hygiene

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.574 | 0.566 | 0.624 | 0.625 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.433 | 0.440 | 0.459 | 0.453 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 0.566 | 0.555 | 0.531 | 0.554 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.428 | 0.439 | 0.386 | 0.368 |

---

## 2. Bradley–Terry strengths (θ)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Relevance

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.6285 | 1.7735 | 1.8999 | 1.7408 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.8928 | 0.9172 | 0.8203 | 0.8560 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 1.1938 | 1.0238 | 1.0576 | 1.0650 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.5761 | 0.6004 | 0.6067 | 0.6302 |

### Consistency

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.6563 | 1.7918 | 1.8338 | 1.8790 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.9395 | 0.9109 | 0.8858 | 0.8867 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 0.8611 | 0.8303 | 0.9409 | 0.9063 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.7463 | 0.7379 | 0.6543 | 0.6623 |

### Newsworthiness

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.0823 | 1.0283 | 1.0577 | 1.0565 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.8573 | 0.7808 | 0.7807 | 0.7942 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 2.1164 | 2.4534 | 2.3134 | 2.2462 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.5093 | 0.5077 | 0.5235 | 0.5306 |

### Hygiene

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.2542 | 1.2205 | 1.4686 | 1.4735 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.8145 | 0.8338 | 0.8828 | 0.8651 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 1.2208 | 1.1807 | 1.0978 | 1.1821 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.8019 | 0.8322 | 0.7026 | 0.6636 |

---

## Export layout

- `json/` — pairwise rows and per-judge G-Eval tables (JSON)
- `tables/` — CSV summaries (win rates, Bradley–Terry)
- `reports/` — Markdown / LaTeX for reading and papers

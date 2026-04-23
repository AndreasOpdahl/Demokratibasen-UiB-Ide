# G-Eval results summary

**Judges:** `gpt-3.5-turbo`, `google/gemini-2.5-flash-preview-05-20`, `anthropic/claude-3-5-haiku-20241022`, `mistral-medium-latest`
**Dimensions:** `faithfulness`, `correctness`, `completeness`, `newsworthiness`, `hygiene`
**Documents in subset:** 25 distinct `doc_id`.
**Datapoints:** 2000 pairwise judgments total (100 rows per G-Eval table × 20 table(s), one per judge × dimension).
Equivalent to 100 pair comparisons × 5 dimensions × 4 judges.

Bradley–Terry: `GPT4o-mini` labels gold summaries (JSONL `reference`). Exported θ use mean-centered β (geom. mean θ = 1); odds vs any other model match the fitted BT model.

---

## 1. Pairwise win rates

### Faithfulness

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.964 | 0.869 | 0.976 | 1.000 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.355 | 0.461 | 0.408 | 0.434 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.341 | 0.341 | 0.415 | 0.232 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.361 | 0.417 | 0.292 | 0.458 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.442 | 0.395 | 0.372 | 0.360 |

### Correctness

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.964 | 0.881 | 0.976 | 1.000 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.368 | 0.487 | 0.382 | 0.461 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.317 | 0.366 | 0.329 | 0.244 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.375 | 0.319 | 0.347 | 0.431 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.442 | 0.419 | 0.430 | 0.349 |

### Completeness

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.940 | 0.952 | 1.000 | 0.964 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.382 | 0.408 | 0.382 | 0.434 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.256 | 0.256 | 0.341 | 0.305 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.431 | 0.444 | 0.319 | 0.389 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.465 | 0.419 | 0.419 | 0.384 |

### Newsworthiness

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.786 | 0.786 | 0.869 | 0.786 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.474 | 0.447 | 0.526 | 0.513 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.305 | 0.305 | 0.317 | 0.354 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.528 | 0.486 | 0.375 | 0.431 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.407 | 0.465 | 0.395 | 0.407 |

### Hygiene

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.000 | 0.929 | 1.000 | 1.000 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.263 | 0.316 | 0.263 | 0.276 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.232 | 0.427 | 0.451 | 0.366 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.458 | 0.361 | 0.292 | 0.319 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.512 | 0.430 | 0.442 | 0.488 |

---

## 2. Bradley–Terry strengths (θ)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Faithfulness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 14.1331 | 4.5968 | 19.5978 | 289042.0301 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.4762 | 0.8518 | 0.5382 | 0.0593 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.4454 | 0.5190 | 0.5407 | 0.0215 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.5167 | 0.7758 | 0.3655 | 0.0747 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.6456 | 0.6343 | 0.4797 | 0.0363 |

### Correctness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 14.1815 | 5.0123 | 19.7081 | 329088.2662 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.5036 | 0.9237 | 0.4926 | 0.0640 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.4010 | 0.5762 | 0.3913 | 0.0220 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.5495 | 0.5282 | 0.4588 | 0.0637 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.6354 | 0.7096 | 0.5738 | 0.0339 |

### Completeness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 9.4904 | 11.4964 | 1012763.6768 | 14.2377 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.5824 | 0.6236 | 0.0335 | 0.6538 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.3354 | 0.3119 | 0.0279 | 0.3640 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.7468 | 0.7795 | 0.0278 | 0.5959 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.7224 | 0.5737 | 0.0380 | 0.4951 |

### Newsworthiness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 2.9242 | 2.8941 | 4.6568 | 2.8587 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.9327 | 0.8503 | 1.0868 | 1.0554 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.4666 | 0.4864 | 0.4680 | 0.5714 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.1855 | 1.0077 | 0.6689 | 0.8372 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.6628 | 0.8291 | 0.6312 | 0.6929 |

### Hygiene

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 4399485.3620 | 7.8882 | 561856.4268 | 843063.2455 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.0143 | 0.4513 | 0.0230 | 0.0224 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.0122 | 0.6967 | 0.0539 | 0.0343 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.0354 | 0.5670 | 0.0273 | 0.0280 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.0367 | 0.7112 | 0.0527 | 0.0551 |

---

## Export layout

- `json/` — pairwise rows and per-judge G-Eval tables (JSON)
- `tables/` — CSV summaries (win rates, Bradley–Terry)
- `reports/` — Markdown / LaTeX for reading and papers

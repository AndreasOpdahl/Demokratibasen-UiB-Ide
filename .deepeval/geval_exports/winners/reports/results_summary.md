# G-Eval results summary

**Judges:** `gpt-5-mini`, `google/gemini-2.5-flash-preview-05-20`, `anthropic/claude-3-5-haiku-20241022`, `mistral-medium-latest`
**Dimensions:** `relevance`, `consistency`, `newsworthiness`, `hygiene`
**Documents in subset:** 200 distinct `doc_id`.
**Datapoints:** 25600 pairwise judgments total (1600 rows per G-Eval table × 16 table(s), one per judge × dimension).
Equivalent to 1600 pair comparisons × 4 dimensions × 4 judges.

Bradley–Terry: `GPT4o-mini` labels gold summaries (JSONL `reference`). Exported θ use mean-centered β (geom. mean θ = 1); odds vs any other model match the fitted BT model.

---

## 1. Pairwise win rates

### Relevance

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.844 | 0.861 | 0.905 | 0.884 |
| gemma-2-9b__checkpoint-2500-inputs-refs-preds-1000-examples | 0.704 | 0.726 | 0.725 | 0.725 |
| llama2-13b-chat-norwegian__checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.481 | 0.443 | 0.427 | 0.435 |
| nb-gpt-j-6b__checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.327 | 0.302 | 0.282 | 0.281 |
| norskgpt-llama3-8b__checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.228 | 0.246 | 0.209 | 0.220 |
| norwai-mistral-7b__checkpoint-9000-inputs-refs-preds-1000-examples | 0.308 | 0.320 | 0.309 | 0.325 |
| viking-13b__checkpoint-3500-inputs-refs-preds-1000-examples | 0.609 | 0.602 | 0.643 | 0.629 |

### Consistency

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.748 | 0.744 | 0.836 | 0.830 |
| gemma-2-9b__checkpoint-2500-inputs-refs-preds-1000-examples | 0.646 | 0.620 | 0.694 | 0.663 |
| llama2-13b-chat-norwegian__checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.485 | 0.478 | 0.481 | 0.486 |
| nb-gpt-j-6b__checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.303 | 0.261 | 0.267 | 0.292 |
| norskgpt-llama3-8b__checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.350 | 0.418 | 0.265 | 0.282 |
| norwai-mistral-7b__checkpoint-9000-inputs-refs-preds-1000-examples | 0.422 | 0.443 | 0.338 | 0.366 |
| viking-13b__checkpoint-3500-inputs-refs-preds-1000-examples | 0.546 | 0.535 | 0.619 | 0.581 |

### Newsworthiness

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.748 | 0.769 | 0.802 | 0.761 |
| gemma-2-9b__checkpoint-2500-inputs-refs-preds-1000-examples | 0.621 | 0.644 | 0.664 | 0.600 |
| llama2-13b-chat-norwegian__checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.551 | 0.499 | 0.480 | 0.498 |
| nb-gpt-j-6b__checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.441 | 0.419 | 0.408 | 0.433 |
| norskgpt-llama3-8b__checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.221 | 0.248 | 0.207 | 0.269 |
| norwai-mistral-7b__checkpoint-9000-inputs-refs-preds-1000-examples | 0.390 | 0.378 | 0.331 | 0.384 |
| viking-13b__checkpoint-3500-inputs-refs-preds-1000-examples | 0.527 | 0.543 | 0.608 | 0.555 |

### Hygiene

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.806 | 0.779 | 0.845 | 0.837 |
| gemma-2-9b__checkpoint-2500-inputs-refs-preds-1000-examples | 0.687 | 0.676 | 0.726 | 0.734 |
| llama2-13b-chat-norwegian__checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.416 | 0.406 | 0.414 | 0.445 |
| nb-gpt-j-6b__checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.380 | 0.368 | 0.323 | 0.344 |
| norskgpt-llama3-8b__checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.178 | 0.224 | 0.173 | 0.188 |
| norwai-mistral-7b__checkpoint-9000-inputs-refs-preds-1000-examples | 0.283 | 0.310 | 0.302 | 0.286 |
| viking-13b__checkpoint-3500-inputs-refs-preds-1000-examples | 0.751 | 0.737 | 0.718 | 0.666 |

---

## 2. Bradley–Terry strengths (θ)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Relevance

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 4.9754 | 5.7341 | 9.2661 | 7.2688 |
| gemma-2-9b__checkpoint-2500-inputs-refs-preds-1000-examples | 2.3709 | 2.6740 | 2.8547 | 2.7669 |
| llama2-13b-chat-norwegian__checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.9042 | 0.7558 | 0.6658 | 0.7141 |
| nb-gpt-j-6b__checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.4701 | 0.4096 | 0.3384 | 0.3540 |
| norskgpt-llama3-8b__checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.2957 | 0.3156 | 0.2327 | 0.2611 |
| norwai-mistral-7b__checkpoint-9000-inputs-refs-preds-1000-examples | 0.4321 | 0.4447 | 0.3857 | 0.4363 |
| viking-13b__checkpoint-3500-inputs-refs-preds-1000-examples | 1.5607 | 1.5013 | 1.8698 | 1.7266 |

### Consistency

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 2.6513 | 2.5911 | 4.6798 | 4.3004 |
| gemma-2-9b__checkpoint-2500-inputs-refs-preds-1000-examples | 1.7240 | 1.5638 | 2.2481 | 1.9047 |
| llama2-13b-chat-norwegian__checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.9391 | 0.9209 | 0.9069 | 0.9260 |
| nb-gpt-j-6b__checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.4676 | 0.3948 | 0.3614 | 0.4200 |
| norskgpt-llama3-8b__checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.5652 | 0.7371 | 0.3582 | 0.4026 |
| norwai-mistral-7b__checkpoint-9000-inputs-refs-preds-1000-examples | 0.7455 | 0.8092 | 0.4987 | 0.5736 |
| viking-13b__checkpoint-3500-inputs-refs-preds-1000-examples | 1.1822 | 1.1382 | 1.6230 | 1.3594 |

### Newsworthiness

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 2.6939 | 2.9665 | 3.6933 | 2.8158 |
| gemma-2-9b__checkpoint-2500-inputs-refs-preds-1000-examples | 1.5909 | 1.7336 | 1.9387 | 1.4441 |
| llama2-13b-chat-norwegian__checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 1.2160 | 0.9907 | 0.9207 | 0.9867 |
| nb-gpt-j-6b__checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.8040 | 0.7332 | 0.6907 | 0.7771 |
| norskgpt-llama3-8b__checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.3234 | 0.3647 | 0.2833 | 0.4058 |
| norwai-mistral-7b__checkpoint-9000-inputs-refs-preds-1000-examples | 0.6617 | 0.6258 | 0.5026 | 0.6463 |
| viking-13b__checkpoint-3500-inputs-refs-preds-1000-examples | 1.1153 | 1.1731 | 1.5427 | 1.2231 |

### Hygiene

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 4.2993 | 3.4462 | 5.7737 | 5.1548 |
| gemma-2-9b__checkpoint-2500-inputs-refs-preds-1000-examples | 2.3165 | 2.0978 | 2.9257 | 2.9102 |
| llama2-13b-chat-norwegian__checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.6824 | 0.6675 | 0.6539 | 0.7710 |
| nb-gpt-j-6b__checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.5811 | 0.5678 | 0.4269 | 0.4888 |
| norskgpt-llama3-8b__checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.2141 | 0.2946 | 0.1946 | 0.2254 |
| norwai-mistral-7b__checkpoint-9000-inputs-refs-preds-1000-examples | 0.3708 | 0.4421 | 0.3875 | 0.3739 |
| viking-13b__checkpoint-3500-inputs-refs-preds-1000-examples | 3.1887 | 2.8028 | 2.8126 | 2.0991 |

---

## Export layout

- `json/` — pairwise rows and per-judge G-Eval tables (JSON)
- `tables/` — CSV summaries (win rates, Bradley–Terry)
- `reports/` — Markdown / LaTeX for reading and papers

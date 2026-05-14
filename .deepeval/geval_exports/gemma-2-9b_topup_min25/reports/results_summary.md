# G-Eval results summary

**Judges:** `gpt-5-mini`, `google/gemini-2.5-flash-preview-05-20`, `anthropic/claude-3-5-haiku-20241022`, `mistral-medium-latest`
**Dimensions:** `relevance`, `consistency`, `newsworthiness`, `hygiene`
**Documents in subset:** 60 distinct `doc_id`.
**Datapoints:** 1392 pairwise judgments total (87 rows per G-Eval table × 16 table(s), one per judge × dimension).
Equivalent to 87 pair comparisons × 4 dimensions × 4 judges.

Bradley–Terry: `GPT4o-mini` labels gold summaries (JSONL `reference`). Exported θ use mean-centered β (geom. mean θ = 1); odds vs any other model match the fitted BT model.

---

## 1. Pairwise win rates

### Relevance

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | — | — | — | — |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.300 | 0.200 | 0.200 | 0.233 |
| checkpoint-1000-inputs-refs-preds-1000-examples | — | — | — | — |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.333 | 0.333 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.500 | 1.000 | 1.000 | 1.000 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.227 | 0.182 | 0.227 | 0.273 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.125 | 0.625 | 0.625 | 0.500 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 1.000 | 0.833 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.562 | 0.562 | 0.656 | 0.594 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.500 | 0.500 |
| checkpoint-2500-inputs-refs-preds-1000-examples | — | — | — | — |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.500 | 0.500 |
| checkpoint-3000-inputs-refs-preds-1000-examples | — | — | — | — |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.423 | 0.423 | 0.423 | 0.462 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.000 | 1.000 | 1.000 | 1.000 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.500 | 1.000 | 1.000 | 0.500 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.625 | 0.500 | 0.125 | 0.250 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 1.000 | 0.000 | 0.500 | 0.000 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.500 | 0.667 | 0.500 | 0.333 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.500 | 1.000 | 0.500 | 0.500 |
| checkpoint-500-inputs-refs-preds-1000-examples | 0.562 | 0.500 | 0.625 | 0.625 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.125 | 0.000 | 0.125 | 0.250 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.500 | 0.500 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.625 | 0.625 | 0.750 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.250 | 0.250 | 0.250 | 0.500 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.600 | 0.500 | 0.650 | 0.650 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.750 | 0.750 | 0.500 | 0.750 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.125 | 0.125 | 0.000 | 0.125 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.000 | 0.000 | 0.000 | 0.000 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | — | — | — | — |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.750 | 0.786 | 0.643 | 0.500 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.750 | 0.500 | 0.500 | 0.500 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.900 | 0.900 | 0.700 | 0.700 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | — | — | — | — |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.567 | 0.633 | 0.533 | 0.667 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.000 | 0.000 | 0.000 | 0.000 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.500 | 0.500 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 1.000 | 1.000 | 0.750 | 0.750 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.357 | 0.357 | 0.643 | 0.571 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.500 | 0.500 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.750 | 0.750 | 0.875 | 0.750 |

### Consistency

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | — | — | — | — |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.300 | 0.267 | 0.300 | 0.467 |
| checkpoint-1000-inputs-refs-preds-1000-examples | — | — | — | — |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.333 | 0.333 | 0.667 | 0.667 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.000 | 0.000 | 0.000 | 0.000 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.227 | 0.273 | 0.182 | 0.273 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.625 | 0.750 | 0.750 | 0.375 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.333 | 0.500 | 0.000 | 0.500 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.469 | 0.531 | 0.531 | 0.562 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.250 | 0.000 | 0.500 | 0.000 |
| checkpoint-2500-inputs-refs-preds-1000-examples | — | — | — | — |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.000 | 0.500 | 0.500 |
| checkpoint-3000-inputs-refs-preds-1000-examples | — | — | — | — |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.423 | 0.423 | 0.269 | 0.308 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.000 | 1.000 | 1.000 | 1.000 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 1.000 | 1.000 | 1.000 | 1.000 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.250 | 0.250 | 0.625 | 0.250 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.000 | 0.000 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.667 | 1.000 | 0.667 | 0.500 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 1.000 | 1.000 | 1.000 | 0.500 |
| checkpoint-500-inputs-refs-preds-1000-examples | 0.562 | 0.562 | 0.750 | 0.750 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.250 | 0.250 | 0.125 | 0.375 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.500 | 0.500 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.750 | 0.625 | 0.625 | 0.500 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.250 | 0.750 | 0.250 | 0.250 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.700 | 0.500 | 0.750 | 0.350 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.750 | 0.750 | 1.000 | 0.750 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.125 | 0.250 | 0.125 | 0.125 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.500 | 0.000 | 0.500 | 0.000 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | — | — | — | — |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.679 | 0.750 | 0.643 | 0.643 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.375 | 0.500 | 0.750 | 0.625 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.900 | 0.900 | 0.700 | 0.600 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | — | — | — | — |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.567 | 0.600 | 0.500 | 0.767 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.000 | 0.000 | 0.000 | 0.500 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.500 | 0.500 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 1.000 | 1.000 | 1.000 | 0.750 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.500 | 0.286 | 0.500 | 0.643 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.250 | 0.500 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.750 | 0.625 | 0.625 | 0.500 |

### Newsworthiness

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | — | — | — | — |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.333 | 0.400 | 0.367 | 0.300 |
| checkpoint-1000-inputs-refs-preds-1000-examples | — | — | — | — |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.333 | 0.167 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.000 | 1.000 | 1.000 | 1.000 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.318 | 0.318 | 0.318 | 0.364 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.125 | 0.250 | 0.375 | 0.125 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.833 | 0.500 | 0.833 | 0.833 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.562 | 0.562 | 0.688 | 0.562 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.500 | 0.500 |
| checkpoint-2500-inputs-refs-preds-1000-examples | — | — | — | — |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.500 | 0.500 |
| checkpoint-3000-inputs-refs-preds-1000-examples | — | — | — | — |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.462 | 0.346 | 0.462 | 0.423 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.000 | 1.000 | 1.000 | 1.000 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.000 | 0.500 | 0.500 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.750 | 0.750 | 0.625 | 0.625 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 1.000 | 0.000 | 0.000 | 1.000 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.333 | 0.667 | 0.500 | 0.333 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.000 | 0.500 | 0.000 | 0.000 |
| checkpoint-500-inputs-refs-preds-1000-examples | 0.625 | 0.375 | 0.625 | 0.625 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.375 | 0.500 | 0.250 | 0.375 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.500 | 0.500 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.625 | 0.500 | 0.500 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.500 | 0.500 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.550 | 0.650 | 0.500 | 0.650 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.750 | 0.750 | 0.750 | 0.750 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.125 | 0.125 | 0.125 | 0.125 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.000 | 1.000 | 0.500 | 1.000 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | — | — | — | — |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.643 | 0.714 | 0.643 | 0.679 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.000 | 0.375 | 0.750 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.900 | 0.900 | 0.700 | 0.800 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | — | — | — | — |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.633 | 0.667 | 0.600 | 0.533 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.000 | 0.000 | 0.000 | 0.500 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.500 | 0.500 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.000 | 0.250 | 0.500 | 0.250 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.429 | 0.429 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.500 | 0.500 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.375 | 0.250 | 0.375 | 0.375 |

### Hygiene

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | — | — | — | — |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.433 | 0.467 | 0.333 | 0.233 |
| checkpoint-1000-inputs-refs-preds-1000-examples | — | — | — | — |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.333 | 0.333 | 0.667 | 0.000 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.500 | 1.000 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.636 | 0.455 | 0.227 | 0.182 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.250 | 0.375 | 0.500 | 0.625 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.667 | 0.500 | 0.667 | 1.000 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.625 | 0.500 | 0.562 | 0.656 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.500 | 0.500 |
| checkpoint-2500-inputs-refs-preds-1000-examples | — | — | — | — |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.000 | 0.000 | 0.500 | 0.500 |
| checkpoint-3000-inputs-refs-preds-1000-examples | — | — | — | — |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.462 | 0.423 | 0.500 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.000 | 1.000 | 1.000 | 1.000 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 1.000 | 0.500 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.375 | 0.375 | 0.375 | 0.625 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.000 | 0.000 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.000 | 0.500 | 0.500 | 0.500 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.250 | 0.250 | 0.250 | 0.000 |
| checkpoint-500-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.625 | 0.500 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.625 | 0.500 | 0.500 | 0.375 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.500 | 0.000 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.625 | 0.625 | 0.500 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.500 | 0.000 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.200 | 0.300 | 0.500 | 0.500 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.500 | 0.750 | 1.000 | 0.500 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.375 | 0.375 | 0.125 | 0.125 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.000 | 0.500 | 0.500 | 0.000 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | — | — | — | — |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.643 | 0.607 | 0.607 | 0.643 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.625 | 0.375 | 0.375 | 0.500 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.600 | 0.600 | 0.800 | 0.900 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | — | — | — | — |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.467 | 0.667 | 0.600 | 0.767 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 1.000 | 1.000 | 0.000 | 0.000 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.250 | 0.250 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.750 | 0.750 | 0.750 | 1.000 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.714 | 0.643 | 0.571 | 0.500 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.500 | 0.750 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.500 | 0.375 | 0.500 | 0.500 |

---

## 2. Bradley–Terry strengths (θ)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Relevance

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.0000 | 0.9999 | 1.0000 | 1.0000 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.3403 | 0.0765 | 0.0823 | 0.0758 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 1.0000 | 0.9999 | 1.0000 | 1.0000 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.8607 | 0.3769 | 0.1595 | 0.1010 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.7631 | 47788055573195.3047 | 5241659221858382.0000 | 44249486653395.4844 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.2927 | 0.0769 | 0.1189 | 0.0896 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.1165 | 0.3087 | 1.0957 | 0.1525 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.4894 | 71156.7052 | 113037541813310.9375 | 253705514600.5793 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.8423 | 0.4086 | 0.9554 | 0.4519 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.8299 | 0.2677 | 0.1755 | 0.1049 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.0000 | 0.9999 | 1.0000 | 1.0000 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 1.7017 | 0.4122 | 0.0000 | 0.1395 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.0000 | 0.9999 | 1.0000 | 1.0000 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.3755 | 0.1371 | 0.1264 | 0.1474 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 101292572.7432 | 2322297780.7184 | 230117792.6579 | 811841004.6809 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.2927 | 32956958.6306 | 281563121.3121 | 0.0896 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.8264 | 0.4117 | 0.0538 | 0.0975 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 309401.8251 | 0.0000 | 1.0958 | 0.0000 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.3180 | 0.2488 | 0.3634 | 0.0928 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.4081 | 19684521887243.7852 | 66729685.5227 | 306428.1059 |
| checkpoint-500-inputs-refs-preds-1000-examples | 1.7630 | 0.6264 | 1.5422 | 0.8938 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.1331 | 0.0000 | 0.0652 | 0.0874 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.8422 | 0.8174 | 1.9111 | 2.2605 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.8423 | 0.8172 | 1.9109 | 2.2601 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.1251 | 0.0457 | 0.0421 | 0.1474 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.9072 | 0.1647 | 0.4432 | 0.3555 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.8520 | 71147.8032 | 16943615.3416 | 253702779757.3069 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.1684 | 0.0688 | 0.0000 | 0.0320 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 1.0000 | 0.9999 | 1.0000 | 1.0000 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 2.0238 | 0.9365 | 0.3742 | 0.1451 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 5.6391 | 0.1624 | 0.1626 | 0.1889 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 17.1778 | 2.4676 | 0.5825 | 0.6071 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 1.0000 | 0.9999 | 1.0000 | 1.0000 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.1318 | 0.4655 | 0.2396 | 0.3281 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 1.5135 | 0.6603 | 0.2995 | 0.2182 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 101344392795.0240 | 17761402662672.9453 | 1.9448 | 1.1863 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.2923 | 0.1515 | 0.6021 | 0.2018 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 1.0132 | 0.2769 | 0.3259 | 0.3415 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 2.7041 | 0.8349 | 0.9919 | 0.3533 |

### Consistency

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.1553 | 0.3856 | 0.1551 | 3.8995 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.1671 | 0.6513 | 0.8427 | 8.2846 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.1827 | 0.5519 | 0.1709 | 2.4695 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.7606 | 5.6742 | 1.2422 | 1.5729 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.3276 | 14014519.8595 | 0.0000 | 6.8569 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.3989 | 1.4647 | 0.8830 | 5.5131 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.0863 | 0.0000 | 0.2978 | 0.0000 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.4968 | 0.0000 | 0.5018 | 2.2890 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.1958 | 0.7515 | 0.0773 | 1.2048 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 512485.5946 | 126584521.4377 | 643682.2177 | 2885788.5047 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 803310.0336 | 13726072.9132 | 264956930.5933 | 278175185.9945 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.1444 | 0.3780 | 0.8183 | 2.4602 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.7604 | 5.6752 | 0.0000 | 0.0000 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.5683 | 529893537788.6418 | 1.0078 | 5.6814 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 4762150.5666 | 4833321813377.6953 | 6564905.2341 | 5.1713 |
| checkpoint-500-inputs-refs-preds-1000-examples | 0.5845 | 1.5481 | 3.8454 | 17.5119 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.1500 | 1.7453 | 0.1790 | 4.3418 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.9937 | 2.9286 | 1.7658 | 5.5097 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 1.9940 | 2.9294 | 1.7660 | 5.5120 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.0653 | 2.2547 | 0.0258 | 0.4016 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.7790 | 0.7967 | 1.1452 | 2.2123 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.6908 | 14017758.6054 | 8345467.6814 | 16.5864 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.0639 | 0.0000 | 0.0959 | 0.6316 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.1958 | 0.0000 | 0.0773 | 0.0000 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.5173 | 2.2051 | 0.5717 | 7.4491 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.1953 | 1.6704 | 1.6463 | 11.2092 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 3.8625 | 7.4668 | 2.6239 | 8.2942 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.3974 | 1.5689 | 0.4150 | 13.0198 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.0000 | 0.0000 | 0.0000 | 2.2117 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.4534 | 1.8599 | 0.4871 | 9.8475 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 49086048.3036 | 146487776794.7698 | 1754541125576449.2500 | 25.1465 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.3464 | 0.4812 | 0.6232 | 7.6669 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.5565 | 1.1179 | 0.2154 | 5.3674 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.6911 | 1.8235 | 0.5338 | 3.1413 |

### Newsworthiness

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.4626 | 3.3983 | 0.4559 | 0.0361 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.9731 | 6.9067 | 0.5856 | 0.0155 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1837355419519.9434 | 6635019.2707 | 134357897022.5976 | 41884564516.5961 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.2969 | 1.9586 | 0.3840 | 0.0551 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.2164 | 0.0000 | 0.1974 | 0.0134 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 51712663.6363 | 5.6268 | 1984573.2850 | 5086637.9252 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.9785 | 7.2338 | 1.9990 | 0.0827 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.7756 | 6.4788 | 0.7264 | 0.0636 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 1.3325 | 6.6562 | 0.5192 | 0.0972 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.7241 | 3.6704 | 0.7525 | 0.0547 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1108134491.2347 | 10151275.4455 | 137229869.0785 | 9546438.4147 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.2968 | 0.0000 | 0.3841 | 0.0551 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 3.5493 | 21.1855 | 1.2523 | 0.1155 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 161697.8527 | 0.0000 | 0.0000 | 5364.2064 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.4274 | 9.3022 | 0.6242 | 0.0329 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.0000 | 4.3736 | 0.0000 | 0.0000 |
| checkpoint-500-inputs-refs-preds-1000-examples | 2.2422 | 3.1989 | 2.5946 | 0.2085 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.7618 | 2.8098 | 0.2259 | 0.0863 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.9789 | 14.4695 | 1.9990 | 0.0828 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.9787 | 14.4670 | 1.9983 | 0.0828 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.7242 | 3.6714 | 0.7524 | 0.0547 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.7904 | 9.7502 | 0.6366 | 0.1889 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 51714221.8487 | 12.2947 | 1984714.8016 | 5084550.3475 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.1608 | 1.1322 | 0.1581 | 0.0132 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.0000 | 10149946.0739 | 0.7523 | 9546087.7083 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.3002 | 12.3529 | 1.1577 | 0.1120 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 1.4621 | 0.0000 | 0.3724 | 0.4726 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 11.0522 | 39.1588 | 1.7036 | 0.7153 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.6747 | 12.3118 | 1.1246 | 0.0986 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.0000 | 0.0000 | 0.0000 | 0.1888 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 1.4756 | 12.3309 | 1.1414 | 0.1051 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.0000 | 0.8220 | 0.9982 | 0.0319 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.8421 | 4.5936 | 0.5590 | 0.0420 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 1.1505 | 10.9559 | 0.8463 | 0.1364 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.7122 | 2.0179 | 0.4072 | 0.0525 |

### Hygiene

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.9999 | 1.0000 | 0.9999 | 1.0000 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 1.9641 | 0.6953 | 0.1997 | 1.0389 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.9999 | 1.0000 | 0.9999 | 1.0000 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 1.4413 | 0.4686 | 0.8397 | 0.0000 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 3.3223 | 0.9211 | 0.9867 | 136273776050496.2188 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 4.2772 | 0.7290 | 0.1180 | 0.8952 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.5164 | 0.5929 | 0.2353 | 10.4889 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 3.9033 | 0.7359 | 1102519.3692 | 61990278605782544.0000 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 2.4684 | 0.7226 | 0.5444 | 8.5379 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 2.9027 | 0.9445 | 0.2905 | 2.3953 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.9999 | 1.0000 | 0.9999 | 1.0000 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.0000 | 0.0000 | 0.3004 | 8.5456 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.9999 | 1.0000 | 0.9999 | 1.0000 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 2.3537 | 0.9161 | 0.3137 | 2.1443 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 266321109.4942 | 9079959.3445 | 11785990.0270 | 1883761993.0134 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 4.2768 | 0.7283 | 168751.7392 | 0.8949 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.4717 | 0.7156 | 0.2508 | 20.8894 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.5163 | 0.5931 | 0.0000 | 0.0000 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.0000 | 1.0016 | 0.6185 | 7.7133 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.8962 | 0.2385 | 0.1998 | 0.0000 |
| checkpoint-500-inputs-refs-preds-1000-examples | 3.3226 | 0.9204 | 0.9868 | 14.3887 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 3.2858 | 0.9064 | 0.8713 | 8.5172 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 2.4670 | 1.4455 | 1.0890 | 0.0000 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 2.4678 | 1.4451 | 1.0888 | 4.2690 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 2.3539 | 0.9161 | 0.3137 | 0.0000 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.6836 | 0.3840 | 0.2230 | 3.4311 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 3.5813 | 2.4596 | 781742088426.0315 | 3093111691.8347 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.5804 | 0.1708 | 0.0615 | 1.0792 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.0000 | 0.9151 | 0.3137 | 0.0000 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.9999 | 1.0000 | 0.9999 | 1.0000 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 4.2885 | 1.2812 | 0.4225 | 5.5248 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 3.7388 | 0.5178 | 0.3139 | 11.8204 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 2.0584 | 0.5974 | 1.4673 | 67.6306 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.9999 | 1.0000 | 0.9999 | 1.0000 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.8647 | 1.5925 | 0.3869 | 21.8647 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 2705677.3781 | 300756.7371 | 0.0000 | 0.0000 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 2.8278 | 1.4302 | 0.1347 | 3.2518 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 11.3556 | 2.4670 | 1.3614 | 39304627096539.3672 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 4.1786 | 1.5415 | 0.4999 | 6.3067 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 1.1289 | 0.7815 | 0.2938 | 32.2575 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 2.6476 | 0.5572 | 0.4102 | 6.7872 |

---

## Export layout

- `json/` — pairwise rows and per-judge G-Eval tables (JSON)
- `tables/` — CSV summaries (win rates, Bradley–Terry)
- `reports/` — Markdown / LaTeX for reading and papers

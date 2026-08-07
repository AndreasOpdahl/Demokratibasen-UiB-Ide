# G-Eval results summary

**Judges:** `gpt-5-mini`, `google/gemini-2.5-flash-preview-05-20`, `anthropic/claude-3-5-haiku-20241022`, `mistral-medium-latest`
**Dimensions:** `relevance`, `consistency`, `newsworthiness`, `hygiene`
**Documents in subset:** 65 distinct `doc_id`.
**Datapoints:** 8320 pairwise judgments total (520 rows per G-Eval table × 16 table(s), one per judge × dimension).
Equivalent to 520 pair comparisons × 4 dimensions × 4 judges.

Bradley–Terry: `GPT4o-mini` labels gold summaries (JSONL `reference`). Exported θ use mean-centered β (geom. mean θ = 1); odds vs any other model match the fitted BT model.

---

## 1. Pairwise win rates

### Relevance

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.846 | 0.923 | 0.885 | 0.846 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.680 | 0.580 | 0.660 | 0.680 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.500 | 0.440 | 0.620 | 0.440 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.540 | 0.340 | 0.260 | 0.400 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.538 | 0.558 | 0.442 | 0.423 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.580 | 0.600 | 0.500 | 0.420 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.558 | 0.423 | 0.481 | 0.442 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.520 | 0.540 | 0.480 | 0.560 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.580 | 0.560 | 0.540 | 0.460 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.540 | 0.560 | 0.480 | 0.580 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.620 | 0.620 | 0.560 | 0.620 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.538 | 0.538 | 0.481 | 0.442 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.480 | 0.660 | 0.480 | 0.740 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.420 | 0.580 | 0.440 | 0.540 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.580 | 0.600 | 0.660 | 0.640 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.462 | 0.385 | 0.442 | 0.538 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.600 | 0.580 | 0.600 | 0.560 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.360 | 0.380 | 0.320 | 0.360 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.400 | 0.500 | 0.500 | 0.440 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 0.540 | 0.520 | 0.500 | 0.540 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.308 | 0.346 | 0.346 | 0.346 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.340 | 0.440 | 0.500 | 0.360 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.385 | 0.404 | 0.538 | 0.442 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.480 | 0.500 | 0.500 | 0.520 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.460 | 0.480 | 0.580 | 0.620 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.560 | 0.560 | 0.460 | 0.540 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.620 | 0.560 | 0.580 | 0.580 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.320 | 0.360 | 0.520 | 0.480 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.560 | 0.600 | 0.600 | 0.500 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.442 | 0.442 | 0.481 | 0.481 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.538 | 0.519 | 0.596 | 0.519 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.365 | 0.346 | 0.423 | 0.500 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.558 | 0.500 | 0.442 | 0.500 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.385 | 0.404 | 0.385 | 0.404 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.519 | 0.404 | 0.423 | 0.423 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.520 | 0.460 | 0.460 | 0.360 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.520 | 0.580 | 0.520 | 0.500 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.560 | 0.440 | 0.480 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.442 | 0.481 | 0.500 | 0.500 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.423 | 0.327 | 0.385 | 0.385 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.380 | 0.360 | 0.500 | 0.400 |

### Consistency

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.808 | 0.865 | 0.904 | 0.923 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.340 | 0.620 | 0.540 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.460 | 0.460 | 0.620 | 0.560 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.440 | 0.380 | 0.260 | 0.360 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.538 | 0.442 | 0.615 | 0.519 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.580 | 0.580 | 0.480 | 0.480 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.615 | 0.538 | 0.596 | 0.538 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.480 | 0.440 | 0.400 | 0.480 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.680 | 0.520 | 0.580 | 0.600 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.360 | 0.480 | 0.460 | 0.500 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.640 | 0.620 | 0.460 | 0.580 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.538 | 0.500 | 0.519 | 0.481 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.600 | 0.700 | 0.580 | 0.600 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.420 | 0.380 | 0.460 | 0.420 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.620 | 0.620 | 0.640 | 0.680 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.462 | 0.404 | 0.538 | 0.615 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.620 | 0.620 | 0.600 | 0.540 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.380 | 0.420 | 0.420 | 0.300 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.540 | 0.560 | 0.620 | 0.440 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 0.560 | 0.560 | 0.540 | 0.600 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.308 | 0.327 | 0.288 | 0.269 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.400 | 0.400 | 0.480 | 0.360 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.558 | 0.558 | 0.538 | 0.462 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.420 | 0.440 | 0.480 | 0.420 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.480 | 0.380 | 0.500 | 0.500 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.440 | 0.460 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.560 | 0.560 | 0.560 | 0.560 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.320 | 0.400 | 0.380 | 0.420 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.580 | 0.640 | 0.560 | 0.620 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.481 | 0.500 | 0.442 | 0.442 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.462 | 0.538 | 0.462 | 0.462 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.423 | 0.365 | 0.404 | 0.385 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.462 | 0.538 | 0.538 | 0.481 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.404 | 0.500 | 0.423 | 0.462 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.577 | 0.519 | 0.442 | 0.423 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.340 | 0.400 | 0.300 | 0.480 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.620 | 0.500 | 0.520 | 0.660 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.460 | 0.640 | 0.560 | 0.480 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.442 | 0.538 | 0.423 | 0.519 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.442 | 0.385 | 0.404 | 0.327 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.420 | 0.440 | 0.440 | 0.560 |

### Newsworthiness

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.885 | 0.808 | 0.865 | 0.731 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.640 | 0.600 | 0.600 | 0.500 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.580 | 0.600 | 0.500 | 0.500 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.420 | 0.440 | 0.340 | 0.480 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.481 | 0.519 | 0.481 | 0.462 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.480 | 0.540 | 0.460 | 0.340 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.538 | 0.462 | 0.385 | 0.442 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.380 | 0.560 | 0.480 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.520 | 0.560 | 0.420 | 0.360 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.520 | 0.500 | 0.460 | 0.380 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.640 | 0.540 | 0.560 | 0.540 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.462 | 0.442 | 0.346 | 0.365 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.560 | 0.500 | 0.580 | 0.600 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.480 | 0.580 | 0.520 | 0.420 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.520 | 0.400 | 0.400 | 0.460 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.519 | 0.442 | 0.500 | 0.481 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.520 | 0.560 | 0.560 | 0.600 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.400 | 0.540 | 0.440 | 0.520 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.400 | 0.340 | 0.420 | 0.460 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 0.620 | 0.540 | 0.600 | 0.560 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.385 | 0.365 | 0.423 | 0.423 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.440 | 0.500 | 0.560 | 0.440 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.404 | 0.538 | 0.519 | 0.519 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.580 | 0.540 | 0.500 | 0.660 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.360 | 0.380 | 0.540 | 0.440 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.560 | 0.700 | 0.540 | 0.560 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.460 | 0.620 | 0.620 | 0.540 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.560 | 0.440 | 0.500 | 0.520 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.580 | 0.560 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.327 | 0.385 | 0.327 | 0.365 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.615 | 0.423 | 0.615 | 0.577 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.365 | 0.365 | 0.423 | 0.500 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.538 | 0.442 | 0.538 | 0.577 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.404 | 0.442 | 0.404 | 0.538 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.442 | 0.481 | 0.500 | 0.327 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.440 | 0.600 | 0.520 | 0.580 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.600 | 0.580 | 0.500 | 0.580 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.520 | 0.560 | 0.480 | 0.500 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.404 | 0.500 | 0.538 | 0.577 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.423 | 0.423 | 0.462 | 0.481 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.500 | 0.440 | 0.420 | 0.560 |

### Hygiene

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.635 | 0.692 | 0.827 | 0.731 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.460 | 0.400 | 0.600 | 0.580 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.480 | 0.460 | 0.540 | 0.580 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.380 | 0.300 | 0.240 | 0.460 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.596 | 0.596 | 0.462 | 0.481 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.520 | 0.460 | 0.440 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.538 | 0.519 | 0.365 | 0.558 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.420 | 0.380 | 0.600 | 0.400 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.480 | 0.480 | 0.520 | 0.560 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.560 | 0.520 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.580 | 0.540 | 0.560 | 0.640 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.519 | 0.558 | 0.538 | 0.519 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.480 | 0.500 | 0.420 | 0.500 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.480 | 0.420 | 0.400 | 0.340 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.500 | 0.540 | 0.660 | 0.680 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.423 | 0.442 | 0.462 | 0.404 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.460 | 0.600 | 0.620 | 0.540 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.460 | 0.540 | 0.460 | 0.440 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.500 | 0.480 | 0.500 | 0.540 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 0.660 | 0.520 | 0.580 | 0.520 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.481 | 0.423 | 0.442 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.400 | 0.400 | 0.340 | 0.460 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.481 | 0.596 | 0.462 | 0.519 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.420 | 0.460 | 0.580 | 0.440 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.500 | 0.540 | 0.420 | 0.520 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.600 | 0.380 | 0.480 | 0.500 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.580 | 0.520 | 0.620 | 0.580 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.480 | 0.460 | 0.500 | 0.440 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.460 | 0.440 | 0.560 | 0.480 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.423 | 0.423 | 0.385 | 0.327 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.577 | 0.519 | 0.558 | 0.558 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.442 | 0.519 | 0.538 | 0.442 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.481 | 0.519 | 0.519 | 0.423 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.519 | 0.481 | 0.365 | 0.500 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.519 | 0.615 | 0.423 | 0.423 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.520 | 0.480 | 0.340 | 0.500 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.520 | 0.520 | 0.520 | 0.540 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.600 | 0.460 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.558 | 0.558 | 0.635 | 0.577 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.538 | 0.423 | 0.385 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.460 | 0.540 | 0.440 | 0.560 |

---

## 2. Bradley–Terry strengths (θ)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Relevance

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 5.1932 | 11.3611 | 7.6434 | 5.5892 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 2.0586 | 1.3228 | 1.8514 | 1.9718 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.9458 | 0.7755 | 1.7452 | 0.8258 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 1.1995 | 0.5505 | 0.3565 | 0.7188 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.1053 | 1.1793 | 0.7681 | 0.6799 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 1.3301 | 1.3892 | 0.9232 | 0.6806 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.1759 | 0.7045 | 0.9062 | 0.7477 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 1.0559 | 1.0840 | 0.8911 | 1.2380 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.2679 | 1.1276 | 1.0587 | 0.7497 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 1.3624 | 1.4740 | 1.0300 | 1.5143 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.6034 | 1.6393 | 1.1986 | 1.5495 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 1.1733 | 1.1386 | 0.9624 | 0.8570 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.9226 | 2.0225 | 0.9420 | 3.1035 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.7069 | 1.2625 | 0.7152 | 1.0643 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.5335 | 1.6114 | 2.1392 | 2.0410 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.8586 | 0.6279 | 0.8372 | 1.1884 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.7028 | 1.5446 | 1.5932 | 1.4215 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.5711 | 0.6026 | 0.4445 | 0.5372 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.6753 | 0.9297 | 1.0193 | 0.7309 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 1.2519 | 1.0851 | 1.0164 | 1.1537 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.4232 | 0.5135 | 0.5223 | 0.4919 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.5307 | 0.8167 | 1.1066 | 0.5923 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.6309 | 0.6645 | 1.1890 | 0.7934 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.9984 | 1.0574 | 1.0017 | 1.0789 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.7979 | 0.8798 | 1.3510 | 1.6231 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 1.3319 | 1.3266 | 0.8853 | 1.2465 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 1.7488 | 1.3204 | 1.4150 | 1.4843 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.4659 | 0.5300 | 1.0774 | 0.9280 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 1.1785 | 1.4551 | 1.3893 | 0.9232 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.7540 | 0.7454 | 0.9183 | 0.8765 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.1361 | 1.0018 | 1.3664 | 0.9807 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.5575 | 0.5043 | 0.7267 | 1.0205 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 1.1891 | 0.9596 | 0.7536 | 0.9718 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.5744 | 0.6168 | 0.5519 | 0.6423 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.0945 | 0.6612 | 0.7271 | 0.7054 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 1.1843 | 0.9562 | 0.8711 | 0.6209 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 1.1020 | 1.3046 | 1.0507 | 0.9789 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.9742 | 1.1400 | 0.7056 | 0.8533 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.7524 | 0.9224 | 0.9909 | 1.0037 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.6985 | 0.4626 | 0.6058 | 0.5622 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.5716 | 0.5286 | 0.9554 | 0.6327 |

### Consistency

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 3.9619 | 6.2160 | 9.3090 | 11.6459 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.9405 | 0.4848 | 1.6188 | 1.0858 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.8424 | 0.8322 | 1.7934 | 1.3670 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.8580 | 0.6918 | 0.3473 | 0.5755 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.1008 | 0.7005 | 1.5896 | 1.0436 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 1.3563 | 1.3263 | 0.8955 | 0.8623 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.5701 | 1.1320 | 1.4109 | 1.0933 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.8886 | 0.7569 | 0.6424 | 0.8471 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 2.0797 | 1.0141 | 1.2533 | 1.3790 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.5846 | 1.0490 | 0.9349 | 1.0788 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.7418 | 1.6292 | 0.8146 | 1.3225 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 1.1734 | 0.9903 | 1.1072 | 0.9355 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.4884 | 2.4525 | 1.3565 | 1.6424 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.7134 | 0.5933 | 0.7641 | 0.6647 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.7241 | 1.7070 | 1.8487 | 2.3805 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.8892 | 0.6690 | 1.2808 | 1.7329 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.7652 | 1.6875 | 1.5280 | 1.2039 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.6073 | 0.7526 | 0.6850 | 0.4122 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.2261 | 1.2569 | 1.6644 | 0.7606 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 1.2902 | 1.2738 | 1.1868 | 1.4609 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.4275 | 0.4551 | 0.3957 | 0.3500 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.6918 | 0.7031 | 1.0219 | 0.5818 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.3407 | 1.2955 | 1.2376 | 0.8281 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.7308 | 0.8096 | 0.8761 | 0.6894 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.9090 | 0.5979 | 0.9942 | 1.0748 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 1.0669 | 1.0251 | 0.8376 | 0.9142 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 1.3632 | 1.4733 | 1.2735 | 1.3226 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.4856 | 0.6706 | 0.6218 | 0.7487 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 1.3432 | 1.7966 | 1.2090 | 1.4845 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.9077 | 0.9477 | 0.7531 | 0.7094 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.8533 | 1.1585 | 0.8034 | 0.8048 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.7505 | 0.5539 | 0.7008 | 0.6356 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.7690 | 1.0498 | 1.0830 | 0.8835 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.6122 | 0.9362 | 0.6190 | 0.7631 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.3492 | 1.0380 | 0.7878 | 0.6994 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.5313 | 0.6978 | 0.4459 | 1.0058 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 1.6374 | 0.9369 | 1.0278 | 1.9603 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.7862 | 1.6037 | 1.1442 | 0.8265 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.7180 | 1.1545 | 0.7055 | 1.0451 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.8034 | 0.6285 | 0.6937 | 0.4351 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.7138 | 0.7244 | 0.7666 | 1.2394 |

### Newsworthiness

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 7.0597 | 4.1042 | 6.2957 | 2.7082 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 1.6841 | 1.4945 | 1.4355 | 0.9517 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 1.4450 | 1.4910 | 1.0439 | 1.0420 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.7230 | 0.8002 | 0.5234 | 0.9626 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.9020 | 1.0234 | 0.8856 | 0.7795 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.8843 | 1.1022 | 0.7961 | 0.4982 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.0717 | 0.7991 | 0.5791 | 0.7675 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.9632 | 0.5880 | 1.2812 | 0.9302 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.0195 | 1.2254 | 0.6650 | 0.5669 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 1.2365 | 1.0880 | 0.9619 | 0.6465 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.7034 | 1.1769 | 1.1848 | 1.1669 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.8894 | 0.8072 | 0.5463 | 0.5935 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.2596 | 1.0253 | 1.4484 | 1.6435 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.8578 | 1.3192 | 0.9974 | 0.7295 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.1696 | 0.7190 | 0.7170 | 0.8571 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 1.1410 | 0.8066 | 1.0464 | 0.9169 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.1897 | 1.3391 | 1.3583 | 1.5263 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.6579 | 1.1772 | 0.7950 | 1.0805 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.6514 | 0.5135 | 0.7482 | 0.8222 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 1.6570 | 1.1971 | 1.5170 | 1.2510 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.6157 | 0.6118 | 0.7371 | 0.7511 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.8374 | 0.9589 | 1.3716 | 0.8070 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.6740 | 1.1921 | 1.1137 | 1.0088 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 1.4414 | 1.2301 | 1.0062 | 1.9391 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.5353 | 0.6056 | 1.0803 | 0.8004 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 1.3489 | 2.4130 | 1.1459 | 1.3279 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.8814 | 1.5815 | 1.6954 | 1.2797 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 1.2092 | 0.7220 | 0.9210 | 1.0052 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.9174 | 1.0070 | 1.2823 | 1.2262 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.4578 | 0.5786 | 0.4811 | 0.5985 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.5051 | 0.7307 | 1.5108 | 1.3475 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.5477 | 0.5440 | 0.6891 | 0.9462 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 1.1291 | 0.8153 | 1.1580 | 1.3595 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.6138 | 0.7320 | 0.6263 | 1.2057 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.7897 | 0.9256 | 1.0110 | 0.4552 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.8245 | 1.5540 | 1.1969 | 1.4299 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 1.5020 | 1.3823 | 0.9779 | 1.4100 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 1.0058 | 1.3025 | 0.8969 | 0.9629 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.6669 | 0.9748 | 1.1205 | 1.3852 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.7123 | 0.7164 | 0.8314 | 0.9163 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.9723 | 0.7462 | 0.6978 | 1.2333 |

### Hygiene

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.6598 | 2.2716 | 4.7588 | 2.6984 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.8350 | 0.6745 | 1.5807 | 1.3100 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.9471 | 0.8936 | 1.2530 | 1.4761 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.6389 | 0.4287 | 0.3299 | 0.8784 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.4339 | 1.4723 | 0.7959 | 0.8756 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.9974 | 1.0903 | 0.8821 | 0.7446 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.1782 | 1.0972 | 0.5949 | 1.2473 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.6940 | 0.6244 | 1.4261 | 0.6389 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.9237 | 0.9219 | 0.9528 | 1.2415 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 1.0450 | 1.0584 | 1.5053 | 1.1711 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.3730 | 1.1614 | 1.2348 | 1.7632 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 1.0499 | 1.2546 | 1.1666 | 1.0670 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.9065 | 1.0070 | 0.7363 | 1.0844 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.9616 | 0.7520 | 0.6218 | 0.5190 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.0445 | 1.1655 | 2.0162 | 2.3286 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.7464 | 0.8234 | 0.8858 | 0.7013 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.8818 | 1.5057 | 1.7640 | 1.1972 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.8353 | 1.1537 | 0.8351 | 0.7736 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.0044 | 0.9211 | 0.9973 | 1.1705 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 1.8791 | 1.0697 | 1.4478 | 1.0790 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.9881 | 0.9407 | 0.6723 | 0.7647 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.6690 | 0.7058 | 0.5764 | 0.8822 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.9137 | 1.3979 | 0.8245 | 1.0309 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.7387 | 0.8108 | 1.4502 | 0.7876 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.9951 | 1.1614 | 0.6610 | 1.0907 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 1.4396 | 0.6077 | 0.9208 | 1.0368 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 1.3933 | 1.1078 | 1.7427 | 1.4019 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.9333 | 0.8867 | 1.0271 | 0.8193 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.8556 | 0.7403 | 1.1698 | 0.9068 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.7088 | 0.7408 | 0.6138 | 0.4738 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.3166 | 1.0209 | 1.1789 | 1.1974 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.8105 | 1.0892 | 1.1725 | 0.7730 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.9468 | 1.0792 | 1.0141 | 0.7120 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 1.0684 | 0.8819 | 0.5250 | 0.9487 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.0699 | 1.6270 | 0.7598 | 0.7357 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 1.1517 | 0.9804 | 0.5633 | 1.0648 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 1.0885 | 1.0447 | 1.0045 | 1.1688 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 1.0236 | 0.9755 | 1.4052 | 0.8020 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.2147 | 1.2323 | 1.7287 | 1.4225 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 1.0107 | 1.2178 | 0.7421 | 0.6042 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.8458 | 1.1860 | 0.7270 | 1.1836 |

---

## Export layout

- `json/` — pairwise rows and per-judge G-Eval tables (JSON)
- `tables/` — CSV summaries (win rates, Bradley–Terry)
- `reports/` — Markdown / LaTeX for reading and papers

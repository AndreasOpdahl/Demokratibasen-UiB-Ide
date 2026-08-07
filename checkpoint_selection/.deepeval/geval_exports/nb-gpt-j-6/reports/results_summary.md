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
| GPT4o-mini | 1.000 | 1.000 | 1.000 | 1.000 |
| checkpoint-1000-gen0-inputs-refs-preds-1000-examples | 0.538 | 0.615 | 0.654 | 0.731 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.269 | 0.327 | 0.288 | 0.327 |
| checkpoint-10000-gen0-inputs-refs-preds-1000-examples | 0.520 | 0.600 | 0.560 | 0.580 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.442 | 0.442 | 0.538 | 0.462 |
| checkpoint-1500-gen0-inputs-refs-preds-1000-examples | 0.635 | 0.519 | 0.673 | 0.635 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.360 | 0.300 | 0.240 | 0.280 |
| checkpoint-2000-gen0-inputs-refs-preds-1000-examples | 0.769 | 0.615 | 0.577 | 0.615 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.260 | 0.380 | 0.240 | 0.280 |
| checkpoint-2500-gen0-inputs-refs-preds-1000-examples | 0.720 | 0.580 | 0.740 | 0.760 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.240 | 0.120 | 0.360 | 0.140 |
| checkpoint-3000-gen0-inputs-refs-preds-1000-examples | 0.519 | 0.654 | 0.635 | 0.615 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.380 | 0.460 | 0.440 | 0.380 |
| checkpoint-3500-gen0-inputs-refs-preds-1000-examples | 0.577 | 0.577 | 0.635 | 0.519 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.460 | 0.460 | 0.440 | 0.460 |
| checkpoint-4000-gen0-inputs-refs-preds-1000-examples | 0.712 | 0.577 | 0.692 | 0.673 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.308 | 0.327 | 0.212 | 0.269 |
| checkpoint-4500-gen0-inputs-refs-preds-1000-examples | 0.580 | 0.620 | 0.660 | 0.680 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.300 | 0.340 | 0.280 | 0.180 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 0.760 | 0.620 | 0.680 | 0.740 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.212 | 0.192 | 0.231 | 0.192 |
| checkpoint-5000-gen0-inputs-refs-preds-1000-examples | 0.640 | 0.660 | 0.640 | 0.700 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.480 | 0.380 | 0.460 | 0.420 |
| checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.580 | 0.600 | 0.640 | 0.720 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.260 | 0.380 | 0.320 |
| checkpoint-6000-gen0-inputs-refs-preds-1000-examples | 0.400 | 0.260 | 0.360 | 0.420 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.320 | 0.360 | 0.260 | 0.320 |
| checkpoint-6500-gen0-inputs-refs-preds-1000-examples | 0.200 | 0.200 | 0.260 | 0.240 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.580 | 0.520 | 0.480 | 0.400 |
| checkpoint-7000-gen0-inputs-refs-preds-1000-examples | 0.596 | 0.712 | 0.654 | 0.654 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.480 | 0.480 | 0.420 | 0.380 |
| checkpoint-7500-gen0-inputs-refs-preds-1000-examples | 0.519 | 0.481 | 0.500 | 0.538 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.520 | 0.620 | 0.600 | 0.580 |
| checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.580 | 0.620 | 0.560 | 0.680 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.460 | 0.400 | 0.360 | 0.440 |
| checkpoint-8500-gen0-inputs-refs-preds-1000-examples | 0.462 | 0.442 | 0.538 | 0.538 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.460 | 0.500 | 0.440 | 0.300 |
| checkpoint-9000-gen0-inputs-refs-preds-1000-examples | 0.519 | 0.596 | 0.577 | 0.538 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.481 | 0.615 | 0.442 | 0.519 |
| checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.740 | 0.840 | 0.760 | 0.800 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.420 | 0.620 | 0.380 | 0.460 |

### Consistency

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.960 | 0.880 | 1.000 | 1.000 |
| checkpoint-1000-gen0-inputs-refs-preds-1000-examples | 0.635 | 0.654 | 0.635 | 0.635 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.212 | 0.308 | 0.250 | 0.288 |
| checkpoint-10000-gen0-inputs-refs-preds-1000-examples | 0.600 | 0.620 | 0.540 | 0.620 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.481 | 0.365 |
| checkpoint-1500-gen0-inputs-refs-preds-1000-examples | 0.654 | 0.635 | 0.712 | 0.712 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.360 | 0.400 | 0.320 | 0.240 |
| checkpoint-2000-gen0-inputs-refs-preds-1000-examples | 0.500 | 0.558 | 0.558 | 0.538 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.280 | 0.360 | 0.280 | 0.160 |
| checkpoint-2500-gen0-inputs-refs-preds-1000-examples | 0.640 | 0.720 | 0.720 | 0.700 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.260 | 0.280 | 0.420 | 0.380 |
| checkpoint-3000-gen0-inputs-refs-preds-1000-examples | 0.635 | 0.654 | 0.846 | 0.673 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.220 | 0.400 | 0.260 | 0.380 |
| checkpoint-3500-gen0-inputs-refs-preds-1000-examples | 0.654 | 0.558 | 0.673 | 0.692 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.360 | 0.260 | 0.400 | 0.300 |
| checkpoint-4000-gen0-inputs-refs-preds-1000-examples | 0.654 | 0.712 | 0.635 | 0.615 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.288 | 0.212 | 0.308 | 0.308 |
| checkpoint-4500-gen0-inputs-refs-preds-1000-examples | 0.660 | 0.680 | 0.740 | 0.780 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.260 | 0.320 | 0.200 | 0.280 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 0.740 | 0.680 | 0.580 | 0.640 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.212 | 0.346 | 0.212 | 0.231 |
| checkpoint-5000-gen0-inputs-refs-preds-1000-examples | 0.840 | 0.700 | 0.800 | 0.720 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.340 | 0.340 | 0.320 | 0.340 |
| checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.480 | 0.540 | 0.560 | 0.660 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.260 | 0.240 | 0.260 | 0.340 |
| checkpoint-6000-gen0-inputs-refs-preds-1000-examples | 0.360 | 0.340 | 0.380 | 0.440 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.300 | 0.280 | 0.280 | 0.240 |
| checkpoint-6500-gen0-inputs-refs-preds-1000-examples | 0.400 | 0.380 | 0.440 | 0.400 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.480 | 0.340 | 0.320 | 0.440 |
| checkpoint-7000-gen0-inputs-refs-preds-1000-examples | 0.596 | 0.558 | 0.635 | 0.596 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.400 | 0.420 | 0.380 | 0.440 |
| checkpoint-7500-gen0-inputs-refs-preds-1000-examples | 0.673 | 0.635 | 0.577 | 0.500 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.520 | 0.560 | 0.520 | 0.460 |
| checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.780 | 0.660 | 0.720 | 0.680 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.440 | 0.460 | 0.440 | 0.380 |
| checkpoint-8500-gen0-inputs-refs-preds-1000-examples | 0.577 | 0.558 | 0.577 | 0.577 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.300 | 0.520 | 0.300 | 0.520 |
| checkpoint-9000-gen0-inputs-refs-preds-1000-examples | 0.654 | 0.519 | 0.596 | 0.596 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.558 | 0.500 | 0.385 | 0.500 |
| checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.720 | 0.700 | 0.680 | 0.760 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.520 | 0.500 | 0.540 | 0.360 |

### Newsworthiness

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.920 | 0.920 | 0.920 | 0.860 |
| checkpoint-1000-gen0-inputs-refs-preds-1000-examples | 0.423 | 0.481 | 0.538 | 0.558 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.327 | 0.404 | 0.365 | 0.404 |
| checkpoint-10000-gen0-inputs-refs-preds-1000-examples | 0.480 | 0.480 | 0.500 | 0.480 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.462 | 0.558 | 0.462 | 0.500 |
| checkpoint-1500-gen0-inputs-refs-preds-1000-examples | 0.500 | 0.481 | 0.558 | 0.538 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.240 | 0.300 | 0.360 | 0.280 |
| checkpoint-2000-gen0-inputs-refs-preds-1000-examples | 0.635 | 0.635 | 0.635 | 0.635 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.440 | 0.540 | 0.340 | 0.520 |
| checkpoint-2500-gen0-inputs-refs-preds-1000-examples | 0.660 | 0.660 | 0.640 | 0.680 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.400 | 0.420 | 0.420 | 0.380 |
| checkpoint-3000-gen0-inputs-refs-preds-1000-examples | 0.596 | 0.558 | 0.462 | 0.577 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.580 | 0.600 | 0.560 | 0.520 |
| checkpoint-3500-gen0-inputs-refs-preds-1000-examples | 0.692 | 0.577 | 0.654 | 0.635 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.560 | 0.580 | 0.460 | 0.460 |
| checkpoint-4000-gen0-inputs-refs-preds-1000-examples | 0.635 | 0.654 | 0.577 | 0.558 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.327 | 0.423 | 0.327 | 0.442 |
| checkpoint-4500-gen0-inputs-refs-preds-1000-examples | 0.420 | 0.580 | 0.600 | 0.600 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.460 | 0.380 | 0.320 | 0.400 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 0.720 | 0.660 | 0.720 | 0.760 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.308 | 0.288 | 0.308 | 0.481 |
| checkpoint-5000-gen0-inputs-refs-preds-1000-examples | 0.540 | 0.580 | 0.580 | 0.460 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.540 | 0.440 | 0.420 | 0.440 |
| checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.540 | 0.560 | 0.500 | 0.440 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.440 | 0.400 | 0.480 | 0.420 |
| checkpoint-6000-gen0-inputs-refs-preds-1000-examples | 0.360 | 0.260 | 0.280 | 0.300 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.320 | 0.400 | 0.360 | 0.380 |
| checkpoint-6500-gen0-inputs-refs-preds-1000-examples | 0.200 | 0.200 | 0.240 | 0.220 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.580 | 0.520 | 0.560 | 0.460 |
| checkpoint-7000-gen0-inputs-refs-preds-1000-examples | 0.654 | 0.635 | 0.558 | 0.558 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.480 | 0.520 | 0.360 | 0.460 |
| checkpoint-7500-gen0-inputs-refs-preds-1000-examples | 0.404 | 0.423 | 0.481 | 0.481 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.480 | 0.460 | 0.440 | 0.400 |
| checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.580 | 0.460 | 0.580 | 0.700 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.380 | 0.460 | 0.660 | 0.500 |
| checkpoint-8500-gen0-inputs-refs-preds-1000-examples | 0.481 | 0.462 | 0.596 | 0.442 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.340 | 0.280 | 0.360 | 0.360 |
| checkpoint-9000-gen0-inputs-refs-preds-1000-examples | 0.673 | 0.577 | 0.712 | 0.558 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.558 | 0.615 | 0.442 | 0.519 |
| checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.680 | 0.600 | 0.720 | 0.640 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.480 | 0.460 | 0.440 | 0.480 |

### Hygiene

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.980 | 0.960 | 1.000 | 0.960 |
| checkpoint-1000-gen0-inputs-refs-preds-1000-examples | 0.692 | 0.712 | 0.712 | 0.692 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.404 | 0.481 | 0.423 | 0.365 |
| checkpoint-10000-gen0-inputs-refs-preds-1000-examples | 0.600 | 0.600 | 0.560 | 0.640 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.615 | 0.712 | 0.538 |
| checkpoint-1500-gen0-inputs-refs-preds-1000-examples | 0.654 | 0.558 | 0.712 | 0.615 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.320 | 0.400 | 0.280 | 0.320 |
| checkpoint-2000-gen0-inputs-refs-preds-1000-examples | 0.558 | 0.462 | 0.500 | 0.596 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.340 | 0.360 | 0.240 | 0.260 |
| checkpoint-2500-gen0-inputs-refs-preds-1000-examples | 0.700 | 0.540 | 0.640 | 0.760 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.380 | 0.320 | 0.320 | 0.260 |
| checkpoint-3000-gen0-inputs-refs-preds-1000-examples | 0.462 | 0.442 | 0.558 | 0.519 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.240 | 0.300 | 0.300 | 0.380 |
| checkpoint-3500-gen0-inputs-refs-preds-1000-examples | 0.654 | 0.577 | 0.769 | 0.692 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.460 | 0.340 | 0.380 | 0.320 |
| checkpoint-4000-gen0-inputs-refs-preds-1000-examples | 0.712 | 0.654 | 0.635 | 0.692 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.365 | 0.327 | 0.481 | 0.327 |
| checkpoint-4500-gen0-inputs-refs-preds-1000-examples | 0.620 | 0.660 | 0.780 | 0.620 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.360 | 0.500 | 0.100 | 0.400 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 0.740 | 0.680 | 0.680 | 0.800 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.288 | 0.442 | 0.250 | 0.212 |
| checkpoint-5000-gen0-inputs-refs-preds-1000-examples | 0.620 | 0.540 | 0.720 | 0.580 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.540 | 0.560 | 0.520 | 0.480 |
| checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.640 | 0.640 | 0.560 | 0.600 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.460 | 0.540 | 0.280 | 0.500 |
| checkpoint-6000-gen0-inputs-refs-preds-1000-examples | 0.580 | 0.500 | 0.300 | 0.440 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.200 | 0.280 | 0.180 | 0.180 |
| checkpoint-6500-gen0-inputs-refs-preds-1000-examples | 0.480 | 0.400 | 0.260 | 0.320 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.120 | 0.160 | 0.360 | 0.280 |
| checkpoint-7000-gen0-inputs-refs-preds-1000-examples | 0.654 | 0.635 | 0.654 | 0.673 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.300 | 0.320 | 0.380 | 0.320 |
| checkpoint-7500-gen0-inputs-refs-preds-1000-examples | 0.500 | 0.558 | 0.538 | 0.519 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.280 | 0.360 | 0.360 | 0.440 |
| checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.680 | 0.580 | 0.640 | 0.580 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.320 | 0.360 | 0.380 | 0.380 |
| checkpoint-8500-gen0-inputs-refs-preds-1000-examples | 0.577 | 0.558 | 0.577 | 0.558 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.300 | 0.340 | 0.380 | 0.400 |
| checkpoint-9000-gen0-inputs-refs-preds-1000-examples | 0.635 | 0.615 | 0.577 | 0.538 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.404 | 0.365 | 0.462 | 0.481 |
| checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.780 | 0.780 | 0.740 | 0.780 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.380 | 0.460 | 0.560 | 0.460 |

---

## 2. Bradley–Terry strengths (θ)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Relevance

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 135873.9252 | 155586.6458 | 364666.3489 | 183052.6695 |
| checkpoint-1000-gen0-inputs-refs-preds-1000-examples | 0.7978 | 1.0792 | 1.3660 | 2.0782 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.2615 | 0.3914 | 0.2872 | 0.3663 |
| checkpoint-10000-gen0-inputs-refs-preds-1000-examples | 0.9050 | 1.4634 | 1.0315 | 1.1447 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.6080 | 0.6110 | 0.9737 | 0.6703 |
| checkpoint-1500-gen0-inputs-refs-preds-1000-examples | 1.5597 | 0.9164 | 1.9541 | 1.7136 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.3864 | 0.2770 | 0.1906 | 0.2399 |
| checkpoint-2000-gen0-inputs-refs-preds-1000-examples | 3.3710 | 1.5021 | 1.2219 | 1.5043 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.2531 | 0.4922 | 0.2210 | 0.2911 |
| checkpoint-2500-gen0-inputs-refs-preds-1000-examples | 1.7677 | 0.9396 | 2.0035 | 2.1471 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.2930 | 0.1098 | 0.5084 | 0.1403 |
| checkpoint-3000-gen0-inputs-refs-preds-1000-examples | 0.8164 | 1.6052 | 1.4035 | 1.3464 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.4285 | 0.6855 | 0.5809 | 0.4292 |
| checkpoint-3500-gen0-inputs-refs-preds-1000-examples | 1.0437 | 1.0118 | 1.3821 | 0.7705 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.7477 | 0.6945 | 0.6540 | 0.7309 |
| checkpoint-4000-gen0-inputs-refs-preds-1000-examples | 2.5600 | 1.2057 | 2.1433 | 2.1344 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.3438 | 0.3696 | 0.1877 | 0.2600 |
| checkpoint-4500-gen0-inputs-refs-preds-1000-examples | 1.2760 | 1.5129 | 1.7328 | 2.0559 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.3067 | 0.3737 | 0.2712 | 0.1461 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 2.6349 | 1.2001 | 1.6806 | 2.4120 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.1983 | 0.1918 | 0.2147 | 0.1805 |
| checkpoint-5000-gen0-inputs-refs-preds-1000-examples | 1.5872 | 1.9275 | 1.6604 | 2.2960 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.6799 | 0.4732 | 0.6237 | 0.5064 |
| checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 1.0464 | 1.1558 | 1.3321 | 2.0337 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.8014 | 0.2251 | 0.4556 | 0.3066 |
| checkpoint-6000-gen0-inputs-refs-preds-1000-examples | 0.4986 | 0.2413 | 0.4175 | 0.5291 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.3784 | 0.4731 | 0.2690 | 0.3578 |
| checkpoint-6500-gen0-inputs-refs-preds-1000-examples | 0.1568 | 0.1618 | 0.2097 | 0.1914 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.9962 | 0.8171 | 0.5940 | 0.4636 |
| checkpoint-7000-gen0-inputs-refs-preds-1000-examples | 1.3792 | 2.2519 | 1.7836 | 1.8404 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.7696 | 0.7488 | 0.5713 | 0.4684 |
| checkpoint-7500-gen0-inputs-refs-preds-1000-examples | 1.0041 | 0.8670 | 0.9461 | 1.1355 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.9236 | 1.5477 | 1.3265 | 1.3145 |
| checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 1.1179 | 1.4579 | 1.0225 | 1.9170 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.6657 | 0.5380 | 0.3895 | 0.6503 |
| checkpoint-8500-gen0-inputs-refs-preds-1000-examples | 0.6746 | 0.5690 | 0.9278 | 0.8971 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.6556 | 0.7764 | 0.6075 | 0.3131 |
| checkpoint-9000-gen0-inputs-refs-preds-1000-examples | 0.8202 | 1.2485 | 1.1228 | 0.9657 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.5962 | 1.1247 | 0.4937 | 0.7118 |
| checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 2.1603 | 4.1477 | 2.4797 | 3.5000 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.5368 | 1.1983 | 0.3894 | 0.5531 |

### Consistency

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 35.6876 | 9.0113 | 1126626.8177 | 342064.9726 |
| checkpoint-1000-gen0-inputs-refs-preds-1000-examples | 1.5469 | 1.7309 | 1.0954 | 1.1972 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.2444 | 0.4570 | 0.2318 | 0.3062 |
| checkpoint-10000-gen0-inputs-refs-preds-1000-examples | 1.5779 | 1.7009 | 0.9092 | 1.4323 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.9174 | 0.9548 | 0.7073 | 0.4133 |
| checkpoint-1500-gen0-inputs-refs-preds-1000-examples | 2.1674 | 2.0474 | 2.5055 | 2.5672 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.4851 | 0.6095 | 0.2998 | 0.1979 |
| checkpoint-2000-gen0-inputs-refs-preds-1000-examples | 1.0835 | 1.4226 | 1.1395 | 1.0074 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.3593 | 0.5521 | 0.2675 | 0.1218 |
| checkpoint-2500-gen0-inputs-refs-preds-1000-examples | 1.4966 | 2.2450 | 1.6849 | 1.5129 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.3704 | 0.4324 | 0.6517 | 0.5353 |
| checkpoint-3000-gen0-inputs-refs-preds-1000-examples | 1.8264 | 1.9238 | 6.1663 | 1.8070 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.2299 | 0.6612 | 0.2109 | 0.4124 |
| checkpoint-3500-gen0-inputs-refs-preds-1000-examples | 1.8937 | 1.1651 | 1.6830 | 1.9377 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.5265 | 0.3307 | 0.4788 | 0.2990 |
| checkpoint-4000-gen0-inputs-refs-preds-1000-examples | 2.3510 | 2.9058 | 1.5218 | 1.3556 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.3714 | 0.2624 | 0.2883 | 0.3135 |
| checkpoint-4500-gen0-inputs-refs-preds-1000-examples | 2.3760 | 2.3431 | 2.6735 | 3.7633 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.2982 | 0.4147 | 0.1494 | 0.2521 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 2.7544 | 2.0002 | 0.9997 | 1.3059 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.2476 | 0.5311 | 0.1790 | 0.2302 |
| checkpoint-5000-gen0-inputs-refs-preds-1000-examples | 8.1260 | 2.9227 | 5.1958 | 2.7576 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.4120 | 0.4447 | 0.2548 | 0.3538 |
| checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.7444 | 1.0566 | 0.8530 | 1.5357 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.2726 | 0.2707 | 0.2252 | 0.3684 |
| checkpoint-6000-gen0-inputs-refs-preds-1000-examples | 0.5189 | 0.4726 | 0.4404 | 0.6163 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.3841 | 0.3857 | 0.2571 | 0.2281 |
| checkpoint-6500-gen0-inputs-refs-preds-1000-examples | 0.5171 | 0.5516 | 0.4797 | 0.4107 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.7780 | 0.4280 | 0.2637 | 0.4942 |
| checkpoint-7000-gen0-inputs-refs-preds-1000-examples | 1.7249 | 1.4037 | 1.5777 | 1.2821 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.6388 | 0.7009 | 0.4083 | 0.6445 |
| checkpoint-7500-gen0-inputs-refs-preds-1000-examples | 2.6544 | 2.1649 | 1.2673 | 0.9100 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 1.2003 | 1.3874 | 0.9597 | 0.6844 |
| checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 4.3801 | 2.0482 | 2.5003 | 1.9823 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.7533 | 0.8351 | 0.6167 | 0.4385 |
| checkpoint-8500-gen0-inputs-refs-preds-1000-examples | 1.3121 | 1.2846 | 1.0731 | 1.0927 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.3595 | 1.0924 | 0.2984 | 0.9524 |
| checkpoint-9000-gen0-inputs-refs-preds-1000-examples | 1.9199 | 1.1275 | 1.1299 | 1.2342 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.9735 | 0.8773 | 0.3505 | 0.6317 |
| checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 2.6697 | 2.3821 | 1.6891 | 2.4920 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.8605 | 0.8803 | 0.6983 | 0.3577 |

### Newsworthiness

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 12.2712 | 12.4460 | 12.2485 | 6.3644 |
| checkpoint-1000-gen0-inputs-refs-preds-1000-examples | 0.6436 | 0.8005 | 1.0388 | 1.1587 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.4326 | 0.6402 | 0.5497 | 0.6770 |
| checkpoint-10000-gen0-inputs-refs-preds-1000-examples | 1.0065 | 1.0017 | 1.0760 | 1.0173 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.9204 | 1.3806 | 0.8630 | 1.0264 |
| checkpoint-1500-gen0-inputs-refs-preds-1000-examples | 0.9899 | 0.9514 | 1.2468 | 1.1763 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.2712 | 0.3908 | 0.5257 | 0.3686 |
| checkpoint-2000-gen0-inputs-refs-preds-1000-examples | 1.8719 | 1.9340 | 1.9745 | 1.8060 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.8186 | 1.2771 | 0.4985 | 1.0716 |
| checkpoint-2500-gen0-inputs-refs-preds-1000-examples | 1.7556 | 1.8194 | 1.5926 | 1.8784 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.7020 | 0.7519 | 0.7912 | 0.6540 |
| checkpoint-3000-gen0-inputs-refs-preds-1000-examples | 1.5292 | 1.2640 | 0.8548 | 1.3867 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 1.3779 | 1.4873 | 1.1662 | 1.0471 |
| checkpoint-3500-gen0-inputs-refs-preds-1000-examples | 2.3165 | 1.3300 | 1.8646 | 1.6710 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 1.3833 | 1.5464 | 0.9065 | 0.9166 |
| checkpoint-4000-gen0-inputs-refs-preds-1000-examples | 1.9451 | 2.2130 | 1.5753 | 1.3963 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.4467 | 0.6589 | 0.4691 | 0.7816 |
| checkpoint-4500-gen0-inputs-refs-preds-1000-examples | 0.7338 | 1.4383 | 1.6052 | 1.5318 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.8264 | 0.6164 | 0.4455 | 0.6261 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 2.3810 | 1.7789 | 2.3601 | 2.9110 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.4502 | 0.3867 | 0.4298 | 0.9448 |
| checkpoint-5000-gen0-inputs-refs-preds-1000-examples | 1.2086 | 1.4461 | 1.4398 | 0.8829 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 1.1386 | 0.7205 | 0.6637 | 0.7413 |
| checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 1.1267 | 1.2253 | 0.9696 | 0.7596 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.7966 | 0.6151 | 0.8570 | 0.6856 |
| checkpoint-6000-gen0-inputs-refs-preds-1000-examples | 0.5582 | 0.3205 | 0.3694 | 0.4289 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.4965 | 0.7337 | 0.6078 | 0.6321 |
| checkpoint-6500-gen0-inputs-refs-preds-1000-examples | 0.2129 | 0.2191 | 0.2723 | 0.2601 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 1.3168 | 1.0506 | 1.1224 | 0.8519 |
| checkpoint-7000-gen0-inputs-refs-preds-1000-examples | 2.0493 | 1.8064 | 1.3855 | 1.3100 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.9720 | 1.1049 | 0.5664 | 0.9045 |
| checkpoint-7500-gen0-inputs-refs-preds-1000-examples | 0.7300 | 0.7990 | 0.9998 | 0.9821 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 1.0234 | 0.9381 | 0.8553 | 0.7489 |
| checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 1.3768 | 0.8364 | 1.3934 | 2.4056 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.5781 | 0.8786 | 2.0139 | 0.9881 |
| checkpoint-8500-gen0-inputs-refs-preds-1000-examples | 0.8970 | 0.8236 | 1.5083 | 0.7756 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.4548 | 0.3394 | 0.5285 | 0.5174 |
| checkpoint-9000-gen0-inputs-refs-preds-1000-examples | 2.1005 | 1.3877 | 2.3484 | 1.2733 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 1.1013 | 1.4738 | 0.6672 | 0.9599 |
| checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 1.8466 | 1.2880 | 2.2962 | 1.6359 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.8252 | 0.7624 | 0.7128 | 0.8542 |

### Hygiene

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 57.9772 | 24.4931 | 1440557.1105 | 26.4603 |
| checkpoint-1000-gen0-inputs-refs-preds-1000-examples | 2.3200 | 2.4330 | 1.8602 | 2.1687 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.6028 | 0.8650 | 0.5869 | 0.5291 |
| checkpoint-10000-gen0-inputs-refs-preds-1000-examples | 1.4179 | 1.4210 | 1.0847 | 1.7656 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 1.1119 | 1.7573 | 2.3759 | 1.2785 |
| checkpoint-1500-gen0-inputs-refs-preds-1000-examples | 2.2762 | 1.4019 | 2.5012 | 1.9111 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.3973 | 0.6106 | 0.2356 | 0.4047 |
| checkpoint-2000-gen0-inputs-refs-preds-1000-examples | 1.3151 | 0.8936 | 0.8687 | 1.6358 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.4401 | 0.4945 | 0.1732 | 0.2964 |
| checkpoint-2500-gen0-inputs-refs-preds-1000-examples | 1.9951 | 0.9599 | 1.2240 | 2.8127 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.6725 | 0.4836 | 0.3710 | 0.3843 |
| checkpoint-3000-gen0-inputs-refs-preds-1000-examples | 0.7993 | 0.7515 | 0.9097 | 1.0223 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.2580 | 0.3996 | 0.2509 | 0.5334 |
| checkpoint-3500-gen0-inputs-refs-preds-1000-examples | 1.9914 | 1.3940 | 3.1099 | 2.2765 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.8586 | 0.4822 | 0.3800 | 0.4405 |
| checkpoint-4000-gen0-inputs-refs-preds-1000-examples | 3.0661 | 2.0520 | 1.5789 | 2.7889 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.5062 | 0.4744 | 0.7548 | 0.4690 |
| checkpoint-4500-gen0-inputs-refs-preds-1000-examples | 1.6964 | 2.0212 | 3.8076 | 1.6989 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.5119 | 0.9785 | 0.0607 | 0.6180 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 2.8903 | 2.1324 | 1.6674 | 3.9303 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.3654 | 0.8022 | 0.2502 | 0.2517 |
| checkpoint-5000-gen0-inputs-refs-preds-1000-examples | 1.7901 | 1.2455 | 2.8203 | 1.4613 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 1.0385 | 1.2970 | 0.8062 | 0.8839 |
| checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 1.6848 | 1.8191 | 0.9093 | 1.4029 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.9020 | 1.1741 | 0.2544 | 1.0456 |
| checkpoint-6000-gen0-inputs-refs-preds-1000-examples | 1.3982 | 0.9388 | 0.2847 | 0.7094 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.2095 | 0.3482 | 0.1277 | 0.1979 |
| checkpoint-6500-gen0-inputs-refs-preds-1000-examples | 0.8302 | 0.6312 | 0.1778 | 0.3938 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.1009 | 0.1597 | 0.3420 | 0.3032 |
| checkpoint-7000-gen0-inputs-refs-preds-1000-examples | 2.2994 | 1.9442 | 1.8610 | 2.4432 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.3941 | 0.4507 | 0.5336 | 0.4584 |
| checkpoint-7500-gen0-inputs-refs-preds-1000-examples | 1.1707 | 1.4160 | 1.1031 | 1.2791 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.3714 | 0.5393 | 0.4288 | 0.7762 |
| checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 2.2177 | 1.3209 | 1.5136 | 1.3143 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.3985 | 0.4753 | 0.4018 | 0.5266 |
| checkpoint-8500-gen0-inputs-refs-preds-1000-examples | 1.4293 | 1.3320 | 1.1663 | 1.3291 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.4352 | 0.5206 | 0.4818 | 0.6652 |
| checkpoint-9000-gen0-inputs-refs-preds-1000-examples | 1.7637 | 1.6479 | 1.1649 | 1.2202 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.4818 | 0.4507 | 0.4565 | 0.7037 |
| checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 3.6729 | 3.6707 | 2.3279 | 3.5600 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.4738 | 0.7369 | 0.8567 | 0.7334 |

---

## Export layout

- `json/` — pairwise rows and per-judge G-Eval tables (JSON)
- `tables/` — CSV summaries (win rates, Bradley–Terry)
- `reports/` — Markdown / LaTeX for reading and papers

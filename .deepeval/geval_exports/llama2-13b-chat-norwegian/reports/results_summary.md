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
| GPT4o-mini | 0.840 | 0.840 | 0.920 | 0.920 |
| checkpoint-1000-gen0-inputs-refs-preds-1000-examples | 0.673 | 0.673 | 0.692 | 0.769 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.346 | 0.327 | 0.250 | 0.385 |
| checkpoint-10000-gen0-inputs-refs-preds-1000-examples | 0.760 | 0.640 | 0.700 | 0.640 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.327 | 0.327 | 0.346 | 0.269 |
| checkpoint-1500-gen0-inputs-refs-preds-1000-examples | 0.635 | 0.635 | 0.692 | 0.635 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.380 | 0.380 | 0.480 | 0.340 |
| checkpoint-2000-gen0-inputs-refs-preds-1000-examples | 0.558 | 0.577 | 0.596 | 0.635 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.260 | 0.320 | 0.380 | 0.300 |
| checkpoint-2500-gen0-inputs-refs-preds-1000-examples | 0.780 | 0.520 | 0.540 | 0.580 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.140 | 0.100 | 0.180 | 0.160 |
| checkpoint-3000-gen0-inputs-refs-preds-1000-examples | 0.558 | 0.673 | 0.558 | 0.615 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.320 | 0.420 | 0.240 | 0.200 |
| checkpoint-3500-gen0-inputs-refs-preds-1000-examples | 0.615 | 0.442 | 0.577 | 0.519 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.420 | 0.360 | 0.400 | 0.380 |
| checkpoint-4000-gen0-inputs-refs-preds-1000-examples | 0.615 | 0.538 | 0.615 | 0.673 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.365 | 0.385 | 0.327 | 0.308 |
| checkpoint-4500-gen0-inputs-refs-preds-1000-examples | 0.540 | 0.700 | 0.560 | 0.660 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.200 | 0.180 | 0.300 | 0.300 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 0.560 | 0.420 | 0.460 | 0.560 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.288 | 0.288 | 0.327 | 0.346 |
| checkpoint-5000-gen0-inputs-refs-preds-1000-examples | 0.620 | 0.660 | 0.660 | 0.640 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.240 | 0.280 | 0.260 | 0.180 |
| checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.680 | 0.720 | 0.740 | 0.720 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.420 | 0.520 | 0.400 | 0.380 |
| checkpoint-6000-gen0-inputs-refs-preds-1000-examples | 0.640 | 0.660 | 0.620 | 0.720 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.440 | 0.500 | 0.340 | 0.420 |
| checkpoint-6500-gen0-inputs-refs-preds-1000-examples | 0.700 | 0.680 | 0.700 | 0.740 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.460 | 0.360 | 0.400 | 0.540 |
| checkpoint-7000-gen0-inputs-refs-preds-1000-examples | 0.673 | 0.673 | 0.654 | 0.635 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.340 | 0.480 | 0.440 | 0.420 |
| checkpoint-7500-gen0-inputs-refs-preds-1000-examples | 0.788 | 0.750 | 0.519 | 0.615 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.520 | 0.540 | 0.500 | 0.540 |
| checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.540 | 0.480 | 0.660 | 0.680 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.480 | 0.520 | 0.480 | 0.360 |
| checkpoint-8500-gen0-inputs-refs-preds-1000-examples | 0.423 | 0.462 | 0.519 | 0.519 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.360 | 0.440 | 0.420 | 0.320 |
| checkpoint-9000-gen0-inputs-refs-preds-1000-examples | 0.481 | 0.481 | 0.423 | 0.519 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.519 | 0.615 | 0.596 | 0.500 |
| checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.700 | 0.680 | 0.680 | 0.600 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.280 | 0.240 | 0.340 | 0.240 |

### Consistency

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.840 | 0.840 | 0.800 | 0.900 |
| checkpoint-1000-gen0-inputs-refs-preds-1000-examples | 0.615 | 0.654 | 0.615 | 0.654 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.288 | 0.231 | 0.308 | 0.327 |
| checkpoint-10000-gen0-inputs-refs-preds-1000-examples | 0.660 | 0.700 | 0.800 | 0.660 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.212 | 0.231 | 0.288 | 0.288 |
| checkpoint-1500-gen0-inputs-refs-preds-1000-examples | 0.596 | 0.596 | 0.750 | 0.692 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.340 | 0.420 | 0.320 | 0.300 |
| checkpoint-2000-gen0-inputs-refs-preds-1000-examples | 0.519 | 0.615 | 0.673 | 0.577 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.280 | 0.220 | 0.240 | 0.180 |
| checkpoint-2500-gen0-inputs-refs-preds-1000-examples | 0.620 | 0.500 | 0.600 | 0.600 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.200 | 0.300 | 0.140 | 0.120 |
| checkpoint-3000-gen0-inputs-refs-preds-1000-examples | 0.577 | 0.596 | 0.538 | 0.692 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.300 | 0.380 | 0.220 | 0.200 |
| checkpoint-3500-gen0-inputs-refs-preds-1000-examples | 0.577 | 0.558 | 0.500 | 0.577 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.420 | 0.440 | 0.420 | 0.460 |
| checkpoint-4000-gen0-inputs-refs-preds-1000-examples | 0.673 | 0.615 | 0.654 | 0.558 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.404 | 0.288 | 0.346 | 0.404 |
| checkpoint-4500-gen0-inputs-refs-preds-1000-examples | 0.680 | 0.580 | 0.660 | 0.560 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.260 | 0.320 | 0.360 | 0.360 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 0.620 | 0.640 | 0.600 | 0.660 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.288 | 0.327 | 0.288 | 0.327 |
| checkpoint-5000-gen0-inputs-refs-preds-1000-examples | 0.760 | 0.700 | 0.700 | 0.740 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.220 | 0.300 | 0.280 | 0.220 |
| checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.720 | 0.760 | 0.820 | 0.820 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.360 | 0.280 | 0.260 |
| checkpoint-6000-gen0-inputs-refs-preds-1000-examples | 0.660 | 0.700 | 0.680 | 0.660 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.360 | 0.500 | 0.380 | 0.320 |
| checkpoint-6500-gen0-inputs-refs-preds-1000-examples | 0.720 | 0.580 | 0.600 | 0.700 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.380 | 0.340 | 0.340 | 0.500 |
| checkpoint-7000-gen0-inputs-refs-preds-1000-examples | 0.654 | 0.577 | 0.635 | 0.673 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.480 | 0.440 | 0.480 | 0.400 |
| checkpoint-7500-gen0-inputs-refs-preds-1000-examples | 0.673 | 0.654 | 0.635 | 0.635 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.520 | 0.520 | 0.480 | 0.520 |
| checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.640 | 0.620 | 0.580 | 0.740 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.300 | 0.340 | 0.440 | 0.420 |
| checkpoint-8500-gen0-inputs-refs-preds-1000-examples | 0.615 | 0.673 | 0.596 | 0.538 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.300 | 0.340 | 0.420 | 0.420 |
| checkpoint-9000-gen0-inputs-refs-preds-1000-examples | 0.654 | 0.596 | 0.519 | 0.577 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.481 | 0.481 | 0.462 | 0.385 |
| checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.640 | 0.740 | 0.660 | 0.540 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.240 | 0.220 | 0.380 | 0.320 |

### Newsworthiness

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.760 | 0.860 | 0.900 | 0.840 |
| checkpoint-1000-gen0-inputs-refs-preds-1000-examples | 0.635 | 0.635 | 0.769 | 0.635 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.385 | 0.442 | 0.365 | 0.385 |
| checkpoint-10000-gen0-inputs-refs-preds-1000-examples | 0.720 | 0.580 | 0.700 | 0.580 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.423 | 0.385 | 0.481 | 0.481 |
| checkpoint-1500-gen0-inputs-refs-preds-1000-examples | 0.654 | 0.769 | 0.712 | 0.712 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.400 | 0.380 | 0.400 | 0.440 |
| checkpoint-2000-gen0-inputs-refs-preds-1000-examples | 0.615 | 0.500 | 0.538 | 0.577 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.380 | 0.460 | 0.600 | 0.560 |
| checkpoint-2500-gen0-inputs-refs-preds-1000-examples | 0.720 | 0.580 | 0.700 | 0.680 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.120 | 0.340 | 0.240 | 0.260 |
| checkpoint-3000-gen0-inputs-refs-preds-1000-examples | 0.635 | 0.692 | 0.635 | 0.692 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.400 | 0.480 | 0.340 | 0.360 |
| checkpoint-3500-gen0-inputs-refs-preds-1000-examples | 0.654 | 0.558 | 0.615 | 0.596 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.340 | 0.340 | 0.300 | 0.360 |
| checkpoint-4000-gen0-inputs-refs-preds-1000-examples | 0.558 | 0.423 | 0.538 | 0.615 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.404 | 0.404 | 0.327 | 0.308 |
| checkpoint-4500-gen0-inputs-refs-preds-1000-examples | 0.500 | 0.600 | 0.420 | 0.540 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.360 | 0.300 | 0.280 | 0.400 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 0.460 | 0.480 | 0.520 | 0.500 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.231 | 0.442 | 0.288 | 0.308 |
| checkpoint-5000-gen0-inputs-refs-preds-1000-examples | 0.540 | 0.460 | 0.580 | 0.580 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.160 | 0.280 | 0.200 | 0.320 |
| checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.640 | 0.500 | 0.660 | 0.560 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.300 | 0.400 | 0.320 | 0.360 |
| checkpoint-6000-gen0-inputs-refs-preds-1000-examples | 0.660 | 0.620 | 0.680 | 0.600 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.380 | 0.340 | 0.420 | 0.360 |
| checkpoint-6500-gen0-inputs-refs-preds-1000-examples | 0.700 | 0.760 | 0.620 | 0.600 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.420 | 0.400 | 0.480 | 0.560 |
| checkpoint-7000-gen0-inputs-refs-preds-1000-examples | 0.692 | 0.654 | 0.692 | 0.654 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.440 | 0.400 | 0.440 | 0.440 |
| checkpoint-7500-gen0-inputs-refs-preds-1000-examples | 0.519 | 0.519 | 0.538 | 0.481 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.420 | 0.460 | 0.420 |
| checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.660 | 0.520 | 0.560 | 0.540 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.440 | 0.460 | 0.540 |
| checkpoint-8500-gen0-inputs-refs-preds-1000-examples | 0.442 | 0.423 | 0.519 | 0.423 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.240 | 0.380 | 0.360 | 0.300 |
| checkpoint-9000-gen0-inputs-refs-preds-1000-examples | 0.500 | 0.519 | 0.404 | 0.500 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.654 | 0.596 | 0.519 | 0.442 |
| checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.720 | 0.660 | 0.600 | 0.600 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.460 | 0.540 | 0.300 | 0.380 |

### Hygiene

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.900 | 0.940 | 0.940 | 0.900 |
| checkpoint-1000-gen0-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.577 | 0.615 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.558 | 0.442 | 0.423 | 0.365 |
| checkpoint-10000-gen0-inputs-refs-preds-1000-examples | 0.740 | 0.660 | 0.640 | 0.560 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.385 | 0.288 | 0.365 | 0.481 |
| checkpoint-1500-gen0-inputs-refs-preds-1000-examples | 0.692 | 0.731 | 0.692 | 0.808 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.260 | 0.240 | 0.520 | 0.480 |
| checkpoint-2000-gen0-inputs-refs-preds-1000-examples | 0.635 | 0.596 | 0.500 | 0.596 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.360 | 0.380 | 0.580 | 0.400 |
| checkpoint-2500-gen0-inputs-refs-preds-1000-examples | 0.760 | 0.620 | 0.740 | 0.680 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.140 | 0.160 | 0.100 | 0.140 |
| checkpoint-3000-gen0-inputs-refs-preds-1000-examples | 0.615 | 0.558 | 0.673 | 0.558 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.300 | 0.340 | 0.260 | 0.320 |
| checkpoint-3500-gen0-inputs-refs-preds-1000-examples | 0.615 | 0.558 | 0.712 | 0.577 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.400 | 0.500 | 0.320 | 0.400 |
| checkpoint-4000-gen0-inputs-refs-preds-1000-examples | 0.692 | 0.692 | 0.654 | 0.654 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.346 | 0.404 | 0.327 | 0.327 |
| checkpoint-4500-gen0-inputs-refs-preds-1000-examples | 0.700 | 0.660 | 0.520 | 0.600 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.300 | 0.340 | 0.360 | 0.340 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 0.620 | 0.580 | 0.420 | 0.520 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.365 | 0.538 | 0.365 | 0.288 |
| checkpoint-5000-gen0-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.480 | 0.560 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.240 | 0.300 | 0.260 | 0.200 |
| checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.600 | 0.640 | 0.580 | 0.660 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.220 | 0.180 | 0.340 | 0.280 |
| checkpoint-6000-gen0-inputs-refs-preds-1000-examples | 0.660 | 0.660 | 0.560 | 0.660 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.380 | 0.480 | 0.360 | 0.380 |
| checkpoint-6500-gen0-inputs-refs-preds-1000-examples | 0.720 | 0.660 | 0.640 | 0.600 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.380 | 0.420 | 0.380 | 0.340 |
| checkpoint-7000-gen0-inputs-refs-preds-1000-examples | 0.654 | 0.500 | 0.673 | 0.673 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.400 | 0.520 | 0.540 | 0.520 |
| checkpoint-7500-gen0-inputs-refs-preds-1000-examples | 0.673 | 0.615 | 0.615 | 0.596 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.480 | 0.420 | 0.520 | 0.480 |
| checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.680 | 0.580 | 0.600 | 0.620 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.360 | 0.340 | 0.420 | 0.520 |
| checkpoint-8500-gen0-inputs-refs-preds-1000-examples | 0.404 | 0.442 | 0.404 | 0.462 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.400 | 0.440 | 0.420 | 0.300 |
| checkpoint-9000-gen0-inputs-refs-preds-1000-examples | 0.442 | 0.423 | 0.442 | 0.500 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.481 | 0.481 | 0.538 | 0.519 |
| checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.640 | 0.660 | 0.640 | 0.640 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.280 | 0.500 | 0.380 | 0.360 |

---

## 2. Bradley–Terry strengths (θ)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Relevance

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 6.9298 | 7.1566 | 15.2189 | 16.3930 |
| checkpoint-1000-gen0-inputs-refs-preds-1000-examples | 2.0394 | 1.8705 | 2.0117 | 3.3547 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.5474 | 0.5145 | 0.3575 | 0.6742 |
| checkpoint-10000-gen0-inputs-refs-preds-1000-examples | 3.4282 | 1.8626 | 2.3093 | 1.7664 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.3927 | 0.3976 | 0.4338 | 0.2800 |
| checkpoint-1500-gen0-inputs-refs-preds-1000-examples | 1.9190 | 2.0582 | 2.5328 | 1.9594 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.5413 | 0.5306 | 0.9380 | 0.4958 |
| checkpoint-2000-gen0-inputs-refs-preds-1000-examples | 1.3375 | 1.5071 | 1.5459 | 1.8831 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.3799 | 0.5021 | 0.6576 | 0.4617 |
| checkpoint-2500-gen0-inputs-refs-preds-1000-examples | 3.4896 | 0.9203 | 0.9752 | 1.1722 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.1826 | 0.1151 | 0.2435 | 0.1875 |
| checkpoint-3000-gen0-inputs-refs-preds-1000-examples | 1.2470 | 2.2734 | 1.3261 | 1.7941 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.3976 | 0.7195 | 0.2807 | 0.1972 |
| checkpoint-3500-gen0-inputs-refs-preds-1000-examples | 1.6105 | 0.7261 | 1.3917 | 1.1382 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.8060 | 0.5790 | 0.6720 | 0.6534 |
| checkpoint-4000-gen0-inputs-refs-preds-1000-examples | 1.8420 | 1.1936 | 1.7387 | 2.2810 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.5025 | 0.5388 | 0.4153 | 0.3670 |
| checkpoint-4500-gen0-inputs-refs-preds-1000-examples | 1.2541 | 2.7163 | 1.3694 | 2.2930 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.2275 | 0.2086 | 0.3897 | 0.3755 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 1.2274 | 0.6851 | 0.7628 | 1.2982 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.4069 | 0.4183 | 0.5096 | 0.5735 |
| checkpoint-5000-gen0-inputs-refs-preds-1000-examples | 1.7786 | 2.3401 | 2.2341 | 2.0892 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.2588 | 0.3473 | 0.2846 | 0.1654 |
| checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 1.8932 | 2.5331 | 2.8254 | 2.4927 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.6176 | 0.9456 | 0.5966 | 0.4831 |
| checkpoint-6000-gen0-inputs-refs-preds-1000-examples | 1.7561 | 2.0699 | 1.7391 | 3.0816 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.7915 | 0.9906 | 0.4973 | 0.6885 |
| checkpoint-6500-gen0-inputs-refs-preds-1000-examples | 2.6532 | 2.3739 | 2.4706 | 3.1218 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.7817 | 0.4756 | 0.6073 | 1.1551 |
| checkpoint-7000-gen0-inputs-refs-preds-1000-examples | 2.5317 | 2.2794 | 2.1761 | 2.0881 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.4632 | 0.8725 | 0.6888 | 0.6389 |
| checkpoint-7500-gen0-inputs-refs-preds-1000-examples | 4.7760 | 3.5658 | 1.1066 | 1.7874 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 1.1488 | 1.2481 | 0.9815 | 1.2213 |
| checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 1.1873 | 0.9504 | 2.1486 | 2.4584 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 1.0419 | 1.2135 | 1.0519 | 0.6207 |
| checkpoint-8500-gen0-inputs-refs-preds-1000-examples | 0.6905 | 0.8711 | 1.0156 | 1.0338 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.5653 | 0.8814 | 0.7043 | 0.4365 |
| checkpoint-9000-gen0-inputs-refs-preds-1000-examples | 0.8935 | 0.9525 | 0.6637 | 1.0714 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.9636 | 1.5623 | 1.3695 | 0.9368 |
| checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 2.4195 | 2.1521 | 2.0724 | 1.4326 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.3349 | 0.2759 | 0.4426 | 0.2266 |

### Consistency

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 7.3990 | 7.1536 | 5.8172 | 13.8810 |
| checkpoint-1000-gen0-inputs-refs-preds-1000-examples | 1.4959 | 1.8191 | 1.4342 | 1.7859 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.3972 | 0.2707 | 0.4481 | 0.5119 |
| checkpoint-10000-gen0-inputs-refs-preds-1000-examples | 1.9328 | 2.4147 | 4.3101 | 1.9706 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.2261 | 0.2456 | 0.3349 | 0.3149 |
| checkpoint-1500-gen0-inputs-refs-preds-1000-examples | 1.7185 | 1.6463 | 3.7159 | 2.6606 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.4693 | 0.6464 | 0.4121 | 0.3781 |
| checkpoint-2000-gen0-inputs-refs-preds-1000-examples | 1.1577 | 1.8413 | 2.4253 | 1.4991 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.3873 | 0.2698 | 0.2984 | 0.2134 |
| checkpoint-2500-gen0-inputs-refs-preds-1000-examples | 1.4276 | 0.7632 | 1.2430 | 1.2404 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.2420 | 0.4744 | 0.1683 | 0.1280 |
| checkpoint-3000-gen0-inputs-refs-preds-1000-examples | 1.4695 | 1.6290 | 1.1992 | 2.8137 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.3784 | 0.5800 | 0.2409 | 0.1933 |
| checkpoint-3500-gen0-inputs-refs-preds-1000-examples | 1.4895 | 1.2991 | 0.9670 | 1.4517 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.7623 | 0.8328 | 0.7294 | 0.8526 |
| checkpoint-4000-gen0-inputs-refs-preds-1000-examples | 2.2721 | 1.7630 | 2.0874 | 1.2566 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.5993 | 0.3309 | 0.4745 | 0.6277 |
| checkpoint-4500-gen0-inputs-refs-preds-1000-examples | 2.4632 | 1.5781 | 2.4326 | 1.5331 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.3272 | 0.4392 | 0.4944 | 0.4813 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 1.6900 | 1.8847 | 1.5187 | 2.0173 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.4308 | 0.5304 | 0.4149 | 0.5412 |
| checkpoint-5000-gen0-inputs-refs-preds-1000-examples | 4.1430 | 2.8490 | 2.7706 | 3.6734 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.2337 | 0.3814 | 0.3339 | 0.2226 |
| checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 2.3161 | 3.0238 | 4.5559 | 4.8319 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.9443 | 0.5086 | 0.3309 | 0.2799 |
| checkpoint-6000-gen0-inputs-refs-preds-1000-examples | 2.1603 | 2.4642 | 2.1727 | 2.2683 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.5074 | 1.0652 | 0.6101 | 0.4038 |
| checkpoint-6500-gen0-inputs-refs-preds-1000-examples | 2.6908 | 1.3606 | 1.4236 | 2.5485 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.5178 | 0.4231 | 0.4546 | 0.9478 |
| checkpoint-7000-gen0-inputs-refs-preds-1000-examples | 2.2288 | 1.4759 | 2.0026 | 2.5314 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.9398 | 0.7185 | 0.9005 | 0.6300 |
| checkpoint-7500-gen0-inputs-refs-preds-1000-examples | 2.6077 | 2.2576 | 2.1734 | 2.0286 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 1.1629 | 1.1499 | 0.8856 | 1.1406 |
| checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 1.9451 | 1.6987 | 1.3467 | 3.3059 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.4201 | 0.5116 | 0.8454 | 0.7599 |
| checkpoint-8500-gen0-inputs-refs-preds-1000-examples | 1.6804 | 2.1763 | 1.5078 | 1.1631 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.4440 | 0.5387 | 0.8047 | 0.7387 |
| checkpoint-9000-gen0-inputs-refs-preds-1000-examples | 2.0909 | 1.5585 | 1.1345 | 1.4740 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.7738 | 0.8106 | 0.7581 | 0.4919 |
| checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 1.7282 | 2.9537 | 2.0140 | 1.1729 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.2358 | 0.2290 | 0.5259 | 0.3506 |

### Newsworthiness

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 3.9349 | 7.2849 | 11.3579 | 6.1217 |
| checkpoint-1000-gen0-inputs-refs-preds-1000-examples | 1.6342 | 1.6242 | 3.1815 | 1.5690 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.7338 | 0.8559 | 0.6419 | 0.6766 |
| checkpoint-10000-gen0-inputs-refs-preds-1000-examples | 2.7037 | 1.3584 | 2.1518 | 1.2908 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.6771 | 0.5719 | 0.8292 | 0.8919 |
| checkpoint-1500-gen0-inputs-refs-preds-1000-examples | 2.0500 | 3.7645 | 2.8985 | 2.6843 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.6547 | 0.5860 | 0.6310 | 0.7781 |
| checkpoint-2000-gen0-inputs-refs-preds-1000-examples | 1.6116 | 0.9490 | 1.1128 | 1.3193 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.7537 | 0.9078 | 1.8697 | 1.5519 |
| checkpoint-2500-gen0-inputs-refs-preds-1000-examples | 2.4681 | 1.2432 | 2.2416 | 1.9884 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.1463 | 0.5382 | 0.3418 | 0.3702 |
| checkpoint-3000-gen0-inputs-refs-preds-1000-examples | 1.9548 | 2.4948 | 1.8967 | 2.4093 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.6090 | 0.9009 | 0.4606 | 0.5167 |
| checkpoint-3500-gen0-inputs-refs-preds-1000-examples | 1.9890 | 1.1817 | 1.6890 | 1.5210 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.5441 | 0.5015 | 0.4467 | 0.6076 |
| checkpoint-4000-gen0-inputs-refs-preds-1000-examples | 1.3652 | 0.6850 | 1.2860 | 1.7343 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.6065 | 0.5952 | 0.4104 | 0.4047 |
| checkpoint-4500-gen0-inputs-refs-preds-1000-examples | 1.0374 | 1.4103 | 0.6976 | 1.1874 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.5492 | 0.3933 | 0.3597 | 0.6729 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 0.7483 | 0.8635 | 0.9318 | 0.9156 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.3308 | 0.8707 | 0.4088 | 0.4492 |
| checkpoint-5000-gen0-inputs-refs-preds-1000-examples | 1.2805 | 0.8791 | 1.4860 | 1.4943 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.1594 | 0.3461 | 0.1991 | 0.3884 |
| checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 1.6853 | 0.9953 | 1.8093 | 1.2439 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.3743 | 0.6131 | 0.3991 | 0.5161 |
| checkpoint-6000-gen0-inputs-refs-preds-1000-examples | 2.0218 | 1.8460 | 2.3362 | 1.5644 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.5987 | 0.4838 | 0.7134 | 0.5606 |
| checkpoint-6500-gen0-inputs-refs-preds-1000-examples | 2.5489 | 3.4578 | 1.7672 | 1.5341 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.7311 | 0.6090 | 0.8905 | 1.2800 |
| checkpoint-7000-gen0-inputs-refs-preds-1000-examples | 2.5237 | 1.8895 | 2.7580 | 2.1438 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.7224 | 0.6442 | 0.6901 | 0.7085 |
| checkpoint-7500-gen0-inputs-refs-preds-1000-examples | 1.1031 | 1.1085 | 1.1899 | 0.9271 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 1.0630 | 0.7456 | 0.8298 | 0.7315 |
| checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 2.0620 | 1.0818 | 1.3335 | 1.2155 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 1.2326 | 0.8368 | 0.9520 | 1.3094 |
| checkpoint-8500-gen0-inputs-refs-preds-1000-examples | 0.7504 | 0.6866 | 1.0856 | 0.7259 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.2894 | 0.6250 | 0.5231 | 0.3885 |
| checkpoint-9000-gen0-inputs-refs-preds-1000-examples | 1.0080 | 1.0321 | 0.6600 | 0.9590 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 1.9706 | 1.4779 | 1.0244 | 0.7805 |
| checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 2.4525 | 1.8053 | 1.3314 | 1.3339 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.7828 | 1.1748 | 0.3647 | 0.5402 |

### Hygiene

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 12.8439 | 20.6274 | 20.0626 | 11.8805 |
| checkpoint-1000-gen0-inputs-refs-preds-1000-examples | 0.9049 | 0.9024 | 1.1837 | 1.4845 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 1.4519 | 0.7991 | 0.8485 | 0.6289 |
| checkpoint-10000-gen0-inputs-refs-preds-1000-examples | 3.3539 | 2.1637 | 1.7847 | 1.1760 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.5300 | 0.3385 | 0.4826 | 0.8422 |
| checkpoint-1500-gen0-inputs-refs-preds-1000-examples | 2.5708 | 3.0712 | 2.6050 | 5.1795 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.2894 | 0.2649 | 1.0475 | 0.8982 |
| checkpoint-2000-gen0-inputs-refs-preds-1000-examples | 1.9041 | 1.5018 | 0.9626 | 1.5093 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.6343 | 0.6237 | 1.7591 | 0.8032 |
| checkpoint-2500-gen0-inputs-refs-preds-1000-examples | 2.8971 | 1.3851 | 2.7841 | 1.9796 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.1558 | 0.1826 | 0.1126 | 0.1750 |
| checkpoint-3000-gen0-inputs-refs-preds-1000-examples | 1.7086 | 1.2489 | 2.0923 | 1.2764 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.3562 | 0.4884 | 0.3204 | 0.4164 |
| checkpoint-3500-gen0-inputs-refs-preds-1000-examples | 1.7317 | 1.1972 | 2.7421 | 1.3721 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.7212 | 1.0506 | 0.4518 | 0.7100 |
| checkpoint-4000-gen0-inputs-refs-preds-1000-examples | 2.4859 | 2.2840 | 1.9917 | 2.1236 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.4391 | 0.5885 | 0.4375 | 0.4220 |
| checkpoint-4500-gen0-inputs-refs-preds-1000-examples | 2.8357 | 2.1969 | 1.1041 | 1.6850 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.3560 | 0.4452 | 0.5274 | 0.4862 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 1.5357 | 1.3882 | 0.5987 | 1.0094 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.5865 | 1.2699 | 0.5934 | 0.4021 |
| checkpoint-5000-gen0-inputs-refs-preds-1000-examples | 1.1268 | 1.0878 | 1.0666 | 1.4755 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.2420 | 0.3758 | 0.2823 | 0.1983 |
| checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 1.3716 | 1.7283 | 1.2917 | 1.8366 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.2143 | 0.1790 | 0.4362 | 0.3370 |
| checkpoint-6000-gen0-inputs-refs-preds-1000-examples | 2.0324 | 2.0429 | 1.3076 | 2.0474 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.5684 | 0.9717 | 0.4990 | 0.5635 |
| checkpoint-6500-gen0-inputs-refs-preds-1000-examples | 2.7346 | 2.0120 | 1.9037 | 1.5228 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.5729 | 0.7189 | 0.6044 | 0.4801 |
| checkpoint-7000-gen0-inputs-refs-preds-1000-examples | 2.2859 | 0.9914 | 2.4876 | 2.4441 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.6398 | 1.1468 | 1.1030 | 1.0231 |
| checkpoint-7500-gen0-inputs-refs-preds-1000-examples | 2.5003 | 1.8286 | 1.7105 | 1.6138 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 1.0113 | 0.7014 | 1.1418 | 0.9558 |
| checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 2.3666 | 1.3753 | 1.6405 | 1.6680 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.6159 | 0.5444 | 0.7936 | 1.2178 |
| checkpoint-8500-gen0-inputs-refs-preds-1000-examples | 0.6100 | 0.7191 | 0.6601 | 0.8612 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.6542 | 0.7965 | 0.6569 | 0.3882 |
| checkpoint-9000-gen0-inputs-refs-preds-1000-examples | 0.8098 | 0.7040 | 0.7757 | 0.9695 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.8390 | 0.8849 | 1.1196 | 1.0180 |
| checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 1.7308 | 1.9147 | 1.5570 | 1.6109 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.2969 | 0.9889 | 0.5098 | 0.4748 |

---

## Export layout

- `json/` — pairwise rows and per-judge G-Eval tables (JSON)
- `tables/` — CSV summaries (win rates, Bradley–Terry)
- `reports/` — Markdown / LaTeX for reading and papers

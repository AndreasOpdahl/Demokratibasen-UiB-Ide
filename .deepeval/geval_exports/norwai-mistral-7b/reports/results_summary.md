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
| GPT4o-mini | 0.940 | 0.940 | 0.980 | 0.980 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.520 | 0.520 | 0.560 | 0.400 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.596 | 0.481 | 0.558 | 0.538 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.615 | 0.558 | 0.462 | 0.538 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.560 | 0.580 | 0.540 | 0.520 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.440 | 0.460 | 0.320 | 0.340 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.560 | 0.520 | 0.580 | 0.660 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.520 | 0.480 | 0.460 | 0.460 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.620 | 0.540 | 0.540 | 0.440 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.240 | 0.360 | 0.340 | 0.360 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.560 | 0.600 | 0.540 | 0.640 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.400 | 0.360 | 0.400 | 0.380 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.640 | 0.660 | 0.660 | 0.620 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.462 | 0.385 | 0.615 | 0.385 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.500 | 0.423 | 0.500 | 0.596 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.519 | 0.500 | 0.538 | 0.519 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.480 | 0.440 | 0.480 | 0.520 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.577 | 0.423 | 0.385 | 0.404 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.500 | 0.480 | 0.520 | 0.560 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 0.538 | 0.519 | 0.558 | 0.615 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.540 | 0.460 | 0.400 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.240 | 0.440 | 0.360 | 0.440 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.540 | 0.520 | 0.540 | 0.620 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.365 | 0.327 | 0.385 | 0.365 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.577 | 0.481 | 0.500 | 0.558 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.340 | 0.360 | 0.340 | 0.360 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.480 | 0.540 | 0.500 | 0.580 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.442 | 0.327 | 0.327 | 0.327 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.420 | 0.480 | 0.520 | 0.580 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.538 | 0.442 | 0.442 | 0.346 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.420 | 0.500 | 0.400 | 0.460 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.420 | 0.400 | 0.480 | 0.380 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.385 | 0.538 | 0.538 | 0.538 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.440 | 0.560 | 0.400 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.620 | 0.720 | 0.680 | 0.660 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.404 | 0.558 | 0.346 | 0.385 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.460 | 0.520 | 0.620 | 0.580 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.360 | 0.300 | 0.260 | 0.360 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.673 | 0.731 | 0.750 | 0.712 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.442 | 0.462 | 0.404 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.520 | 0.680 | 0.500 | 0.580 |

### Consistency

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.900 | 0.920 | 1.000 | 0.920 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.460 | 0.440 | 0.380 | 0.400 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.558 | 0.577 | 0.577 | 0.558 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.404 | 0.385 | 0.462 | 0.365 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.780 | 0.780 | 0.700 | 0.820 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.320 | 0.400 | 0.280 | 0.300 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.740 | 0.680 | 0.700 | 0.760 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.480 | 0.380 | 0.440 | 0.300 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.620 | 0.640 | 0.600 | 0.620 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.240 | 0.260 | 0.300 | 0.280 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.640 | 0.620 | 0.740 | 0.740 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.240 | 0.220 | 0.260 | 0.240 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.540 | 0.660 | 0.580 | 0.740 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.538 | 0.365 | 0.462 | 0.462 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.577 | 0.635 | 0.654 | 0.577 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.385 | 0.365 | 0.404 | 0.404 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.660 | 0.660 | 0.640 | 0.560 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.365 | 0.385 | 0.346 | 0.404 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.620 | 0.640 | 0.660 | 0.620 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 0.596 | 0.596 | 0.692 | 0.788 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.320 | 0.320 | 0.260 | 0.300 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.120 | 0.200 | 0.200 | 0.260 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.700 | 0.640 | 0.680 | 0.680 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.404 | 0.365 | 0.404 | 0.442 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.673 | 0.654 | 0.596 | 0.615 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.220 | 0.160 | 0.300 | 0.280 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.640 | 0.600 | 0.680 | 0.580 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.269 | 0.288 | 0.212 | 0.269 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.580 | 0.660 | 0.620 | 0.580 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.365 | 0.462 | 0.346 | 0.308 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.560 | 0.580 | 0.600 | 0.560 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.260 | 0.180 | 0.300 | 0.240 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.577 | 0.692 | 0.596 | 0.577 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.360 | 0.400 | 0.480 | 0.420 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.660 | 0.620 | 0.580 | 0.680 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.308 | 0.481 | 0.231 | 0.269 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.720 | 0.620 | 0.660 | 0.580 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.240 | 0.140 | 0.240 | 0.260 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.788 | 0.712 | 0.788 | 0.750 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.346 | 0.365 | 0.269 | 0.308 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.740 | 0.760 | 0.600 | 0.700 |

### Newsworthiness

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.700 | 0.820 | 0.940 | 0.840 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.480 | 0.440 | 0.480 | 0.440 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.673 | 0.673 | 0.577 | 0.558 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.654 | 0.750 | 0.519 | 0.673 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.500 | 0.420 | 0.520 | 0.400 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.560 | 0.660 | 0.500 | 0.480 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.600 | 0.500 | 0.480 | 0.460 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.620 | 0.600 | 0.640 | 0.580 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.580 | 0.480 | 0.520 | 0.400 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.400 | 0.480 | 0.480 | 0.440 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.600 | 0.400 | 0.600 | 0.580 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.640 | 0.380 | 0.560 | 0.580 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.640 | 0.640 | 0.680 | 0.560 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.462 | 0.404 | 0.462 | 0.423 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.500 | 0.346 | 0.519 | 0.442 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.423 | 0.538 | 0.538 | 0.558 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.380 | 0.420 | 0.480 | 0.460 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.538 | 0.558 | 0.481 | 0.500 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.400 | 0.420 | 0.440 | 0.320 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 0.269 | 0.346 | 0.385 | 0.423 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.520 | 0.500 | 0.540 | 0.520 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.520 | 0.600 | 0.300 | 0.660 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.520 | 0.460 | 0.660 | 0.460 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.365 | 0.423 | 0.404 | 0.538 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.462 | 0.481 | 0.385 | 0.442 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.260 | 0.480 | 0.380 | 0.440 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.360 | 0.420 | 0.460 | 0.380 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.519 | 0.577 | 0.423 | 0.442 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.340 | 0.420 | 0.460 | 0.380 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.577 | 0.481 | 0.442 | 0.577 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.480 | 0.340 | 0.420 | 0.440 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.580 | 0.580 | 0.420 | 0.560 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.327 | 0.385 | 0.519 | 0.404 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.660 | 0.540 | 0.580 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.460 | 0.520 | 0.500 | 0.440 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.538 | 0.462 | 0.423 | 0.519 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.520 | 0.340 | 0.440 | 0.620 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.560 | 0.640 | 0.440 | 0.560 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.481 | 0.481 | 0.596 | 0.442 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.596 | 0.462 | 0.577 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.500 | 0.380 | 0.500 | 0.400 |

### Hygiene

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.980 | 0.900 | 0.980 | 1.000 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.400 | 0.460 | 0.440 | 0.340 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.635 | 0.519 | 0.442 | 0.423 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.365 | 0.500 | 0.365 | 0.365 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.520 | 0.500 | 0.480 | 0.460 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.260 | 0.260 | 0.340 | 0.300 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.500 | 0.420 | 0.460 | 0.480 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.760 | 0.580 | 0.560 | 0.640 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.500 | 0.540 | 0.500 | 0.440 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.380 | 0.420 | 0.420 | 0.540 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.620 | 0.460 | 0.480 | 0.560 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.280 | 0.260 | 0.420 | 0.580 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.600 | 0.600 | 0.520 | 0.600 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.538 | 0.346 | 0.654 | 0.462 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.462 | 0.538 | 0.481 | 0.288 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.346 | 0.500 | 0.692 | 0.596 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.440 | 0.500 | 0.440 | 0.440 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.654 | 0.596 | 0.500 | 0.635 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.520 | 0.500 | 0.460 | 0.520 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 0.635 | 0.500 | 0.500 | 0.596 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.560 | 0.560 | 0.560 | 0.520 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.480 | 0.580 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.500 | 0.600 | 0.580 | 0.600 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.462 | 0.538 | 0.519 | 0.365 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.423 | 0.577 | 0.365 | 0.462 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.440 | 0.440 | 0.360 | 0.460 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.500 | 0.400 | 0.560 | 0.580 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.423 | 0.423 | 0.442 | 0.288 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.520 | 0.540 | 0.440 | 0.480 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.519 | 0.538 | 0.558 | 0.500 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.500 | 0.540 | 0.520 | 0.600 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.300 | 0.340 | 0.420 | 0.400 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.462 | 0.577 | 0.538 | 0.404 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.440 | 0.420 | 0.600 | 0.500 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.580 | 0.580 | 0.520 | 0.500 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.538 | 0.481 | 0.615 | 0.500 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.480 | 0.520 | 0.520 | 0.560 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.320 | 0.400 | 0.260 | 0.460 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.673 | 0.615 | 0.596 | 0.500 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.385 | 0.346 | 0.385 | 0.462 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.580 | 0.660 | 0.520 | 0.540 |

---

## 2. Bradley–Terry strengths (θ)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Relevance

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 16.3558 | 16.6545 | 50.2034 | 53.8362 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.9379 | 0.9174 | 1.1073 | 0.5411 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 1.4914 | 0.8873 | 1.2520 | 1.1108 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 1.6903 | 1.2090 | 0.8518 | 1.1451 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.2831 | 1.3842 | 1.1630 | 1.0198 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.7276 | 0.7209 | 0.3938 | 0.4290 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.2296 | 1.0478 | 1.3742 | 1.9654 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 1.0591 | 0.8725 | 0.8024 | 0.7564 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.6481 | 1.1556 | 1.1781 | 0.7432 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.2844 | 0.4841 | 0.4564 | 0.4805 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.2165 | 1.5152 | 1.1312 | 1.8016 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.6348 | 0.5360 | 0.6422 | 0.5792 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.7525 | 1.9757 | 1.8991 | 1.6705 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.7633 | 0.5397 | 1.4134 | 0.5443 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.9932 | 0.7204 | 0.9596 | 1.4276 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 1.1349 | 0.9835 | 1.1625 | 1.0529 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.9327 | 0.8950 | 0.9521 | 1.2348 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 1.4656 | 0.7369 | 0.5788 | 0.6390 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.0370 | 0.9243 | 1.0605 | 1.2781 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 1.0905 | 1.0325 | 1.0979 | 1.4576 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 1.0060 | 1.2248 | 0.8538 | 0.6903 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.3041 | 0.7139 | 0.4921 | 0.7027 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.1412 | 1.0669 | 1.1991 | 1.6012 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.5228 | 0.4319 | 0.5606 | 0.4977 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 1.4177 | 0.8880 | 0.9076 | 1.1684 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.4431 | 0.4856 | 0.4288 | 0.4632 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.9223 | 1.2230 | 0.9486 | 1.5014 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.7946 | 0.4655 | 0.4564 | 0.4453 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.7967 | 0.9817 | 1.2239 | 1.5606 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 1.1491 | 0.8323 | 0.7520 | 0.5063 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.6669 | 0.9704 | 0.6244 | 0.8276 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.7356 | 0.6572 | 0.8183 | 0.5482 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.6021 | 1.1572 | 1.2278 | 1.1023 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 1.0512 | 0.8323 | 1.4102 | 0.6818 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.6897 | 2.6679 | 2.2580 | 1.9209 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.6894 | 1.3163 | 0.5267 | 0.6001 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.8209 | 1.0220 | 1.4979 | 1.2862 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.4879 | 0.3992 | 0.3240 | 0.5365 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 2.0251 | 2.6087 | 2.6738 | 2.3002 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.9520 | 0.7361 | 0.7855 | 0.5722 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 1.1737 | 2.2671 | 0.9477 | 1.4050 |

### Consistency

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 14.0112 | 18.5636 | 513683.0556 | 17.4972 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.6175 | 0.5651 | 0.3277 | 0.4297 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 1.2778 | 1.3742 | 1.0554 | 1.1269 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.5767 | 0.5135 | 0.6131 | 0.4511 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 4.6360 | 4.6640 | 2.0983 | 6.1448 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.3460 | 0.5055 | 0.2033 | 0.2846 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 3.2890 | 2.4023 | 2.0753 | 3.5872 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.8296 | 0.5256 | 0.5105 | 0.3200 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.8890 | 2.0733 | 1.3092 | 1.9008 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.2594 | 0.2642 | 0.2728 | 0.3025 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 2.0202 | 1.9372 | 2.9271 | 3.6133 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.2664 | 0.2314 | 0.2511 | 0.2694 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.2080 | 2.2738 | 1.1642 | 3.3976 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 1.1002 | 0.4644 | 0.6118 | 0.7431 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.2497 | 1.7652 | 1.5146 | 1.2835 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.5510 | 0.5409 | 0.4805 | 0.6061 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 2.7272 | 2.7881 | 1.9563 | 1.6433 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.5360 | 0.5979 | 0.3694 | 0.6063 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 2.2042 | 2.2758 | 1.9095 | 1.9791 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 1.4227 | 1.4597 | 1.6891 | 3.9831 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.4924 | 0.5095 | 0.2663 | 0.4140 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.0998 | 0.1850 | 0.1529 | 0.2898 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 3.0167 | 2.1864 | 2.0440 | 2.7246 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.6188 | 0.4660 | 0.4469 | 0.6570 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 2.1715 | 1.9084 | 1.0615 | 1.6480 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.1790 | 0.1166 | 0.2203 | 0.2601 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 2.1411 | 1.8302 | 2.3014 | 1.6503 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.2972 | 0.3323 | 0.1655 | 0.3191 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 1.6962 | 2.4483 | 1.7694 | 1.7254 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.5360 | 0.9488 | 0.3736 | 0.3913 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.4111 | 1.5168 | 1.3713 | 1.3296 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.2972 | 0.1749 | 0.2606 | 0.2443 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 1.3581 | 2.4069 | 1.1739 | 1.3535 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.5407 | 0.6838 | 0.7520 | 0.7233 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 2.0958 | 1.8495 | 1.0776 | 2.4644 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.4322 | 1.0148 | 0.2117 | 0.3267 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 2.9076 | 1.8328 | 1.6843 | 1.4372 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.2694 | 0.1154 | 0.2262 | 0.3257 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 4.5956 | 3.1928 | 3.5414 | 3.6298 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.3957 | 0.4332 | 0.2047 | 0.3411 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 3.9503 | 4.5734 | 1.4709 | 3.1073 |

### Newsworthiness

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 2.3407 | 4.4178 | 15.1002 | 5.1921 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.8255 | 0.7589 | 0.8592 | 0.7553 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 2.1443 | 2.1703 | 1.4320 | 1.2790 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 1.9476 | 3.2587 | 1.0902 | 2.1410 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.0732 | 0.7048 | 1.0636 | 0.6735 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 1.2555 | 1.8430 | 0.9009 | 0.9247 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.3716 | 1.0042 | 0.8785 | 0.8202 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 1.6587 | 1.5177 | 1.6717 | 1.3926 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.3511 | 0.9199 | 1.0703 | 0.6421 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.6118 | 0.8209 | 0.8060 | 0.7166 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.4561 | 0.6400 | 1.4597 | 1.3800 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 1.7922 | 0.6037 | 1.3397 | 1.3800 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.7731 | 1.7476 | 2.0981 | 1.3219 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.8599 | 0.6446 | 0.8070 | 0.7006 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.1276 | 0.5621 | 1.1255 | 0.8561 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.7448 | 1.3396 | 1.1598 | 1.2996 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.6044 | 0.6667 | 0.9567 | 0.8477 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 1.2335 | 1.3011 | 0.8626 | 1.0241 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.6769 | 0.7056 | 0.7448 | 0.4569 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 0.3517 | 0.4880 | 0.5825 | 0.7048 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 1.1090 | 0.9552 | 1.1325 | 1.0962 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 1.0933 | 1.4623 | 0.3989 | 1.8651 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.0296 | 0.7881 | 1.8524 | 0.7774 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.5649 | 0.6798 | 0.6533 | 1.1204 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.9385 | 1.0429 | 0.5858 | 0.8567 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.3325 | 0.9401 | 0.5395 | 0.8179 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.5352 | 0.7037 | 0.8723 | 0.6028 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 1.0589 | 1.4516 | 0.6612 | 0.7763 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.5445 | 0.7216 | 0.9755 | 0.6230 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 1.3436 | 0.8814 | 0.7485 | 1.2726 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.8748 | 0.4789 | 0.6902 | 0.7761 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 1.4784 | 1.4584 | 0.7266 | 1.2752 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.5115 | 0.6459 | 1.1219 | 0.7407 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 1.1017 | 2.0931 | 1.2938 | 1.4854 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.8255 | 1.1747 | 1.0370 | 0.7636 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 1.1726 | 0.7943 | 0.7326 | 1.0835 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 1.0606 | 0.5179 | 0.7893 | 1.6956 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 1.2258 | 1.5948 | 0.7539 | 1.1829 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.9752 | 0.8607 | 1.4230 | 0.7326 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.9383 | 1.5713 | 0.7778 | 1.2726 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 1.0769 | 0.6144 | 0.9919 | 0.6651 |

### Hygiene

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 48.8736 | 9.4500 | 48.9009 | 119790.2710 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.5299 | 0.7166 | 0.7296 | 0.3464 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 1.7421 | 1.0912 | 0.7918 | 0.5981 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.5378 | 0.9984 | 0.5698 | 0.4686 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.0456 | 1.0678 | 0.9749 | 0.7252 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.2974 | 0.3175 | 0.4405 | 0.2928 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.9836 | 0.7351 | 0.8375 | 0.6644 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 2.9942 | 1.2968 | 1.2103 | 1.3789 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.9454 | 1.1919 | 1.0162 | 0.6576 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.5596 | 0.6505 | 0.6616 | 0.8549 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.6973 | 0.8723 | 0.9140 | 1.0134 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.3652 | 0.3464 | 0.6985 | 1.0888 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.4165 | 1.4960 | 1.0336 | 1.1915 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 1.0671 | 0.4957 | 1.6533 | 0.6318 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.8121 | 1.1240 | 0.9305 | 0.3105 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.4921 | 1.0272 | 2.1986 | 1.2036 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.8300 | 1.0701 | 0.7940 | 0.6596 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 2.0908 | 1.5185 | 0.9966 | 1.4857 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.0120 | 1.0141 | 0.8444 | 0.9129 |
| checkpoint-500-gen0-inputs-refs-preds-1000-examples | 1.6094 | 0.9354 | 0.8974 | 1.2153 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 1.2469 | 1.3337 | 1.1197 | 0.7793 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.8732 | 0.9509 | 0.8341 | 1.0625 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.8821 | 1.4000 | 1.2197 | 1.1499 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.7513 | 1.0451 | 0.9876 | 0.4023 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.7226 | 1.3639 | 0.5266 | 0.6872 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.6666 | 0.7047 | 0.5127 | 0.6718 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 1.1041 | 0.7484 | 1.2830 | 1.2517 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.6454 | 0.6955 | 0.6789 | 0.2899 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 1.1494 | 1.2147 | 0.8204 | 0.7995 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 1.1030 | 1.2416 | 1.1918 | 0.7780 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.9928 | 1.2043 | 1.0273 | 1.3553 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.4014 | 0.5320 | 0.6521 | 0.4829 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.7605 | 1.3108 | 1.1617 | 0.5276 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.7831 | 0.7354 | 1.5951 | 0.8045 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.3725 | 1.4101 | 1.0714 | 0.8011 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 1.2350 | 0.9067 | 1.6422 | 0.8210 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.9410 | 1.1530 | 1.0798 | 1.1157 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.4008 | 0.6341 | 0.3192 | 0.6473 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.9619 | 1.5249 | 1.3240 | 0.7040 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.5404 | 0.5118 | 0.5648 | 0.6144 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 1.5910 | 2.1511 | 1.0891 | 1.0232 |

---

## Export layout

- `json/` — pairwise rows and per-judge G-Eval tables (JSON)
- `tables/` — CSV summaries (win rates, Bradley–Terry)
- `reports/` — Markdown / LaTeX for reading and papers

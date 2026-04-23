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
| GPT4o-mini | 1.000 | 1.000 | 1.000 | 1.000 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.385 | 0.462 | 0.538 | 0.577 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.458 | 0.333 | 0.667 | 0.375 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.237 | 0.316 | 0.316 | 0.316 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.467 | 0.267 | 0.300 | 0.200 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.542 | 0.750 | 0.583 | 0.708 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.533 | 0.500 | 0.533 | 0.533 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.469 | 0.375 | 0.375 | 0.438 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.568 | 0.591 | 0.432 | 0.500 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.553 | 0.579 | 0.474 | 0.526 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.400 | 0.400 | 0.350 | 0.550 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.353 | 0.294 | 0.412 | 0.382 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.417 | 0.542 | 0.542 | 0.375 |

### Correctness

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.000 | 1.000 | 1.000 | 1.000 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.423 | 0.500 | 0.538 | 0.577 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.375 | 0.458 | 0.500 | 0.417 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.368 | 0.316 | 0.289 | 0.342 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.433 | 0.267 | 0.367 | 0.233 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.500 | 0.792 | 0.708 | 0.708 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.500 | 0.467 | 0.533 | 0.533 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.500 | 0.406 | 0.375 | 0.438 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.545 | 0.545 | 0.409 | 0.500 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.553 | 0.526 | 0.605 | 0.526 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.450 | 0.400 | 0.400 | 0.400 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.324 | 0.324 | 0.353 | 0.382 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.375 | 0.458 | 0.417 | 0.375 |

### Completeness

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.000 | 0.944 | 1.000 | 1.000 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.500 | 0.385 | 0.462 | 0.500 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.333 | 0.417 | 0.417 | 0.375 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.447 | 0.316 | 0.316 | 0.342 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.500 | 0.467 | 0.367 | 0.333 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.583 | 0.417 | 0.625 | 0.542 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.533 | 0.567 | 0.567 | 0.500 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.500 | 0.438 | 0.438 | 0.375 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.477 | 0.568 | 0.477 | 0.545 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.421 | 0.579 | 0.579 | 0.553 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.350 | 0.500 | 0.450 | 0.550 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.353 | 0.353 | 0.353 | 0.412 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.375 | 0.417 | 0.375 | 0.375 |

### Newsworthiness

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.944 | 0.944 | 1.000 | 1.000 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.423 | 0.462 | 0.385 | 0.423 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.375 | 0.292 | 0.375 | 0.375 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.395 | 0.316 | 0.368 | 0.421 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.433 | 0.467 | 0.467 | 0.400 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.458 | 0.417 | 0.583 | 0.500 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.600 | 0.567 | 0.533 | 0.467 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.438 | 0.500 | 0.438 | 0.438 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.545 | 0.591 | 0.455 | 0.500 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.500 | 0.526 | 0.553 | 0.526 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.450 | 0.600 | 0.500 | 0.600 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.412 | 0.382 | 0.353 | 0.353 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.375 | 0.292 | 0.417 | 0.417 |

### Hygiene

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.000 | 0.972 | 1.000 | 1.000 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.654 | 0.538 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.333 | 0.500 | 0.417 | 0.417 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.395 | 0.579 | 0.211 | 0.395 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.400 | 0.533 | 0.333 | 0.333 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.375 | 0.417 | 0.583 | 0.500 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.600 | 0.233 | 0.600 | 0.600 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.438 | 0.438 | 0.375 | 0.406 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.545 | 0.591 | 0.568 | 0.386 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.632 | 0.500 | 0.632 | 0.500 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.450 | 0.300 | 0.250 | 0.650 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.265 | 0.235 | 0.294 | 0.382 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.375 | 0.500 | 0.458 | 0.417 |

---

## 2. Bradley–Terry strengths (θ)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Faithfulness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 2068749.1855 | 867242.0908 | 2336649.2892 | 1065380.3214 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.2315 | 0.3607 | 0.5079 | 0.6004 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.3556 | 0.2299 | 1.0286 | 0.2428 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.1225 | 0.1803 | 0.1601 | 0.1637 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.4199 | 0.1668 | 0.1662 | 0.1122 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.4375 | 1.0459 | 0.3616 | 0.6991 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.4185 | 0.4018 | 0.3331 | 0.4239 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.3353 | 0.2500 | 0.1967 | 0.3036 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.4502 | 0.5403 | 0.2391 | 0.3627 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.3080 | 0.4223 | 0.2618 | 0.3796 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.2749 | 0.2619 | 0.1770 | 0.4434 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.2163 | 0.1858 | 0.2815 | 0.2833 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.2255 | 0.3964 | 0.4168 | 0.2253 |

### Correctness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1158222.4469 | 4204147.3762 | 3615322.6702 | 2892439.9194 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.2942 | 0.3536 | 0.5063 | 0.5737 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.2625 | 0.3687 | 0.3721 | 0.2743 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.2363 | 0.1577 | 0.1416 | 0.1814 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.3784 | 0.1531 | 0.2347 | 0.1293 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.3672 | 1.1070 | 0.6350 | 0.6770 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.3811 | 0.2852 | 0.3626 | 0.3990 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.4128 | 0.2225 | 0.2023 | 0.2773 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.4216 | 0.4199 | 0.2233 | 0.3289 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.3557 | 0.2860 | 0.4225 | 0.3482 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.3314 | 0.2228 | 0.2010 | 0.2110 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.2072 | 0.1678 | 0.2198 | 0.2558 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.2102 | 0.2396 | 0.2277 | 0.2023 |

### Completeness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 3177845.2637 | 13.4535 | 5904087.2388 | 11014844.7254 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.3454 | 0.6458 | 0.3120 | 0.3731 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.1991 | 0.7633 | 0.2501 | 0.2163 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.2952 | 0.4685 | 0.1565 | 0.1612 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.5167 | 1.0107 | 0.2354 | 0.1906 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.4680 | 0.6920 | 0.4757 | 0.3074 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.4094 | 1.1975 | 0.4314 | 0.2859 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.3825 | 0.8077 | 0.2765 | 0.1891 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.3031 | 1.1451 | 0.2851 | 0.3474 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.2100 | 0.9769 | 0.3560 | 0.3203 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.1719 | 1.0821 | 0.2594 | 0.4088 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.2081 | 0.6127 | 0.2112 | 0.2686 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.1797 | 0.6414 | 0.1867 | 0.1802 |

### Newsworthiness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 13.7330 | 14.3513 | 610536.8305 | 3273196.9404 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.7333 | 0.8257 | 0.2581 | 0.2680 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.6342 | 0.4208 | 0.2594 | 0.2471 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.6424 | 0.4428 | 0.2355 | 0.2496 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.9173 | 1.0737 | 0.4576 | 0.2901 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.8024 | 0.6972 | 0.4917 | 0.2923 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.4130 | 1.2455 | 0.4497 | 0.2759 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.8223 | 1.0392 | 0.3439 | 0.2829 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.0543 | 1.2869 | 0.3186 | 0.3284 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.7839 | 0.8432 | 0.3663 | 0.2987 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.8099 | 1.6618 | 0.3955 | 0.5637 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.7825 | 0.6993 | 0.2490 | 0.2167 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.5440 | 0.3707 | 0.2597 | 0.2331 |

### Hygiene

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 23059542.7025 | 34.8887 | 2076332.9671 | 2496567.2991 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.3722 | 1.1220 | 1.1599 | 0.4613 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.1616 | 1.5905 | 0.2697 | 0.2567 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.2066 | 1.5871 | 0.1030 | 0.2085 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.2335 | 1.6045 | 0.1942 | 0.1988 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.1706 | 0.6538 | 0.4506 | 0.2482 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.4287 | 0.2218 | 0.5326 | 0.4490 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.2633 | 0.6058 | 0.2160 | 0.2574 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.3027 | 1.3114 | 0.3930 | 0.2014 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.4266 | 0.6683 | 0.5872 | 0.3119 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.2427 | 0.4348 | 0.1139 | 0.6221 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.1411 | 0.2629 | 0.1968 | 0.2764 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.1754 | 0.7168 | 0.2870 | 0.2635 |

---

## Export layout

- `json/` — pairwise rows and per-judge G-Eval tables (JSON)
- `tables/` — CSV summaries (win rates, Bradley–Terry)
- `reports/` — Markdown / LaTeX for reading and papers

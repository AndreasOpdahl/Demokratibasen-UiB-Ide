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
| GPT4o-mini | 0.846 | 0.654 | 0.846 | 0.846 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.438 | 0.562 | 0.625 | 0.688 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.444 | 0.167 | 0.389 | 0.278 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.182 | 0.500 | 0.455 | 0.227 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.500 | 0.542 | 0.542 | 0.583 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.536 | 0.500 | 0.286 | 0.286 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.308 | 0.231 | 0.385 | 0.308 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.400 | 0.600 | 0.700 | 0.600 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.231 | 0.462 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.273 | 0.500 | 0.136 | 0.364 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.571 | 0.536 | 0.357 | 0.464 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.769 | 0.462 | 0.808 | 0.654 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.636 | 0.500 | 0.591 | 0.636 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.500 | 0.667 | 0.750 | 0.542 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.333 | 0.389 | 0.444 | 0.444 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.533 | 0.533 | 0.500 | 0.467 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.500 | 0.583 | 0.667 | 0.708 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.500 | 0.600 | 0.300 | 0.500 |

### Correctness

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.846 | 0.885 | 0.846 | 0.692 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.312 | 0.562 | 0.500 | 0.688 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.556 | 0.444 | 0.611 | 0.222 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.227 | 0.545 | 0.455 | 0.318 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.542 | 0.500 | 0.458 | 0.417 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.464 | 0.500 | 0.357 | 0.286 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.346 | 0.231 | 0.462 | 0.538 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.500 | 0.600 | 0.700 | 0.600 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.269 | 0.346 | 0.231 | 0.500 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.636 | 0.227 | 0.227 | 0.273 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.464 | 0.429 | 0.321 | 0.393 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.692 | 0.577 | 0.808 | 0.692 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.409 | 0.364 | 0.500 | 0.773 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.667 | 0.667 | 0.708 | 0.792 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.278 | 0.444 | 0.444 | 0.167 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.600 | 0.667 | 0.433 | 0.433 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.542 | 0.583 | 0.583 | 0.792 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.500 | 0.300 | 0.500 | 0.300 |

### Completeness

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.962 | 0.923 | 0.846 | 0.462 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.312 | 0.438 | 0.500 | 0.562 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.389 | 0.278 | 0.500 | 0.389 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.227 | 0.227 | 0.409 | 0.500 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.375 | 0.500 | 0.500 | 0.417 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.464 | 0.536 | 0.393 | 0.286 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.500 | 0.385 | 0.462 | 0.385 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.400 | 0.600 | 0.600 | 0.600 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.231 | 0.192 | 0.269 | 0.500 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.455 | 0.273 | 0.273 | 0.273 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.679 | 0.500 | 0.321 | 0.500 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.654 | 0.500 | 0.769 | 0.654 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.636 | 0.545 | 0.545 | 0.773 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.667 | 0.792 | 0.667 | 0.708 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.278 | 0.389 | 0.389 | 0.444 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.433 | 0.567 | 0.433 | 0.500 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.583 | 0.750 | 0.667 | 0.708 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.500 | 0.300 |

### Newsworthiness

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.962 | 0.615 | 0.846 | 0.654 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.188 | 0.500 | 0.312 | 0.312 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.222 | 0.556 | 0.556 | 0.389 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.318 | 0.818 | 0.364 | 0.591 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.458 | 0.375 | 0.333 | 0.250 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.321 | 0.429 | 0.464 | 0.393 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.462 | 0.692 | 0.385 | 0.231 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.400 | 0.600 | 0.600 | 0.400 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.346 | 0.538 | 0.192 | 0.500 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.591 | 0.227 | 0.409 | 0.409 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.571 | 0.679 | 0.500 | 0.607 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.654 | 0.423 | 0.692 | 0.654 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.636 | 0.591 | 0.545 | 0.727 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.625 | 0.458 | 0.625 | 0.500 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.500 | 0.389 | 0.444 | 0.556 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.467 | 0.433 | 0.500 | 0.533 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.542 | 0.292 | 0.625 | 0.667 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.500 | 0.300 | 0.700 | 0.500 |

### Hygiene

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.846 | 0.885 | 0.769 | 0.885 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.188 | 0.688 | 0.500 | 0.500 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.222 | 0.444 | 0.222 | 0.389 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.273 | 0.545 | 0.591 | 0.273 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.417 | 0.292 | 0.417 | 0.417 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.357 | 0.250 | 0.429 | 0.357 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.423 | 0.654 | 0.500 | 0.615 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.400 | 0.500 | 0.600 | 0.400 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.231 | 0.500 | 0.538 | 0.462 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.682 | 0.545 | 0.318 | 0.409 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.571 | 0.500 | 0.464 | 0.714 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.731 | 0.577 | 0.577 | 0.500 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.591 | 0.727 | 0.591 | 0.500 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.708 | 0.458 | 0.417 | 0.542 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.389 | 0.611 | 0.167 | 0.278 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.500 | 0.233 | 0.533 | 0.400 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.708 | 0.333 | 0.833 | 0.667 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.500 | 0.300 | 0.300 | 0.500 |

---

## 2. Bradley–Terry strengths (θ)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Faithfulness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 5.2310 | 1.9735 | 5.8228 | 5.3762 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.6366 | 1.5425 | 2.0229 | 3.0689 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.7736 | 0.2117 | 0.4586 | 0.2657 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.2875 | 1.1648 | 1.0228 | 0.2451 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.1574 | 1.2010 | 1.1734 | 1.3958 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.0357 | 0.9121 | 0.3776 | 0.3092 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.3797 | 0.3139 | 0.7069 | 0.4694 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.3399 | 0.8577 | 0.9922 | 0.8291 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.4534 | 1.1156 | 0.2888 | 0.9182 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.2936 | 0.8251 | 0.1192 | 0.4350 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 2.0460 | 1.2999 | 0.5490 | 0.9541 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 4.0591 | 0.8642 | 4.2193 | 1.3733 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 1.9329 | 1.0632 | 1.2285 | 2.0063 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.3988 | 2.0578 | 4.4511 | 1.2590 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.4773 | 0.5584 | 0.8495 | 0.7548 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.2005 | 1.0899 | 1.2225 | 0.8253 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.9912 | 1.6548 | 2.5447 | 3.5439 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.6031 | 1.9831 | 0.5053 | 1.8978 |

### Correctness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 5.1010 | 8.6336 | 5.0034 | 2.4234 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.3788 | 1.5587 | 0.9737 | 4.6033 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.3266 | 0.6718 | 1.4689 | 0.2087 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.3588 | 2.2135 | 0.9435 | 0.3974 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.9287 | 0.8767 | 0.6888 | 0.8060 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.8891 | 0.9589 | 0.5723 | 0.3440 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.5636 | 0.3423 | 0.8673 | 1.0767 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.1594 | 0.5373 | 1.5694 | 0.8513 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.4088 | 0.5333 | 0.2647 | 0.9627 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.8739 | 0.2229 | 0.2817 | 0.2426 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 1.0756 | 0.9492 | 0.4377 | 0.6846 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 2.4372 | 1.6139 | 3.8455 | 1.7735 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.7853 | 0.4742 | 0.8681 | 3.4286 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 2.2706 | 2.7422 | 3.0332 | 4.5097 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.4261 | 0.7300 | 0.8029 | 0.1924 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.3133 | 2.3057 | 0.8432 | 0.8593 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 1.0984 | 1.5734 | 1.2833 | 7.4988 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.9156 | 0.5155 | 0.9612 | 0.7824 |

### Completeness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 31.2574 | 15.0550 | 5.1121 | 0.8266 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.4030 | 1.1593 | 1.2076 | 1.4393 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.5760 | 0.2524 | 0.8271 | 0.6672 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.2956 | 0.3532 | 0.7728 | 0.9068 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.5119 | 0.8480 | 0.8905 | 0.9173 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.7566 | 0.9315 | 0.7097 | 0.4013 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.9030 | 0.7545 | 0.8928 | 0.5298 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.5988 | 0.7396 | 0.9778 | 0.9637 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.3206 | 0.2236 | 0.3248 | 1.2304 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.8624 | 0.3388 | 0.3504 | 0.3047 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 2.9287 | 0.9065 | 0.4513 | 1.1388 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 1.8578 | 0.9514 | 2.7839 | 1.8602 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 1.6022 | 1.1681 | 1.1334 | 2.9826 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 2.1885 | 4.4859 | 2.2568 | 2.8773 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.3705 | 0.5882 | 0.6330 | 0.7150 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.7230 | 1.2526 | 0.8198 | 1.0544 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 1.5169 | 4.4330 | 2.1826 | 2.5732 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.0390 | 1.3037 | 1.1090 | 0.5591 |

### Newsworthiness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 29.7687 | 2.2465 | 5.6402 | 2.2856 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.1657 | 0.7438 | 0.4928 | 0.4717 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.2449 | 1.4272 | 1.1913 | 0.7212 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.3625 | 5.4287 | 0.6184 | 1.5253 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.8492 | 0.6431 | 0.3820 | 0.4009 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.3600 | 0.6036 | 0.8354 | 0.6399 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.9539 | 1.9843 | 0.6283 | 0.2647 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.6317 | 1.2555 | 1.3388 | 0.4357 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.6355 | 1.2414 | 0.2285 | 1.2472 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.4306 | 0.2890 | 0.7841 | 0.5908 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 1.7270 | 2.4386 | 0.9801 | 2.0290 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 1.6906 | 0.7123 | 1.9973 | 1.9805 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 1.7228 | 1.1649 | 1.1923 | 2.4236 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.9233 | 0.9369 | 1.6529 | 1.3317 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 1.3909 | 0.6412 | 0.6957 | 1.0644 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.9361 | 1.0446 | 0.9753 | 1.2659 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.9488 | 0.3489 | 1.4711 | 1.6480 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.1409 | 0.5024 | 2.6369 | 1.3431 |

### Hygiene

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 7.2455 | 9.2951 | 4.0293 | 10.3057 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.2177 | 1.3790 | 2.1192 | 1.2338 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.2573 | 0.7520 | 0.2536 | 0.4839 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.3170 | 0.9961 | 1.7595 | 0.3677 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.6850 | 0.3827 | 0.7365 | 0.6627 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.5498 | 0.2981 | 0.8574 | 0.4492 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.8188 | 2.3407 | 1.1268 | 1.7082 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.7234 | 1.5464 | 0.9177 | 0.5069 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.3348 | 0.7623 | 0.8686 | 0.9102 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 2.4266 | 1.4692 | 0.4093 | 0.6195 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 1.7934 | 0.9374 | 0.9698 | 3.2921 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 2.7500 | 0.7937 | 1.0216 | 0.8101 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 1.4415 | 2.2839 | 1.4688 | 0.9035 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 3.0528 | 0.7163 | 0.6958 | 1.1952 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.8197 | 1.9608 | 0.1847 | 0.4171 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.0053 | 0.3697 | 1.2912 | 0.7184 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 2.6690 | 0.4920 | 6.8033 | 2.5799 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.8987 | 0.5187 | 0.6883 | 1.3661 |

---

## Export layout

- `json/` — pairwise rows and per-judge G-Eval tables (JSON)
- `tables/` — CSV summaries (win rates, Bradley–Terry)
- `reports/` — Markdown / LaTeX for reading and papers

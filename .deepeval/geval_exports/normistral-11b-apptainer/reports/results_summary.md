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
| GPT4o-mini | 0.714 | 0.893 | 1.000 | 0.929 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.214 | 0.714 | 0.500 | 0.571 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.600 | 0.400 | 0.800 | 0.400 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.429 | 0.500 | 0.357 | 0.500 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.346 | 0.577 | 0.385 | 0.385 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.812 | 0.562 | 0.562 | 0.625 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.667 | 0.611 | 0.444 | 0.556 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.154 | 0.462 | 0.500 | 0.423 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.300 | 0.400 | 0.100 | 0.500 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.462 | 0.231 | 0.269 | 0.308 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.750 | 0.417 | 0.583 | 0.250 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.667 | 0.542 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.577 | 0.346 | 0.269 | 0.577 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.591 | 0.364 | 0.545 | 0.273 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.389 | 0.611 | 0.444 | 0.333 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.375 | 0.625 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.556 | 0.111 | 0.222 | 0.611 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.143 | 0.571 | 0.357 | 0.643 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.500 | 0.542 | 0.708 | 0.458 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.643 | 0.607 | 0.500 | 0.464 |

### Correctness

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.750 | 0.893 | 1.000 | 0.929 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.214 | 0.643 | 0.500 | 0.643 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.550 | 0.350 | 0.800 | 0.350 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.429 | 0.500 | 0.500 | 0.500 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.308 | 0.577 | 0.385 | 0.385 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.562 | 0.562 | 0.812 | 0.625 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.611 | 0.667 | 0.444 | 0.556 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.385 | 0.346 | 0.269 | 0.192 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.300 | 0.400 | 0.100 | 0.500 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.538 | 0.192 | 0.192 | 0.269 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.667 | 0.250 | 0.417 | 0.250 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.625 | 0.542 | 0.667 | 0.542 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.615 | 0.615 | 0.423 | 0.500 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.545 | 0.455 | 0.364 | 0.364 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.389 | 0.556 | 0.444 | 0.667 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.500 | 0.375 | 0.375 | 0.750 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.500 | 0.222 | 0.333 | 0.556 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.286 | 0.357 | 0.357 | 0.571 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.375 | 0.542 | 0.792 | 0.500 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.571 | 0.643 | 0.500 | 0.464 |

### Completeness

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.607 | 0.786 | 0.857 | 0.786 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.071 | 0.643 | 0.357 | 0.500 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.550 | 0.500 | 0.750 | 0.350 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.429 | 0.429 | 0.500 | 0.500 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.269 | 0.346 | 0.308 | 0.346 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.562 | 0.625 | 0.688 | 0.375 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.444 | 0.667 | 0.333 | 0.278 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.385 | 0.462 | 0.423 | 0.346 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.500 | 0.500 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.538 | 0.269 | 0.269 | 0.385 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.667 | 0.417 | 0.750 | 0.417 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.500 | 0.625 | 0.708 | 0.625 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.692 | 0.462 | 0.269 | 0.500 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.500 | 0.591 | 0.500 | 0.500 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.500 | 0.667 | 0.444 | 0.500 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.625 | 0.438 | 0.500 | 0.750 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.556 | 0.167 | 0.333 | 0.722 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.286 | 0.286 | 0.357 | 0.571 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.583 | 0.375 | 0.625 | 0.417 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.571 | 0.643 | 0.536 | 0.607 |

### Newsworthiness

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.464 | 0.536 | 0.714 | 0.571 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.214 | 0.571 | 0.357 | 0.286 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.500 | 0.600 | 0.600 | 0.450 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.786 | 0.500 | 0.500 | 0.643 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.269 | 0.500 | 0.462 | 0.423 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.562 | 0.438 | 0.562 | 0.375 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.444 | 0.500 | 0.389 | 0.500 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.423 | 0.423 | 0.269 | 0.423 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.700 | 0.700 | 0.300 | 0.500 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.346 | 0.269 | 0.346 | 0.308 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.750 | 0.750 | 0.750 | 0.417 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.625 | 0.417 | 0.625 | 0.583 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.500 | 0.538 | 0.577 | 0.500 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.409 | 0.591 | 0.636 | 0.591 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.778 | 0.667 | 0.556 | 0.944 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.625 | 0.500 | 0.438 | 0.500 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.389 | 0.444 | 0.389 | 0.556 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.714 | 0.571 | 0.500 | 0.786 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.375 | 0.458 | 0.583 | 0.375 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.571 | 0.393 | 0.393 | 0.429 |

### Hygiene

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.786 | 0.929 | 0.929 | 0.929 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.357 | 0.214 | 0.214 | 0.500 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.250 | 0.750 | 0.400 | 0.300 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.786 | 0.500 | 0.500 | 0.500 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.346 | 0.385 | 0.500 | 0.308 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.438 | 0.438 | 0.688 | 0.500 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.556 | 0.167 | 0.556 | 0.444 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.423 | 0.577 | 0.346 | 0.308 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.500 | 0.100 | 0.300 | 0.500 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.269 | 0.462 | 0.192 | 0.231 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.583 | 0.500 | 0.750 | 0.500 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.625 | 0.667 | 0.667 | 0.792 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.654 | 0.385 | 0.423 | 0.423 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.364 | 0.500 | 0.636 | 0.591 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.611 | 0.500 | 0.667 | 0.667 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.500 | 0.688 | 0.625 | 0.562 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.611 | 0.278 | 0.333 | 0.389 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.571 | 0.286 | 0.286 | 0.571 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.292 | 0.542 | 0.458 | 0.375 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.571 | 0.571 | 0.429 | 0.607 |

---

## 2. Bradley–Terry strengths (θ)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Faithfulness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 2.7951 | 15.1609 | 3985862.4078 | 27.5404 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.2330 | 5.0605 | 0.7713 | 1.4720 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.3861 | 0.5869 | 1.6395 | 0.6084 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.7176 | 0.5998 | 0.1654 | 0.5865 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.6989 | 2.4897 | 0.4324 | 0.6276 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 5.8932 | 1.6563 | 0.9808 | 1.7721 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.4283 | 1.9649 | 0.5384 | 0.9939 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.1258 | 0.6551 | 0.4877 | 0.6570 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.1144 | 2.2882 | 0.3382 | 7.0717 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.0267 | 0.1872 | 0.1540 | 0.2857 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 4.4819 | 0.5343 | 0.6894 | 0.2467 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 1.1269 | 0.6108 | 0.8774 | 0.7038 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.9558 | 0.3545 | 0.2110 | 0.9766 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 2.1103 | 0.3968 | 0.7036 | 0.2387 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.5539 | 1.1023 | 0.2982 | 0.3679 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.8784 | 1.0154 | 0.2084 | 1.4363 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.4760 | 0.0779 | 0.1118 | 1.3338 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.1188 | 1.7885 | 0.3354 | 2.6734 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.8256 | 1.1386 | 1.8453 | 0.7530 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 2.5419 | 1.9989 | 0.5360 | 0.8204 |

### Correctness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 2.6087 | 13.3041 | 5182176.5692 | 26.2206 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.2725 | 3.7574 | 0.8044 | 2.0151 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.1462 | 0.4543 | 1.8053 | 0.5022 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.7337 | 0.5032 | 0.2751 | 0.6491 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.5936 | 2.3486 | 0.3094 | 0.7588 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.3993 | 1.6478 | 11.0238 | 1.5150 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.2860 | 2.5559 | 0.4443 | 0.9849 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.5743 | 0.5745 | 0.1790 | 0.1891 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.9054 | 1.7030 | 0.1622 | 6.6628 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.3117 | 0.1905 | 0.0936 | 0.2759 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 2.2627 | 0.2541 | 0.2264 | 0.2616 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 1.9169 | 1.0859 | 0.6514 | 0.6342 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 1.3993 | 1.2228 | 0.3469 | 0.5567 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 1.6834 | 0.7689 | 0.2144 | 0.3168 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.5989 | 0.9330 | 0.2819 | 1.4344 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.8556 | 0.4833 | 0.2224 | 2.8906 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.2741 | 0.1905 | 0.1576 | 1.0711 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.4095 | 0.6104 | 0.3501 | 1.5960 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.5974 | 1.4208 | 5.3229 | 0.7429 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 1.4067 | 2.4545 | 0.5808 | 0.9576 |

### Completeness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.4130 | 4.5780 | 10.3871 | 4.4009 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.0717 | 2.8496 | 0.5170 | 0.9580 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.6150 | 0.8325 | 3.1559 | 0.4757 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.7134 | 0.5020 | 0.6099 | 0.9312 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.3873 | 0.7066 | 0.4410 | 0.6131 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.3841 | 2.0739 | 2.5236 | 0.6189 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.5508 | 2.0414 | 0.3956 | 0.2910 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.6664 | 0.8045 | 0.4553 | 0.4178 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.6255 | 2.2828 | 7.0833 | 2.1050 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.2063 | 0.3178 | 0.3228 | 0.6401 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 2.5051 | 0.6856 | 4.8845 | 0.8447 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 1.0331 | 1.5277 | 1.3851 | 1.2357 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 2.5175 | 0.7118 | 0.2987 | 0.7590 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 1.1932 | 1.5705 | 0.9947 | 0.9205 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.2471 | 1.3681 | 0.5029 | 0.9174 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 1.5035 | 0.5878 | 0.6458 | 3.1772 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.7430 | 0.1771 | 0.3961 | 3.4468 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.4408 | 0.3927 | 0.5138 | 1.3039 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.4850 | 0.6845 | 1.3991 | 0.5001 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 1.3351 | 2.3439 | 1.1355 | 1.8033 |

### Newsworthiness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.6683 | 1.3144 | 2.4224 | 1.0988 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.2149 | 0.9788 | 0.4208 | 0.2988 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.5843 | 2.0387 | 1.6435 | 1.0106 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 3.2113 | 0.5867 | 0.6725 | 1.1134 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.3361 | 1.0240 | 1.0304 | 0.8159 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.0461 | 0.5698 | 1.2497 | 0.4833 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.4994 | 0.8507 | 0.5767 | 0.7946 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.5551 | 0.6536 | 0.3791 | 0.7076 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 2.7219 | 4.3855 | 1.0142 | 0.9829 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.5055 | 0.2758 | 0.5959 | 0.4646 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 3.2039 | 4.2276 | 3.3380 | 0.6820 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.9689 | 0.4951 | 1.4486 | 1.0870 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.9010 | 1.0676 | 1.2971 | 1.0478 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.5487 | 1.0560 | 1.9996 | 1.0799 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 3.0572 | 1.6559 | 1.0460 | 14.9619 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 2.2598 | 1.0011 | 0.6677 | 1.4208 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.7141 | 0.7178 | 0.6363 | 1.3658 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 2.3778 | 1.3637 | 1.0849 | 3.3823 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.4409 | 0.7128 | 1.2945 | 0.4992 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 1.2857 | 0.6067 | 0.6666 | 0.6501 |

### Hygiene

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 3.7156 | 9.9242 | 18.2223 | 27.5162 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.6263 | 0.2723 | 0.2024 | 0.9728 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.3873 | 2.5363 | 0.5627 | 0.3173 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 2.3198 | 1.1717 | 0.3907 | 0.5166 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.6544 | 0.7806 | 1.9692 | 0.6190 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.6768 | 1.0559 | 2.3317 | 1.0720 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.8851 | 0.1999 | 1.1732 | 0.5812 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.6525 | 1.0784 | 0.3613 | 0.3072 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 2.0455 | 0.3532 | 2.2611 | 7.8238 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.3850 | 1.1048 | 0.2314 | 0.2761 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.3979 | 0.9329 | 3.8809 | 1.1197 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 1.2405 | 1.6892 | 1.6726 | 2.5614 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 1.4378 | 0.7465 | 0.4660 | 0.4304 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.5246 | 1.3040 | 2.3936 | 1.5700 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.2725 | 1.1372 | 1.3098 | 1.0851 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 1.1266 | 1.9982 | 1.3087 | 0.9437 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.9459 | 0.5222 | 0.4766 | 0.6050 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 1.2758 | 0.5652 | 0.4088 | 1.1003 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.3433 | 1.2634 | 0.5968 | 0.3788 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 1.4862 | 1.3815 | 0.8394 | 1.7795 |

---

## Export layout

- `json/` — pairwise rows and per-judge G-Eval tables (JSON)
- `tables/` — CSV summaries (win rates, Bradley–Terry)
- `reports/` — Markdown / LaTeX for reading and papers

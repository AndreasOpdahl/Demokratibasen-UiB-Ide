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
| GPT4o-mini | 0.786 | 0.821 | 0.857 | 0.893 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.143 | 0.286 | 0.500 | 0.357 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.200 | 0.550 | 0.400 | 0.450 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.714 | 0.929 | 0.714 | 0.857 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.231 | 0.462 | 0.346 | 0.308 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.188 | 0.250 | 0.188 | 0.250 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.556 | 0.278 | 0.500 | 0.333 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.500 | 0.577 | 0.500 | 0.462 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.200 | 0.100 | 0.300 | 0.100 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.538 | 0.500 | 0.385 | 0.308 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.583 | 0.333 | 0.417 | 0.500 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.708 | 0.708 | 0.500 | 0.667 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.577 | 0.538 | 0.423 | 0.615 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.409 | 0.364 | 0.500 | 0.591 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.389 | 0.611 | 0.389 | 0.333 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.688 | 0.125 | 0.250 | 0.375 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.444 | 0.500 | 0.722 | 0.500 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.714 | 0.286 | 0.643 | 0.714 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.542 | 0.667 | 0.708 | 0.625 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.607 | 0.500 | 0.536 | 0.464 |

### Correctness

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.821 | 0.893 | 0.857 | 0.857 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.214 | 0.286 | 0.286 | 0.500 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.400 | 0.500 | 0.550 | 0.500 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.571 | 1.000 | 0.714 | 0.857 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.269 | 0.346 | 0.308 | 0.346 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.250 | 0.188 | 0.188 | 0.375 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.556 | 0.333 | 0.556 | 0.278 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.423 | 0.654 | 0.577 | 0.308 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.200 | 0.100 | 0.300 | 0.100 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.538 | 0.462 | 0.462 | 0.308 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.333 | 0.417 | 0.583 | 0.500 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.667 | 0.750 | 0.542 | 0.750 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.577 | 0.423 | 0.462 | 0.615 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.455 | 0.273 | 0.409 | 0.591 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.278 | 0.500 | 0.333 | 0.333 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.500 | 0.250 | 0.188 | 0.375 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.611 | 0.500 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.786 | 0.500 | 0.643 | 0.643 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.500 | 0.667 | 0.625 | 0.583 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.714 | 0.500 | 0.536 | 0.464 |

### Completeness

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.786 | 0.857 | 0.857 | 0.929 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.500 | 0.571 | 0.571 | 0.214 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.350 | 0.450 | 0.600 | 0.350 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.786 | 0.857 | 0.571 | 0.429 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.231 | 0.500 | 0.308 | 0.308 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.188 | 0.188 | 0.188 | 0.188 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.444 | 0.278 | 0.500 | 0.722 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.500 | 0.462 | 0.500 | 0.462 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.600 | 0.500 | 0.500 | 0.400 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.423 | 0.385 | 0.500 | 0.538 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.333 | 0.667 | 0.417 | 0.667 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.625 | 0.500 | 0.583 | 0.500 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.538 | 0.654 | 0.423 | 0.577 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.682 | 0.318 | 0.500 | 0.545 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.333 | 0.278 | 0.278 | 0.500 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.625 | 0.188 | 0.000 | 0.125 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.333 | 0.500 | 0.500 | 0.500 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.714 | 0.571 | 0.643 | 0.786 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.417 | 0.625 | 0.625 | 0.417 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.571 | 0.536 | 0.643 | 0.571 |

### Newsworthiness

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.536 | 0.786 | 0.857 | 0.643 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.643 | 0.214 | 0.357 | 0.429 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.300 | 0.650 | 0.600 | 0.550 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.571 | 0.714 | 0.429 | 0.357 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.423 | 0.423 | 0.385 | 0.346 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.375 | 0.375 | 0.250 | 0.562 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.389 | 0.389 | 0.500 | 0.444 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.615 | 0.385 | 0.385 | 0.500 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.600 | 0.400 | 0.400 | 0.400 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.538 | 0.538 | 0.538 | 0.500 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.583 | 0.417 | 0.333 | 0.417 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.458 | 0.500 | 0.500 | 0.458 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.577 | 0.615 | 0.500 | 0.577 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.091 | 0.500 | 0.545 | 0.500 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.500 | 0.444 | 0.556 | 0.556 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.625 | 0.312 | 0.250 | 0.375 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.444 | 0.444 | 0.611 | 0.500 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.857 | 0.786 | 0.714 | 0.786 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.417 | 0.500 | 0.542 | 0.458 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.643 | 0.429 | 0.464 | 0.536 |

### Hygiene

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.500 | 0.429 | 0.857 | 0.821 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.857 | 0.429 | 0.429 | 0.429 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.350 | 0.450 | 0.550 | 0.600 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.714 | 0.643 | 0.857 | 0.857 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.308 | 0.500 | 0.346 | 0.192 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.312 | 0.312 | 0.250 | 0.250 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.389 | 0.500 | 0.611 | 0.389 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.346 | 0.577 | 0.462 | 0.462 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.500 | 0.200 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.654 | 0.462 | 0.385 | 0.462 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.750 | 0.417 | 0.417 | 0.417 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.375 | 0.625 | 0.542 | 0.833 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.538 | 0.462 | 0.654 | 0.615 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.455 | 0.727 | 0.591 | 0.591 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.500 | 0.444 | 0.222 | 0.278 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.250 | 0.375 | 0.375 | 0.250 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.500 | 0.611 | 0.500 | 0.444 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.929 | 0.571 | 0.500 | 0.643 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.583 | 0.417 | 0.500 | 0.500 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.536 | 0.500 | 0.357 | 0.464 |

---

## 2. Bradley–Terry strengths (θ)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Faithfulness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 2.6406 | 3.6214 | 6.2623 | 6.6871 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.2484 | 0.6048 | 1.6098 | 0.7328 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.2672 | 1.0527 | 0.5885 | 0.8044 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 3.0048 | 18.6085 | 1.9638 | 4.1686 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.4539 | 0.8268 | 0.5769 | 0.4797 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.3300 | 0.3869 | 0.2955 | 0.4455 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.1553 | 0.3638 | 0.8621 | 0.4400 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.0519 | 1.6875 | 1.0786 | 1.0540 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.2636 | 0.1319 | 0.8350 | 0.1964 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.6404 | 2.5319 | 0.7443 | 0.6559 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.8771 | 0.3336 | 0.6547 | 0.7037 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 2.7645 | 3.0445 | 1.0020 | 1.9222 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 1.3925 | 1.6898 | 0.8162 | 1.9456 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.8660 | 0.7840 | 0.8850 | 1.6168 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.6459 | 1.9922 | 0.5606 | 0.4409 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 2.5216 | 0.1167 | 0.3202 | 0.5302 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.1799 | 1.3327 | 2.6342 | 1.0625 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 2.5740 | 0.2940 | 1.4746 | 2.9307 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.1462 | 2.9974 | 2.4473 | 1.8284 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 1.4552 | 0.9675 | 1.0771 | 0.8420 |

### Correctness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 4.0826 | 2.9882 | 5.6838 | 4.8673 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.4052 | 0.3322 | 0.5749 | 1.2772 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.6103 | 0.4199 | 1.2891 | 0.9072 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.4904 | 9988869.3885 | 1.9912 | 5.0245 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.5112 | 0.2637 | 0.4937 | 0.5614 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.5430 | 0.1356 | 0.2831 | 0.7380 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.1014 | 0.2467 | 1.0316 | 0.3505 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.7809 | 1.0000 | 1.3524 | 0.5049 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.3103 | 0.0743 | 1.2379 | 0.1971 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.4117 | 0.9701 | 0.9550 | 0.7336 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.3777 | 0.1943 | 1.3760 | 0.7660 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 2.3001 | 1.5656 | 1.1728 | 2.7389 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 1.4520 | 0.4842 | 1.0371 | 1.6244 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 1.0268 | 0.2061 | 0.6938 | 1.7226 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.3872 | 0.5255 | 0.4434 | 0.4136 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 1.1024 | 0.1565 | 0.2222 | 0.5215 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.2219 | 0.6251 | 1.6769 | 1.0065 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 4.0606 | 0.4252 | 1.7084 | 1.8150 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.9888 | 1.3405 | 1.7628 | 1.4166 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 2.3434 | 0.4244 | 0.8953 | 1.0161 |

### Completeness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 4.5512 | 9.5813 | 18.7979 | 21.3072 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 1.3305 | 2.1204 | 4.5926 | 0.2537 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.5298 | 0.9133 | 2.6898 | 0.5141 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 3.7100 | 4.9298 | 2.6756 | 0.4026 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.3438 | 0.9424 | 0.8276 | 0.6946 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.2907 | 0.2217 | 0.6934 | 0.2808 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.6265 | 0.2876 | 1.6438 | 1.5519 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.8346 | 0.7198 | 1.9028 | 0.7776 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 3.2692 | 4.7211 | 8.2431 | 4.0720 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.8757 | 0.5992 | 2.0952 | 0.9773 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.4236 | 2.3060 | 1.6679 | 2.3834 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 1.4507 | 0.6682 | 2.7307 | 1.0513 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.9735 | 1.5902 | 1.6078 | 1.2102 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 2.0984 | 0.3998 | 2.1169 | 1.3838 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.3587 | 0.2930 | 0.4398 | 0.6852 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 1.4635 | 0.1703 | 0.0000 | 0.1167 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.4753 | 0.9001 | 1.6581 | 1.0858 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 2.3889 | 1.1871 | 2.9388 | 4.5322 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.5899 | 1.5347 | 3.6291 | 0.6063 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 1.2717 | 1.1148 | 3.2604 | 0.9536 |

### Newsworthiness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.1859 | 4.1184 | 7.2856 | 1.7552 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 2.2363 | 0.2215 | 0.4969 | 0.7043 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.5387 | 2.3522 | 1.5138 | 1.2628 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.5346 | 2.4684 | 0.6140 | 0.4709 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.6758 | 0.8044 | 0.7861 | 0.5438 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.6194 | 0.6573 | 0.3743 | 1.6844 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.5413 | 0.4461 | 0.7821 | 0.6168 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.5716 | 0.5615 | 0.6245 | 1.2014 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.5010 | 1.7073 | 2.0210 | 0.8937 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.8070 | 1.3995 | 1.1670 | 0.8457 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.0511 | 0.7307 | 0.5801 | 0.6650 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.5302 | 0.8317 | 0.9688 | 0.7635 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 1.3299 | 1.6365 | 1.0763 | 1.6424 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.0853 | 0.9950 | 1.1378 | 0.9944 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.0443 | 0.7138 | 1.1413 | 1.2087 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 2.0243 | 0.4581 | 0.3395 | 0.5727 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.9357 | 0.7244 | 1.5596 | 1.0119 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 6.6102 | 4.8033 | 2.7484 | 4.3864 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.6975 | 0.9294 | 1.1559 | 0.9628 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 1.8340 | 0.5701 | 0.6635 | 1.0806 |

### Hygiene

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.0021 | 0.7043 | 7.0295 | 3.4103 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 7.3483 | 0.7509 | 0.8712 | 0.9381 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.5409 | 0.7293 | 1.4638 | 1.4640 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 4.2028 | 1.6440 | 4.5676 | 6.9261 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.3462 | 1.0001 | 0.5548 | 0.2062 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.5235 | 0.4850 | 0.2821 | 0.4391 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.3390 | 1.0444 | 1.3221 | 0.5036 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.4771 | 1.4086 | 0.8022 | 1.0162 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.0372 | 0.7278 | 3.3836 | 0.4173 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 2.0085 | 1.1516 | 0.7373 | 1.6428 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 2.2348 | 0.8017 | 0.7071 | 0.5077 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.4351 | 2.1552 | 1.1073 | 5.7474 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.9149 | 0.9660 | 1.6721 | 2.0075 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.6885 | 2.9793 | 1.3041 | 2.1009 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.7161 | 0.7689 | 0.2165 | 0.2818 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.2673 | 0.6274 | 0.4812 | 0.2241 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.8349 | 1.5665 | 0.8163 | 0.7963 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 7.4601 | 1.0762 | 0.9610 | 1.6686 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.1310 | 0.6837 | 0.9978 | 1.2260 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 1.1845 | 0.9538 | 0.4236 | 0.7665 |

---

## Export layout

- `json/` — pairwise rows and per-judge G-Eval tables (JSON)
- `tables/` — CSV summaries (win rates, Bradley–Terry)
- `reports/` — Markdown / LaTeX for reading and papers

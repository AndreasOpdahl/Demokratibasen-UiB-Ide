# G-Eval results summary

**Judges:** `gpt-3.5-turbo`, `google/gemini-2.5-flash-preview-05-20`, `anthropic/claude-3-5-haiku-20241022`, `mistral-medium-latest`
**Dimensions:** `faithfulness`, `correctness`, `completeness`, `newsworthiness`, `hygiene`
**Documents in subset:** 71 distinct `doc_id`.
**Datapoints:** 5680 pairwise judgments total (284 rows per G-Eval table × 20 table(s), one per judge × dimension).
Equivalent to 284 pair comparisons × 5 dimensions × 4 judges.

Bradley–Terry: `GPT4o-mini` labels gold summaries (JSONL `reference`). Exported θ use mean-centered β (geom. mean θ = 1); odds vs any other model match the fitted BT model.

---

## 1. Pairwise win rates

### Faithfulness

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.641 | 0.828 | 0.828 | 0.812 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.387 | 0.500 | 0.565 | 0.355 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.486 | 0.514 | 0.471 | 0.429 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.521 | 0.521 | 0.438 | 0.604 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.621 | 0.500 | 0.414 | 0.552 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.429 | 0.476 | 0.405 | 0.381 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.600 | 0.460 | 0.540 | 0.680 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.350 | 0.450 | 0.467 | 0.433 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.522 | 0.413 | 0.500 | 0.478 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.519 | 0.596 | 0.635 | 0.596 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.574 | 0.333 | 0.352 | 0.389 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.519 | 0.404 | 0.481 | 0.462 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.485 | 0.455 | 0.379 | 0.515 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.520 | 0.580 | 0.540 | 0.500 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.460 | 0.520 | 0.520 | 0.500 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.426 | 0.456 | 0.441 | 0.397 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.537 | 0.370 | 0.463 | 0.537 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.353 | 0.544 | 0.515 | 0.441 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.552 | 0.483 | 0.483 | 0.466 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.547 | 0.531 | 0.531 | 0.500 |

### Correctness

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.719 | 0.812 | 0.859 | 0.797 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.403 | 0.387 | 0.500 | 0.355 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.429 | 0.557 | 0.471 | 0.486 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.521 | 0.500 | 0.438 | 0.500 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.603 | 0.517 | 0.414 | 0.552 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.476 | 0.357 | 0.500 | 0.429 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.600 | 0.540 | 0.500 | 0.580 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.400 | 0.467 | 0.450 | 0.433 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.500 | 0.435 | 0.500 | 0.500 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.462 | 0.615 | 0.673 | 0.596 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.593 | 0.407 | 0.370 | 0.463 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.519 | 0.442 | 0.442 | 0.442 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.530 | 0.439 | 0.424 | 0.485 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.480 | 0.500 | 0.500 | 0.480 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.520 | 0.520 | 0.520 | 0.500 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.456 | 0.382 | 0.471 | 0.382 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.519 | 0.444 | 0.500 | 0.593 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.353 | 0.559 | 0.515 | 0.471 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.448 | 0.466 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.469 | 0.547 | 0.484 | 0.500 |

### Completeness

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.656 | 0.766 | 0.688 | 0.641 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.387 | 0.452 | 0.548 | 0.403 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.443 | 0.557 | 0.414 | 0.486 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.542 | 0.479 | 0.583 | 0.625 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.603 | 0.586 | 0.517 | 0.638 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.548 | 0.429 | 0.500 | 0.452 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.540 | 0.420 | 0.440 | 0.520 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.300 | 0.467 | 0.483 | 0.483 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.457 | 0.435 | 0.587 | 0.457 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.462 | 0.500 | 0.481 | 0.442 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.611 | 0.389 | 0.444 | 0.519 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.558 | 0.385 | 0.519 | 0.462 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.545 | 0.439 | 0.409 | 0.485 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.460 | 0.580 | 0.600 | 0.440 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.520 | 0.520 | 0.500 | 0.540 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.471 | 0.456 | 0.456 | 0.485 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.519 | 0.611 | 0.574 | 0.648 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.368 | 0.441 | 0.426 | 0.382 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.586 | 0.483 | 0.534 | 0.534 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.484 | 0.547 | 0.375 | 0.391 |

### Newsworthiness

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.609 | 0.781 | 0.766 | 0.688 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.468 | 0.419 | 0.452 | 0.355 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.443 | 0.600 | 0.443 | 0.486 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.562 | 0.500 | 0.646 | 0.625 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.517 | 0.500 | 0.517 | 0.552 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.452 | 0.381 | 0.405 | 0.500 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.580 | 0.460 | 0.480 | 0.560 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.417 | 0.367 | 0.467 | 0.417 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.478 | 0.543 | 0.500 | 0.522 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.423 | 0.500 | 0.462 | 0.481 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.500 | 0.463 | 0.500 | 0.500 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.481 | 0.481 | 0.481 | 0.538 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.591 | 0.348 | 0.485 | 0.424 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.380 | 0.500 | 0.560 | 0.420 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.560 | 0.460 | 0.540 | 0.540 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.544 | 0.456 | 0.368 | 0.456 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.519 | 0.611 | 0.611 | 0.556 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.500 | 0.485 | 0.382 | 0.426 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.534 | 0.552 | 0.483 | 0.500 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.422 | 0.547 | 0.500 | 0.516 |

### Hygiene

| model | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.719 | 0.859 | 0.875 | 0.859 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.548 | 0.629 | 0.548 | 0.403 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.429 | 0.529 | 0.429 | 0.371 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.542 | 0.562 | 0.521 | 0.604 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.517 | 0.466 | 0.483 | 0.552 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.405 | 0.452 | 0.429 | 0.429 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.480 | 0.400 | 0.620 | 0.560 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.483 | 0.500 | 0.533 | 0.400 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.391 | 0.500 | 0.326 | 0.370 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.481 | 0.558 | 0.654 | 0.577 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.556 | 0.463 | 0.352 | 0.463 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.538 | 0.500 | 0.365 | 0.423 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.500 | 0.455 | 0.455 | 0.576 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.460 | 0.480 | 0.620 | 0.580 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.540 | 0.560 | 0.520 | 0.600 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.471 | 0.279 | 0.426 | 0.412 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.444 | 0.519 | 0.481 | 0.519 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.471 | 0.441 | 0.441 | 0.485 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.500 | 0.431 | 0.397 | 0.414 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.484 | 0.422 | 0.500 | 0.422 |

---

## 2. Bradley–Terry strengths (θ)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Faithfulness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.9034 | 4.8479 | 4.7399 | 4.4097 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.6609 | 1.0137 | 1.3065 | 0.5709 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.9106 | 1.0248 | 0.8818 | 0.7442 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.1505 | 1.3998 | 0.9624 | 1.7453 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.8025 | 1.0996 | 0.7540 | 1.3884 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.7025 | 0.9650 | 0.6925 | 0.6136 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.4006 | 0.7512 | 1.0390 | 1.8337 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.4958 | 0.7360 | 0.7875 | 0.6697 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.1697 | 0.8152 | 1.1945 | 1.1274 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.0696 | 1.5436 | 1.7533 | 1.4644 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.4130 | 0.4905 | 0.5242 | 0.6140 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.9908 | 0.6597 | 0.9261 | 0.7705 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.8169 | 0.7643 | 0.5750 | 0.8885 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 1.1228 | 1.3471 | 1.1905 | 1.0168 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.8462 | 0.9918 | 0.9265 | 0.8856 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.7989 | 0.8462 | 0.7983 | 0.6881 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.2454 | 0.5803 | 0.9020 | 1.1907 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.5116 | 1.1854 | 1.0030 | 0.7333 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.0530 | 0.8628 | 0.8294 | 0.7552 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 1.1842 | 1.1578 | 1.1531 | 1.0311 |

### Correctness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 2.6107 | 4.3141 | 6.0094 | 4.0152 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.6943 | 0.6552 | 0.9565 | 0.5486 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.7437 | 1.2818 | 0.8785 | 0.9323 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.1286 | 1.1812 | 0.9759 | 1.1222 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.7567 | 1.1780 | 0.7436 | 1.3436 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.8329 | 0.5998 | 0.9932 | 0.7444 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.3697 | 1.0676 | 0.8609 | 1.2423 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.6000 | 0.8001 | 0.7245 | 0.6830 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.1753 | 0.8709 | 1.2573 | 1.1999 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.8241 | 1.5886 | 2.0469 | 1.4348 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.5140 | 0.6813 | 0.5701 | 0.8823 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.9169 | 0.7831 | 0.7429 | 0.7247 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.9641 | 0.7306 | 0.6449 | 0.8025 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.9097 | 1.0193 | 1.0209 | 0.9708 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.0565 | 0.9937 | 0.9423 | 0.9194 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.8999 | 0.6705 | 0.9072 | 0.6710 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.1168 | 0.8318 | 1.0226 | 1.5160 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.5230 | 1.2269 | 1.0086 | 0.8355 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.9094 | 0.8661 | 0.7320 | 0.7390 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.9180 | 1.1956 | 0.9513 | 1.0325 |

### Completeness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.9378 | 3.4946 | 2.3540 | 1.8961 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.6593 | 0.7942 | 1.1799 | 0.6637 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.7944 | 1.1357 | 0.6799 | 0.9233 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.1967 | 1.1004 | 1.5585 | 1.6778 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.7289 | 1.4933 | 1.0925 | 1.8639 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.1141 | 0.7313 | 0.9531 | 0.7658 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.0889 | 0.6743 | 0.7269 | 1.0391 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.4076 | 0.7585 | 0.8043 | 0.8068 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.9168 | 0.8515 | 1.6264 | 0.9455 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.8484 | 1.0649 | 1.0392 | 0.8247 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.5938 | 0.7460 | 0.8270 | 1.1982 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 1.0667 | 0.6150 | 1.0012 | 0.7674 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 1.0471 | 0.7003 | 0.6283 | 0.8412 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.8598 | 1.3991 | 1.5112 | 0.7806 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.1098 | 1.0511 | 0.8488 | 1.1416 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.9694 | 0.8981 | 0.8670 | 1.0663 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.0906 | 1.6480 | 1.2826 | 1.7331 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.5547 | 0.7717 | 0.6969 | 0.6014 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.3114 | 0.7896 | 0.9983 | 1.0309 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.9479 | 1.3360 | 0.6397 | 0.6962 |

### Newsworthiness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.5524 | 4.0508 | 3.4516 | 2.3696 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.9217 | 0.7157 | 0.8224 | 0.5588 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.8488 | 1.4652 | 0.7455 | 0.9368 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.2498 | 1.1954 | 2.0043 | 1.7770 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.1730 | 1.0724 | 1.1712 | 1.3485 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.8208 | 0.5985 | 0.6375 | 0.9628 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.3650 | 0.7992 | 0.8031 | 1.1600 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.7134 | 0.5092 | 0.7410 | 0.6387 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.0187 | 1.3688 | 1.1988 | 1.2364 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.7036 | 1.0607 | 0.9145 | 0.9400 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.9991 | 0.9627 | 1.0390 | 1.0290 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.8391 | 0.8919 | 0.8156 | 0.9896 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 1.4073 | 0.4733 | 0.7988 | 0.6415 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.5692 | 1.0356 | 1.3470 | 0.7400 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.2701 | 0.7794 | 1.0756 | 1.1086 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 1.2680 | 0.9517 | 0.6314 | 0.9254 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.0183 | 1.6904 | 1.6126 | 1.2611 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.9857 | 0.8743 | 0.5675 | 0.7040 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.1912 | 0.9408 | 0.7602 | 0.8451 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.7580 | 1.1964 | 1.1233 | 1.0906 |

### Hygiene

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 2.4571 | 6.2344 | 6.5279 | 5.9233 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 1.2571 | 1.6818 | 1.1734 | 0.6607 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.7784 | 1.0969 | 0.6981 | 0.5737 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.2545 | 1.5311 | 1.3685 | 1.8223 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.2349 | 0.9215 | 1.0625 | 1.4881 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.6531 | 0.7819 | 0.7123 | 0.6963 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.8577 | 0.6050 | 1.3064 | 1.0076 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.8516 | 0.8414 | 0.9337 | 0.5520 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.7349 | 1.3375 | 0.5937 | 0.7015 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.9175 | 1.2858 | 1.9622 | 1.4240 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.2344 | 0.8043 | 0.5250 | 0.8768 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 1.0738 | 0.9278 | 0.5717 | 0.6248 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.9323 | 0.7495 | 0.7022 | 1.0473 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.8749 | 0.9613 | 1.7113 | 1.4616 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.1668 | 1.0545 | 0.9578 | 1.3827 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.9645 | 0.4071 | 0.7363 | 0.7570 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.8188 | 0.9889 | 0.9735 | 1.1039 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.8705 | 0.7433 | 0.7700 | 0.8823 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.9585 | 0.6693 | 0.5957 | 0.6077 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 1.0041 | 0.7929 | 1.1190 | 0.8173 |

---

## Export layout

- `json/` — pairwise rows and per-judge G-Eval tables (JSON)
- `tables/` — CSV summaries (win rates, Bradley–Terry)
- `reports/` — Markdown / LaTeX for reading and papers

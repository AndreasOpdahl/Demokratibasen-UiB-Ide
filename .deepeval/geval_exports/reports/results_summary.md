# G-Eval results summary

**Judges:** `google/gemma-3-4b`, `gpt-3.5-turbo`, `google/gemini-2.5-flash-preview-05-20`, `anthropic/claude-3-5-haiku-20241022`, `mistral-medium-latest`
**Dimensions:** `faithfulness`, `correctness`, `completeness`, `newsworthiness`
**Documents in subset:** 21 distinct `doc_id`.
**Datapoints:** 1680 pairwise judgments total (84 rows per G-Eval table × 20 table(s), one per judge × dimension).
Equivalent to 84 pair comparisons × 4 dimensions × 5 judges.

Bradley–Terry: `GPT4o-mini` labels gold summaries (JSONL `reference`). Exported θ use mean-centered β (geom. mean θ = 1); odds vs any other model match the fitted BT model.

---

## 1. Pairwise win rates

### Faithfulness

| model | google/gemma-3-4b_win_rate | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- | --- |
| GPT4o-mini | 0.600 | 0.833 | 0.900 | 1.000 | 0.967 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 0.625 | 0.750 | 0.594 | 0.688 | 0.719 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.583 | 0.333 | 0.250 | 0.278 | 0.222 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.531 | 0.656 | 0.688 | 0.844 | 0.781 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 0.611 | 0.361 | 0.472 | 0.528 | 0.528 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.342 | 0.105 | 0.079 | 0.105 | 0.000 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.425 | 0.675 | 0.500 | 0.425 | 0.475 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.346 | 0.154 | 0.577 | 0.231 | 0.269 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.375 | 0.500 | 0.594 | 0.375 | 0.438 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 0.559 | 0.647 | 0.500 | 0.618 | 0.706 |

### Correctness

| model | google/gemma-3-4b_win_rate | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- | --- |
| GPT4o-mini | 0.633 | 0.900 | 1.000 | 0.933 | 0.900 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 0.688 | 0.750 | 0.594 | 0.719 | 0.688 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.500 | 0.222 | 0.333 | 0.333 | 0.167 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.438 | 0.625 | 0.656 | 0.781 | 0.781 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 0.667 | 0.417 | 0.472 | 0.611 | 0.556 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.421 | 0.158 | 0.079 | 0.105 | 0.026 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.400 | 0.675 | 0.375 | 0.375 | 0.475 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.269 | 0.192 | 0.500 | 0.192 | 0.462 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.375 | 0.375 | 0.625 | 0.375 | 0.375 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 0.588 | 0.706 | 0.529 | 0.647 | 0.706 |

### Completeness

| model | google/gemma-3-4b_win_rate | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- | --- |
| GPT4o-mini | 0.600 | 0.733 | 0.867 | 0.900 | 0.900 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 0.688 | 0.781 | 0.781 | 0.844 | 0.812 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.444 | 0.167 | 0.250 | 0.222 | 0.167 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.531 | 0.594 | 0.750 | 0.719 | 0.781 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 0.500 | 0.500 | 0.500 | 0.694 | 0.611 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.526 | 0.158 | 0.184 | 0.053 | 0.026 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.400 | 0.725 | 0.475 | 0.375 | 0.375 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.385 | 0.192 | 0.385 | 0.269 | 0.538 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.375 | 0.438 | 0.250 | 0.375 | 0.375 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 0.559 | 0.706 | 0.647 | 0.647 | 0.588 |

### Newsworthiness

| model | google/gemma-3-4b_win_rate | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- | --- |
| GPT4o-mini | 0.733 | 0.700 | 0.767 | 0.867 | 0.833 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 0.500 | 0.688 | 0.812 | 0.812 | 0.656 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.444 | 0.278 | 0.333 | 0.361 | 0.306 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.406 | 0.562 | 0.688 | 0.531 | 0.594 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 0.556 | 0.417 | 0.444 | 0.500 | 0.556 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.316 | 0.211 | 0.316 | 0.211 | 0.184 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.575 | 0.700 | 0.500 | 0.525 | 0.650 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.423 | 0.462 | 0.385 | 0.269 | 0.500 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.469 | 0.438 | 0.281 | 0.406 | 0.281 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 0.588 | 0.588 | 0.529 | 0.559 | 0.500 |

---

## 2. Bradley–Terry strengths (θ)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Faithfulness

| model | google/gemma-3-4b_theta | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- | --- |
| GPT4o-mini | 1.3943 | 3.6538 | 7.3923 | 19999509.8049 | 118.9748 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 1.3908 | 3.1543 | 1.3554 | 0.2720 | 8.2853 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 1.4959 | 0.6677 | 0.3867 | 0.0829 | 1.6316 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 1.0325 | 1.7508 | 1.9760 | 1.2338 | 15.0013 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 1.6156 | 0.7408 | 0.8502 | 0.3801 | 8.0890 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.5466 | 0.1440 | 0.1188 | 0.0296 | 0.0000 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.8601 | 2.3181 | 0.8825 | 0.1555 | 4.5320 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.5399 | 0.1941 | 1.2226 | 0.0421 | 1.1514 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.5829 | 1.0164 | 1.8758 | 0.1071 | 2.8305 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 1.3967 | 1.5207 | 0.6390 | 0.2275 | 7.9983 |

### Correctness

| model | google/gemma-3-4b_theta | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- | --- |
| GPT4o-mini | 1.7863 | 6.4661 | 1474592.6963 | 14.8311 | 7.8708 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 1.6492 | 2.7599 | 0.3301 | 1.5068 | 1.8760 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 1.0537 | 0.3370 | 0.1540 | 0.5533 | 0.2977 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.7339 | 1.3724 | 0.4896 | 3.2358 | 3.3015 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 2.1844 | 1.1553 | 0.2116 | 2.4976 | 1.9872 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.7895 | 0.2260 | 0.0298 | 0.1340 | 0.0387 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.8113 | 2.5220 | 0.1291 | 0.6378 | 1.1669 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.3654 | 0.2028 | 0.2319 | 0.1670 | 0.7642 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.5722 | 0.4713 | 0.7606 | 0.4873 | 0.4921 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 1.5004 | 1.9247 | 0.1895 | 1.4389 | 2.0423 |

### Completeness

| model | google/gemma-3-4b_theta | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- | --- |
| GPT4o-mini | 1.3970 | 2.0878 | 4.5131 | 10.8325 | 7.8435 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 1.7803 | 4.1861 | 2.6668 | 3.4440 | 3.4550 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.8251 | 0.2378 | 0.3951 | 0.3291 | 0.2692 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 1.0635 | 1.2686 | 2.3814 | 2.3089 | 3.2821 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 1.0419 | 1.5680 | 1.3037 | 3.8842 | 2.0297 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 1.1361 | 0.2335 | 0.2594 | 0.0657 | 0.0382 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.7426 | 3.7906 | 1.0839 | 0.7633 | 0.7469 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.6409 | 0.2096 | 0.5392 | 0.2576 | 1.0866 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.6730 | 0.5830 | 0.2984 | 0.4499 | 0.5903 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 1.2086 | 2.2372 | 1.4972 | 1.5634 | 1.1234 |

### Newsworthiness

| model | google/gemma-3-4b_theta | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- | --- |
| GPT4o-mini | 2.5018 | 1.5419 | 2.2758 | 4.7788 | 3.9830 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 0.9143 | 2.4050 | 3.3882 | 3.2687 | 1.6715 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.8816 | 0.4813 | 0.5429 | 0.6252 | 0.4870 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.6412 | 1.0818 | 1.7786 | 0.9248 | 1.1806 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 1.5188 | 0.8159 | 0.8985 | 1.1817 | 1.7422 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.5319 | 0.3047 | 0.5096 | 0.3110 | 0.2595 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 1.3711 | 2.4566 | 1.1314 | 1.2174 | 2.1974 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.6795 | 0.9600 | 0.6190 | 0.3573 | 0.8705 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.7734 | 0.7182 | 0.4337 | 0.6641 | 0.3401 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 1.3285 | 1.2301 | 0.9659 | 1.0431 | 0.8884 |

---

## Export layout

- `json/` — pairwise rows and per-judge G-Eval tables (JSON)
- `tables/` — CSV summaries (win rates, Bradley–Terry)
- `reports/` — Markdown / LaTeX for reading and papers

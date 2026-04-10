# G-Eval results summary

**Judges:** `google/gemma-3-4b`, `gpt-3.5-turbo`, `google/gemini-2.5-flash-preview-05-20`, `anthropic/claude-3-5-haiku-20241022`, `mistral-medium-latest`
**Dimensions:** `faithfulness`, `correctness`, `completeness`, `newsworthiness`
**Documents in subset:** 20 distinct `doc_id`.
**Datapoints:** 1600 pairwise judgments total (80 rows per G-Eval table × 20 table(s), one per judge × dimension).
Equivalent to 80 pair comparisons × 4 dimensions × 5 judges.

Bradley–Terry: `GPT4o-mini` labels gold summaries (JSONL `reference`). Exported θ use mean-centered β (geom. mean θ = 1); odds vs any other model match the fitted BT model.

---

## 1. Pairwise win rates

### Faithfulness

| model | google/gemma-3-4b_win_rate | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- | --- |
| GPT4o-mini | 0.600 | 0.833 | 0.900 | 1.000 | 0.967 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 0.667 | 0.800 | 0.633 | 0.733 | 0.700 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.594 | 0.250 | 0.156 | 0.250 | 0.250 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.500 | 0.656 | 0.688 | 0.844 | 0.781 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 0.611 | 0.361 | 0.500 | 0.528 | 0.528 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.368 | 0.105 | 0.079 | 0.105 | 0.000 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.395 | 0.658 | 0.500 | 0.395 | 0.500 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.409 | 0.182 | 0.591 | 0.273 | 0.227 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.375 | 0.500 | 0.594 | 0.375 | 0.438 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 0.500 | 0.667 | 0.500 | 0.567 | 0.667 |

### Correctness

| model | google/gemma-3-4b_win_rate | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- | --- |
| GPT4o-mini | 0.633 | 0.900 | 0.933 | 0.933 | 0.900 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 0.733 | 0.800 | 0.633 | 0.767 | 0.667 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.500 | 0.188 | 0.250 | 0.312 | 0.188 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.438 | 0.625 | 0.688 | 0.781 | 0.781 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 0.667 | 0.417 | 0.528 | 0.611 | 0.556 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.421 | 0.158 | 0.079 | 0.105 | 0.026 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.368 | 0.658 | 0.395 | 0.342 | 0.500 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.318 | 0.227 | 0.500 | 0.227 | 0.455 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.375 | 0.375 | 0.625 | 0.375 | 0.375 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 0.533 | 0.667 | 0.500 | 0.600 | 0.667 |

### Completeness

| model | google/gemma-3-4b_win_rate | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- | --- |
| GPT4o-mini | 0.600 | 0.733 | 0.867 | 0.900 | 0.900 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 0.733 | 0.767 | 0.767 | 0.833 | 0.800 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.438 | 0.188 | 0.281 | 0.250 | 0.188 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.531 | 0.594 | 0.688 | 0.688 | 0.781 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 0.500 | 0.500 | 0.500 | 0.694 | 0.611 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.526 | 0.158 | 0.211 | 0.053 | 0.026 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.421 | 0.711 | 0.474 | 0.342 | 0.395 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.364 | 0.227 | 0.455 | 0.318 | 0.455 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.375 | 0.438 | 0.281 | 0.406 | 0.375 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 0.500 | 0.667 | 0.567 | 0.600 | 0.600 |

### Newsworthiness

| model | google/gemma-3-4b_win_rate | gpt-3.5-turbo_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- | --- |
| GPT4o-mini | 0.733 | 0.700 | 0.767 | 0.867 | 0.833 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 0.533 | 0.667 | 0.800 | 0.800 | 0.633 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.438 | 0.312 | 0.312 | 0.406 | 0.281 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.406 | 0.562 | 0.719 | 0.531 | 0.594 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 0.556 | 0.417 | 0.444 | 0.500 | 0.556 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.316 | 0.211 | 0.342 | 0.211 | 0.184 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.605 | 0.737 | 0.526 | 0.500 | 0.658 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.409 | 0.455 | 0.318 | 0.318 | 0.500 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.469 | 0.438 | 0.312 | 0.406 | 0.281 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 0.533 | 0.533 | 0.467 | 0.500 | 0.533 |

---

## 2. Bradley–Terry strengths (θ)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Faithfulness

| model | google/gemma-3-4b_theta | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- | --- |
| GPT4o-mini | 1.3793 | 3.4327 | 7.6225 | 3888144.2955 | 125.0768 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 1.6008 | 4.5470 | 1.7433 | 0.4240 | 7.7259 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 1.4427 | 0.4081 | 0.2184 | 0.0815 | 1.7013 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.9154 | 1.7116 | 2.0185 | 1.4558 | 15.2265 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 1.5058 | 0.7348 | 0.9586 | 0.4174 | 8.7650 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.5953 | 0.1461 | 0.1241 | 0.0361 | 0.0000 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.7839 | 2.4269 | 0.9042 | 0.1756 | 5.6122 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.7104 | 0.2028 | 1.3769 | 0.0615 | 0.6973 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.6127 | 0.9909 | 1.9775 | 0.1370 | 2.5430 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 1.1213 | 1.7515 | 0.5829 | 0.2292 | 7.4494 |

### Correctness

| model | google/gemma-3-4b_theta | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- | --- |
| GPT4o-mini | 1.7248 | 6.3078 | 13.0934 | 14.0576 | 8.2066 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 1.9932 | 4.0722 | 1.4491 | 1.9550 | 1.7918 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.9752 | 0.2412 | 0.3603 | 0.4670 | 0.3250 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.7261 | 1.3594 | 1.9975 | 3.1722 | 3.3565 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 2.0377 | 1.1301 | 0.9920 | 2.3172 | 2.0730 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.7755 | 0.2263 | 0.1133 | 0.1369 | 0.0383 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.7537 | 2.6507 | 0.5388 | 0.6033 | 1.3552 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.4634 | 0.2110 | 0.8768 | 0.2024 | 0.6582 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.5959 | 0.4671 | 2.3622 | 0.5139 | 0.4824 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 1.2491 | 1.7767 | 0.5838 | 1.2336 | 1.8231 |

### Completeness

| model | google/gemma-3-4b_theta | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- | --- |
| GPT4o-mini | 1.3923 | 2.0778 | 4.4376 | 10.1045 | 8.3532 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 2.2062 | 3.9829 | 2.5324 | 3.2504 | 3.2265 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.7473 | 0.2583 | 0.4312 | 0.3713 | 0.2992 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 1.0652 | 1.2642 | 1.7162 | 1.8918 | 3.3362 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 1.0372 | 1.5247 | 1.1944 | 3.4460 | 2.2537 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 1.1371 | 0.2339 | 0.3125 | 0.0704 | 0.0379 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.8397 | 3.6205 | 1.0522 | 0.6679 | 0.8681 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.6082 | 0.2339 | 0.7170 | 0.3409 | 0.7071 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.6816 | 0.5947 | 0.3874 | 0.5862 | 0.5437 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 0.9960 | 2.0606 | 1.1022 | 1.3387 | 1.3029 |

### Newsworthiness

| model | google/gemma-3-4b_theta | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- | --- |
| GPT4o-mini | 2.5354 | 1.5925 | 2.5011 | 4.7544 | 4.0569 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 1.0471 | 2.3112 | 3.2054 | 3.0211 | 1.5700 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.8328 | 0.5288 | 0.4449 | 0.7095 | 0.4380 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.6452 | 1.0964 | 2.1616 | 0.9272 | 1.1805 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 1.5419 | 0.8334 | 0.9514 | 1.1197 | 1.8371 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.5343 | 0.3031 | 0.6075 | 0.3069 | 0.2630 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 1.6015 | 2.9355 | 1.3022 | 1.1217 | 2.3394 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.6153 | 0.8586 | 0.4421 | 0.4503 | 0.7779 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.7812 | 0.7174 | 0.5098 | 0.6921 | 0.3269 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 1.1054 | 1.0261 | 0.7646 | 0.8809 | 1.0567 |

---

## Export layout

- `json/` — pairwise rows and per-judge G-Eval tables (JSON)
- `tables/` — CSV summaries (win rates, Bradley–Terry)
- `reports/` — Markdown / LaTeX for reading and papers

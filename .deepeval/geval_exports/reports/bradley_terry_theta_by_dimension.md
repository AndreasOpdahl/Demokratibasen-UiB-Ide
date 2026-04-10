# Bradley–Terry strengths (θ = exp(β), mean-centered β)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Faithfulness

| model | google/gemma-3-4b_theta | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.3793 | 3.4327 | 7.6225 | 3888144.2955 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 1.6008 | 4.5470 | 1.7433 | 0.4240 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 1.4427 | 0.4081 | 0.2184 | 0.0815 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.9154 | 1.7116 | 2.0185 | 1.4558 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 1.5058 | 0.7348 | 0.9586 | 0.4174 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.5953 | 0.1461 | 0.1241 | 0.0361 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.7839 | 2.4269 | 0.9042 | 0.1756 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.7104 | 0.2028 | 1.3769 | 0.0615 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.6127 | 0.9909 | 1.9775 | 0.1370 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 1.1213 | 1.7515 | 0.5829 | 0.2292 |

### Correctness

| model | google/gemma-3-4b_theta | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.7248 | 6.3078 | 13.0934 | 14.0576 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 1.9932 | 4.0722 | 1.4491 | 1.9550 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.9752 | 0.2412 | 0.3603 | 0.4670 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.7261 | 1.3594 | 1.9975 | 3.1722 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 2.0377 | 1.1301 | 0.9920 | 2.3172 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.7755 | 0.2263 | 0.1133 | 0.1369 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.7537 | 2.6507 | 0.5388 | 0.6033 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.4634 | 0.2110 | 0.8768 | 0.2024 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.5959 | 0.4671 | 2.3622 | 0.5139 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 1.2491 | 1.7767 | 0.5838 | 1.2336 |

### Completeness

| model | google/gemma-3-4b_theta | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.3923 | 2.0778 | 4.4376 | 10.1045 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 2.2062 | 3.9829 | 2.5324 | 3.2504 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.7473 | 0.2583 | 0.4312 | 0.3713 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 1.0652 | 1.2642 | 1.7162 | 1.8918 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 1.0372 | 1.5247 | 1.1944 | 3.4460 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 1.1371 | 0.2339 | 0.3125 | 0.0704 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.8397 | 3.6205 | 1.0522 | 0.6679 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.6082 | 0.2339 | 0.7170 | 0.3409 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.6816 | 0.5947 | 0.3874 | 0.5862 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 0.9960 | 2.0606 | 1.1022 | 1.3387 |

### Newsworthiness

| model | google/gemma-3-4b_theta | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 2.5354 | 1.5925 | 2.5011 | 4.7544 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 1.0471 | 2.3112 | 3.2054 | 3.0211 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.8328 | 0.5288 | 0.4449 | 0.7095 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.6452 | 1.0964 | 2.1616 | 0.9272 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 1.5419 | 0.8334 | 0.9514 | 1.1197 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.5343 | 0.3031 | 0.6075 | 0.3069 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 1.6015 | 2.9355 | 1.3022 | 1.1217 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.6153 | 0.8586 | 0.4421 | 0.4503 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.7812 | 0.7174 | 0.5098 | 0.6921 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 1.1054 | 1.0261 | 0.7646 | 0.8809 |

# Bradley–Terry strengths (θ = exp(β), mean-centered β)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Relevance

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 4.9754 | 5.7341 | 9.2661 | 7.2688 |
| gemma-2-9b__checkpoint-2500-inputs-refs-preds-1000-examples | 2.3709 | 2.6740 | 2.8547 | 2.7669 |
| llama2-13b-chat-norwegian__checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.9042 | 0.7558 | 0.6658 | 0.7141 |
| nb-gpt-j-6b__checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.4701 | 0.4096 | 0.3384 | 0.3540 |
| norskgpt-llama3-8b__checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.2957 | 0.3156 | 0.2327 | 0.2611 |
| norwai-mistral-7b__checkpoint-9000-inputs-refs-preds-1000-examples | 0.4321 | 0.4447 | 0.3857 | 0.4363 |
| viking-13b__checkpoint-3500-inputs-refs-preds-1000-examples | 1.5607 | 1.5013 | 1.8698 | 1.7266 |

### Consistency

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 2.6513 | 2.5911 | 4.6798 | 4.3004 |
| gemma-2-9b__checkpoint-2500-inputs-refs-preds-1000-examples | 1.7240 | 1.5638 | 2.2481 | 1.9047 |
| llama2-13b-chat-norwegian__checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.9391 | 0.9209 | 0.9069 | 0.9260 |
| nb-gpt-j-6b__checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.4676 | 0.3948 | 0.3614 | 0.4200 |
| norskgpt-llama3-8b__checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.5652 | 0.7371 | 0.3582 | 0.4026 |
| norwai-mistral-7b__checkpoint-9000-inputs-refs-preds-1000-examples | 0.7455 | 0.8092 | 0.4987 | 0.5736 |
| viking-13b__checkpoint-3500-inputs-refs-preds-1000-examples | 1.1822 | 1.1382 | 1.6230 | 1.3594 |

### Newsworthiness

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 2.6939 | 2.9665 | 3.6933 | 2.8158 |
| gemma-2-9b__checkpoint-2500-inputs-refs-preds-1000-examples | 1.5909 | 1.7336 | 1.9387 | 1.4441 |
| llama2-13b-chat-norwegian__checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 1.2160 | 0.9907 | 0.9207 | 0.9867 |
| nb-gpt-j-6b__checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.8040 | 0.7332 | 0.6907 | 0.7771 |
| norskgpt-llama3-8b__checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.3234 | 0.3647 | 0.2833 | 0.4058 |
| norwai-mistral-7b__checkpoint-9000-inputs-refs-preds-1000-examples | 0.6617 | 0.6258 | 0.5026 | 0.6463 |
| viking-13b__checkpoint-3500-inputs-refs-preds-1000-examples | 1.1153 | 1.1731 | 1.5427 | 1.2231 |

### Hygiene

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 4.2993 | 3.4462 | 5.7737 | 5.1548 |
| gemma-2-9b__checkpoint-2500-inputs-refs-preds-1000-examples | 2.3165 | 2.0978 | 2.9257 | 2.9102 |
| llama2-13b-chat-norwegian__checkpoint-5500-gen0-inputs-refs-preds-1000-examples | 0.6824 | 0.6675 | 0.6539 | 0.7710 |
| nb-gpt-j-6b__checkpoint-9500-gen0-inputs-refs-preds-1000-examples | 0.5811 | 0.5678 | 0.4269 | 0.4888 |
| norskgpt-llama3-8b__checkpoint-8000-gen0-inputs-refs-preds-1000-examples | 0.2141 | 0.2946 | 0.1946 | 0.2254 |
| norwai-mistral-7b__checkpoint-9000-inputs-refs-preds-1000-examples | 0.3708 | 0.4421 | 0.3875 | 0.3739 |
| viking-13b__checkpoint-3500-inputs-refs-preds-1000-examples | 3.1887 | 2.8028 | 2.8126 | 2.0991 |

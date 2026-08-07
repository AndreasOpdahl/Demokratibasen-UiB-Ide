# Bradley–Terry strengths (θ = exp(β), mean-centered β)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Relevance

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.6285 | 1.7735 | 1.8999 | 1.7408 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.8928 | 0.9172 | 0.8203 | 0.8560 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 1.1938 | 1.0238 | 1.0576 | 1.0650 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.5761 | 0.6004 | 0.6067 | 0.6302 |

### Consistency

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.6563 | 1.7918 | 1.8338 | 1.8790 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.9395 | 0.9109 | 0.8858 | 0.8867 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 0.8611 | 0.8303 | 0.9409 | 0.9063 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.7463 | 0.7379 | 0.6543 | 0.6623 |

### Newsworthiness

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.0823 | 1.0283 | 1.0577 | 1.0565 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.8573 | 0.7808 | 0.7807 | 0.7942 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 2.1164 | 2.4534 | 2.3134 | 2.2462 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.5093 | 0.5077 | 0.5235 | 0.5306 |

### Hygiene

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.2542 | 1.2205 | 1.4686 | 1.4735 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.8145 | 0.8338 | 0.8828 | 0.8651 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 1.2208 | 1.1807 | 1.0978 | 1.1821 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.8019 | 0.8322 | 0.7026 | 0.6636 |

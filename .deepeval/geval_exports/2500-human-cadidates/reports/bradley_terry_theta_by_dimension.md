# Bradley–Terry strengths (θ = exp(β), mean-centered β)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Relevance

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.4363 | 1.7262 | 1.8028 | 1.5633 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.8690 | 0.8396 | 0.9326 | 0.8814 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 1.2189 | 1.0135 | 0.9111 | 0.9887 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.6573 | 0.6808 | 0.6528 | 0.7340 |

### Consistency

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.5205 | 1.5943 | 1.6371 | 1.5703 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.9964 | 1.0324 | 0.9339 | 0.9538 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 0.8465 | 0.7372 | 0.8738 | 0.8480 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.7798 | 0.8241 | 0.7485 | 0.7873 |

### Newsworthiness

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.9857 | 0.9402 | 1.0734 | 1.0416 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.8054 | 0.8101 | 0.7282 | 0.7447 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 2.2193 | 2.5345 | 2.2399 | 2.3154 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.5676 | 0.5180 | 0.5711 | 0.5568 |

### Hygiene

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.2843 | 1.2795 | 1.3988 | 1.5178 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.8074 | 0.8267 | 0.9951 | 0.9601 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 1.2842 | 1.1972 | 1.0311 | 1.0553 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.7510 | 0.7896 | 0.6967 | 0.6503 |

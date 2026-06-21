# Bradley–Terry strengths (θ = exp(β), mean-centered β)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Relevance

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.6265 | 1.8115 | 1.9203 | 1.7317 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.8901 | 0.8754 | 0.8284 | 0.8461 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 1.1613 | 0.9801 | 0.9891 | 0.9999 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.5948 | 0.6434 | 0.6356 | 0.6826 |

### Consistency

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.6382 | 1.7085 | 1.7430 | 1.7718 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 1.0053 | 1.0100 | 0.9085 | 0.9183 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 0.7939 | 0.7573 | 0.9322 | 0.8830 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.7648 | 0.7652 | 0.6775 | 0.6960 |

### Newsworthiness

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.0284 | 0.9892 | 1.0483 | 1.0520 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.8577 | 0.7810 | 0.7588 | 0.7600 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 2.2025 | 2.6462 | 2.4118 | 2.3919 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.5147 | 0.4892 | 0.5212 | 0.5229 |

### Hygiene

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.2553 | 1.2082 | 1.4457 | 1.4976 |
| gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples | 0.8024 | 0.8143 | 0.9003 | 0.8881 |
| gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples | 1.2038 | 1.1911 | 1.0724 | 1.1331 |
| viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples | 0.8248 | 0.8533 | 0.7164 | 0.6635 |

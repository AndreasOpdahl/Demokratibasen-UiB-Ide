# Bradley–Terry strengths (θ = exp(β), mean-centered β)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Faithfulness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 14.1331 | 4.5968 | 19.5978 | 289042.0301 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.4762 | 0.8518 | 0.5382 | 0.0593 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.4454 | 0.5190 | 0.5407 | 0.0215 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.5167 | 0.7758 | 0.3655 | 0.0747 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.6456 | 0.6343 | 0.4797 | 0.0363 |

### Correctness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 14.1815 | 5.0123 | 19.7081 | 329088.2662 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.5036 | 0.9237 | 0.4926 | 0.0640 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.4010 | 0.5762 | 0.3913 | 0.0220 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.5495 | 0.5282 | 0.4588 | 0.0637 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.6354 | 0.7096 | 0.5738 | 0.0339 |

### Completeness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 9.4904 | 11.4964 | 1012763.6768 | 14.2377 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.5824 | 0.6236 | 0.0335 | 0.6538 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.3354 | 0.3119 | 0.0279 | 0.3640 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.7468 | 0.7795 | 0.0278 | 0.5959 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.7224 | 0.5737 | 0.0380 | 0.4951 |

### Newsworthiness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 2.9242 | 2.8941 | 4.6568 | 2.8587 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.9327 | 0.8503 | 1.0868 | 1.0554 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.4666 | 0.4864 | 0.4680 | 0.5714 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.1855 | 1.0077 | 0.6689 | 0.8372 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.6628 | 0.8291 | 0.6312 | 0.6929 |

### Hygiene

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 4399485.3620 | 7.8882 | 561856.4268 | 843063.2455 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.0143 | 0.4513 | 0.0230 | 0.0224 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.0122 | 0.6967 | 0.0539 | 0.0343 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.0354 | 0.5670 | 0.0273 | 0.0280 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.0367 | 0.7112 | 0.0527 | 0.0551 |

# Bradley–Terry strengths (θ = exp(β), mean-centered β)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Faithfulness

| model | google/gemma-3-4b_theta |
| --- | --- |
| GPT4o-mini | 1.4890 |
| eurollm-9B-Instruct-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.7305 |
| gemma-2-9b-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.1162 |
| gemma-2b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.3874 |
| gemma-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.7128 |
| llama-2-13b-chat-norwegian-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.1782 |
| llama-3.1-8b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.2883 |
| nb-gpt-j-6b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.7479 |
| normistral-11b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.2139 |
| normistral-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.7770 |
| normistral-7b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.9818 |
| norskgpt-llama3-8b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.7923 |

### Correctness

| model | google/gemma-3-4b_theta |
| --- | --- |
| GPT4o-mini | 2.2514 |
| eurollm-9B-Instruct-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.6310 |
| gemma-2-9b-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.8341 |
| gemma-2b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.2919 |
| gemma-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.5580 |
| llama-2-13b-chat-norwegian-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.3214 |
| llama-3.1-8b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.6453 |
| nb-gpt-j-6b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.4991 |
| normistral-11b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.3226 |
| normistral-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.8025 |
| normistral-7b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.7497 |
| norskgpt-llama3-8b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.6166 |

### Completeness

| model | google/gemma-3-4b_theta |
| --- | --- |
| GPT4o-mini | 1.9771 |
| eurollm-9B-Instruct-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.5352 |
| gemma-2-9b-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.4650 |
| gemma-2b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.9031 |
| gemma-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.5459 |
| llama-2-13b-chat-norwegian-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.3512 |
| llama-3.1-8b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 1.8288 |
| nb-gpt-j-6b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.4285 |
| normistral-11b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 2.0695 |
| normistral-7b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.9744 |
| normistral-7b-instruct-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.9076 |
| norskgpt-llama3-8b-apptainer-checkpoint-5000-inputs-refs-preds-examples_1000 | 0.6753 |

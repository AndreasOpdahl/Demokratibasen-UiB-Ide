# Bradley–Terry strengths (θ = exp(β), mean-centered β)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Faithfulness

| model | google/gemma-3-4b_theta | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta |
| --- | --- | --- | --- |
| GPT4o-mini | 1.7531 | 2.6109 | 5.0107 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 1.0536 | 2.1460 | 1.2774 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.9030 | 0.6409 | 0.3861 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 1.3866 | 1.4467 | 1.0188 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 0.9845 | 0.9233 | 1.1234 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.8387 | 0.4906 | 0.4133 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.9297 | 0.6181 | 0.8026 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.7970 | 0.7278 | 0.6681 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.5093 | 0.6646 | 1.1074 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 1.3879 | 1.4213 | 1.4404 |

### Correctness

| model | google/gemma-3-4b_theta | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta |
| --- | --- | --- | --- |
| GPT4o-mini | 1.6152 | 2.8401 | 1.1003 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 1.0997 | 1.5892 | 0.9708 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.8210 | 0.6034 | 0.9307 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 1.6212 | 1.5596 | 0.9969 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 1.1351 | 0.8525 | 1.0109 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.6909 | 0.5001 | 0.9681 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.6372 | 0.8368 | 1.0606 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.9055 | 0.6411 | 0.9568 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.5916 | 0.6793 | 0.9210 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 1.5799 | 1.5153 | 1.1031 |

### Completeness

| model | google/gemma-3-4b_theta | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta |
| --- | --- | --- | --- |
| GPT4o-mini | 1.0744 | 2.4682 | 2.9436 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 1.0114 | 1.7549 | 1.4490 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 1.1402 | 0.7127 | 0.5518 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 1.8861 | 1.9984 | 1.5916 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 1.1321 | 0.9454 | 0.7931 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.6387 | 0.3922 | 0.4274 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.7436 | 0.9380 | 1.3251 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.8271 | 0.7077 | 0.6779 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.6274 | 0.5940 | 0.8772 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 1.5338 | 1.1089 | 0.9994 |

### Newsworthiness

| model | google/gemma-3-4b_theta | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta |
| --- | --- | --- | --- |
| GPT4o-mini | 0.9428 | 1.7875 | 3.6755 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 1.2843 | 1.9734 | 1.8236 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.6868 | 0.6952 | 0.6251 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 1.2226 | 1.5173 | 1.3889 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 1.0646 | 0.9093 | 0.7897 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.8330 | 0.4436 | 0.4661 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 1.0504 | 1.0539 | 1.1444 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 1.3312 | 0.8413 | 0.5927 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.9625 | 0.7728 | 0.8147 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 0.8240 | 0.9725 | 0.8449 |

# Bradley–Terry strengths (θ = exp(β), mean-centered β)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Faithfulness

| model | google/gemma-3-4b_theta | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta |
| --- | --- | --- | --- |
| GPT4o-mini | 2405869.9244 | 72.7822 | 38147010239596.0078 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 0.0000 | 0.0000 | 0.0042 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.3736 | 72.7786 | 0.0058 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.3774 | 133803300831694.3906 | 0.0115 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 174939781078034624.0000 | 0.0000 | 0.0007 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.0954 | 0.0000 | 0.0016 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.1803 | 133791110897398.4531 | 84532.6279 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.0000 | 0.0000 | 0.0110 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.1888 | 72.7648 | 0.0286 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 0.7229 | 133812497858010.3594 | 0.0030 |

### Correctness

| model | google/gemma-3-4b_theta | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta |
| --- | --- | --- | --- |
| GPT4o-mini | 98467.7470 | 4.8552 | 83540573967732572160.0000 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 0.0029 | 0.1846 | 0.0000 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 0.0128 | 0.7870 | 211461011.3706 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.0109 | 15.5017 | 600484540.8673 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 4549895488234.4727 | 0.3682 | 0.0000 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 0.0047 | 0.0510 | 0.0000 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 0.0032 | 12.4572 | 905188926.2403 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.0017 | 0.0784 | 0.0000 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 0.0104 | 0.3103 | 74.8162 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 0.0220 | 16.0680 | 229812520.9321 |

### Completeness

| model | google/gemma-3-4b_theta | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta |
| --- | --- | --- | --- |
| GPT4o-mini | 25.2077 | 465.0122 | 242373025396482240.0000 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 2.1635 | 0.0000 | 0.0000 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 1.7082 | 258.0346 | 1326.1128 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 3.5411 | 4664.3174 | 1606921176.6118 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 22.5274 | 1784.0112 | 0.0000 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 2.3283 | 0.0000 | 0.0000 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 2.0105 | 5576.1209 | 3716549311.5607 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.0000 | 0.0000 | 0.0000 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 11.1244 | 346.3572 | 0.0032 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 9.4867 | 2058.3190 | 2439623573.8186 |

### Newsworthiness

| model | google/gemma-3-4b_theta | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta |
| --- | --- | --- | --- |
| GPT4o-mini | 27.4563 | 2.8643 | 9042263.1210 |
| gemma-2-9b-checkpoint-9000-inputs-refs-preds-examples_2500 | 2.6332 | 272154119.7128 | 2.4968 |
| gemma-2b-checkpoint-5000-inputs-refs-preds-examples_2500 | 26.6029 | 1.6146 | 0.4731 |
| llama-2-13b-chat-norwegian-checkpoint-3000-inputs-refs-preds-examples_2500 | 1.7375 | 1.5792 | 4.4973 |
| llama-3.1-8b-instruct-checkpoint-500-inputs-refs-preds-examples_2500 | 5.7617 | 6.0923 | 0.6871 |
| nb-gpt-j-6b-checkpoint-3000-inputs-refs-preds-examples_2500 | 2.4590 | 0.2184 | 1.5652 |
| normistral-11b-checkpoint-1000-inputs-refs-preds-examples_2500 | 2.8197 | 9.0123 | 3.9817 |
| norskgpt-llama3-8b-checkpoint-500-inputs-refs-preds-examples_2500 | 0.0000 | 0.0000 | 0.0000 |
| norwai-mistral-7b-instruct-checkpoint-8000-inputs-refs-preds-examples_2500 | 5.1808 | 0.4266 | 0.8608 |
| viking-13b-checkpoint-4000-inputs-refs-preds-examples_2500 | 6.9224 | 0.3134 | 0.5672 |

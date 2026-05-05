# Bradley–Terry strengths (θ = exp(β), mean-centered β)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Faithfulness

| model | mistral-medium-latest_theta |
| --- | --- |
| GPT4o-mini | 34.2000 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.8289 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.1164 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.4968 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.2612 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.5862 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.5935 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.5197 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.4658 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.6843 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.2120 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.4051 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.4153 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.4926 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.2422 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 1.3390 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.9136 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 1.8689 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.2330 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.8255 |

### Correctness

| model | mistral-medium-latest_theta |
| --- | --- |
| GPT4o-mini | 1895409.6989 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.3103 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.6876 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.4724 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.6604 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.3251 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.4924 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.4390 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.2709 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.3962 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.7234 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.2316 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.3256 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.1734 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.7155 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.7727 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.5492 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 1.4802 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.7318 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.3379 |

### Completeness

| model | mistral-medium-latest_theta |
| --- | --- |
| GPT4o-mini | 14.3795 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.8763 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.8009 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.8792 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 2.4196 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.4921 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.7005 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.8795 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.0902 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.4965 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.8568 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.5469 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.4313 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.4275 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.9330 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 1.3094 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.0559 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.9748 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.8313 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 1.0426 |

### Newsworthiness

| model | mistral-medium-latest_theta |
| --- | --- |
| GPT4o-mini | 5.8289 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.3958 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.6852 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.9064 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.8448 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.5350 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.7785 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.1490 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.6834 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.9265 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 2.1026 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.6090 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.7475 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.6418 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.7805 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 1.5843 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.1607 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 1.0768 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.8325 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.6411 |

### Hygiene

| model | mistral-medium-latest_theta |
| --- | --- |
| GPT4o-mini | 48.7808 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.3159 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.0034 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.2339 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.6969 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.4118 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.9708 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.8612 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 2.6436 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.8180 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.3577 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.3134 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.5548 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.4991 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.1591 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.7094 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.7028 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.6105 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.6124 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.4396 |

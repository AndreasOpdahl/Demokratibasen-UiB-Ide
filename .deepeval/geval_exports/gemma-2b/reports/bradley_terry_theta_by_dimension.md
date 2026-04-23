# Bradley–Terry strengths (θ = exp(β), mean-centered β)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Faithfulness

| model | mistral-medium-latest_theta |
| --- | --- |
| GPT4o-mini | 26.9962 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 1.3858 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.2106 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.3602 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.9439 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.4320 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.4003 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.4843 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.1008 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.9007 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.6092 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.3692 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.3992 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.6084 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.0080 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 1.5532 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.5596 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 2.1848 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.3096 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.5724 |

### Correctness

| model | mistral-medium-latest_theta |
| --- | --- |
| GPT4o-mini | 941679.3458 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.3438 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.8495 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.3512 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.4611 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.2214 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.2777 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.4119 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.1035 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.4529 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.4334 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.2001 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.3587 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.1853 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.6156 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.8843 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.3550 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 2.5580 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.8300 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.2269 |

### Completeness

| model | mistral-medium-latest_theta |
| --- | --- |
| GPT4o-mini | 32.8640 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.5316 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.8834 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.5565 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 2.4957 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.2335 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.6573 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.8880 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.0600 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.3667 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.8776 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.5929 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.4504 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.5371 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.7986 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 1.4799 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.6897 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 1.1797 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.4009 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.6514 |

### Newsworthiness

| model | mistral-medium-latest_theta |
| --- | --- |
| GPT4o-mini | 21.8489 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.3956 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.6952 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.4896 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.8544 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.4661 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.6896 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.0374 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 4.7819 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.7041 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 3.3476 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.4860 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.6300 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.4819 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.8267 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 1.7313 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.6502 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 1.2868 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.7872 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.3507 |

### Hygiene

| model | mistral-medium-latest_theta |
| --- | --- |
| GPT4o-mini | 997243.9315 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.2479 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.5175 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.6659 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.5638 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.7901 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.6819 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.5361 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.5852 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.7392 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.6606 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.2227 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.3985 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.4691 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.6447 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.4634 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.4615 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.1755 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.8670 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.2888 |

# Bradley–Terry strengths (θ = exp(β), mean-centered β)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Faithfulness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.8477 | 1.7933 | 1.3825 | 2.5036 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.5300 | 3.0704 | 1.0879 | 1.6705 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.9654 | 2.1323 | 1.2625 | 0.6774 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 4.4739 | 0.4724 | 0.9075 | 1.4538 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.2678 | 0.2817 | 0.8743 | 0.4861 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.2560 | 1.6936 | 1.3446 | 1.4983 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.8604 | 0.9204 | 0.7342 | 1.4673 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.0191 | 2.4309 | 0.8405 | 1.0333 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.8025 | 1.0678 | 0.9626 | 1.4139 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.8375 | 0.3239 | 0.8410 | 0.7138 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.5642 | 0.8845 | 0.1888 | 0.4938 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 1.6116 | 1.1521 | 3.0138 | 0.9796 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 1.6016 | 0.9767 | 0.6014 | 0.5568 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 1.4437 | 0.8003 | 0.9008 | 0.9524 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.1962 | 1.0098 | 0.6994 | 1.1977 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.9446 | 1.2976 | 2.0209 | 1.6169 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.5637 | 0.3131 | 0.4693 | 1.1005 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.2977 | 0.3101 | 0.6057 | 0.1521 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.9142 | 1.6438 | 3.0908 | 1.0452 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.3328 | 2.9307 | 2.5818 | 2.5070 |

### Correctness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.5526 | 1.9183 | 2.2703 | 1.6121 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.7085 | 2.9149 | 0.6923 | 1.2220 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.6423 | 3.4832 | 2.0812 | 0.7227 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 3.3312 | 0.9804 | 1.4780 | 1.3440 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.0963 | 0.5643 | 0.9217 | 0.3681 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.7096 | 1.5513 | 2.4747 | 1.4646 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.4188 | 0.7827 | 0.8104 | 1.5165 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.5328 | 1.6359 | 0.7170 | 1.3686 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.3965 | 1.0624 | 1.5032 | 1.0718 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.0409 | 0.3317 | 0.7603 | 0.6958 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.3841 | 0.4619 | 0.1947 | 0.4492 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 2.3454 | 1.3578 | 2.0047 | 1.0067 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 1.6915 | 1.2582 | 0.5224 | 0.5660 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 1.4409 | 0.8192 | 0.8449 | 0.7371 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.7612 | 0.7753 | 0.6140 | 1.3373 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.9328 | 0.8506 | 2.1549 | 1.6938 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.6128 | 0.2683 | 0.4235 | 1.7242 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.4102 | 0.3507 | 0.7235 | 0.2432 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.8628 | 1.1337 | 1.4099 | 1.7837 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.9950 | 2.9162 | 1.3865 | 1.9599 |

### Completeness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 2.0217 | 1.6389 | 1.8697 | 1.2453 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.7005 | 1.7102 | 0.8920 | 1.2479 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.1604 | 0.9176 | 1.1722 | 1.2314 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 2.5331 | 2.7144 | 1.9060 | 3.0101 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.8120 | 0.4162 | 0.8147 | 0.6112 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.5207 | 3.5096 | 3.7075 | 2.5174 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.8419 | 1.5499 | 0.7133 | 1.4068 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.5061 | 0.7731 | 0.5478 | 0.4725 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.4386 | 1.0090 | 3.0456 | 1.1603 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.0525 | 1.4653 | 1.1174 | 0.9263 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.6190 | 0.2023 | 0.2685 | 0.8830 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.8223 | 1.2117 | 1.8293 | 0.7387 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 1.1066 | 0.5310 | 0.5894 | 0.3956 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.7861 | 0.1913 | 0.6615 | 0.2383 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.1729 | 2.6819 | 0.8454 | 1.1904 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 1.2505 | 1.4560 | 1.8231 | 2.0362 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.4751 | 0.6736 | 0.5221 | 1.4803 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.2878 | 0.2502 | 0.6337 | 0.5138 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.7618 | 1.4617 | 1.4114 | 1.0442 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.8815 | 2.3120 | 0.4847 | 1.3342 |

### Newsworthiness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 2.4303 | 1.8533 | 4.7158 | 1.6237 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 1.6367 | 1.3493 | 0.4354 | 0.3569 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.6865 | 1.5578 | 1.0565 | 0.9531 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.6771 | 1.5463 | 1.1880 | 1.3333 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.7938 | 0.5205 | 0.9203 | 0.7951 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.7650 | 1.1002 | 0.8838 | 1.7908 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.9360 | 1.4034 | 0.7557 | 0.6574 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.3582 | 0.2969 | 0.4312 | 0.4017 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 4.0575 | 2.4525 | 1.3197 | 1.2079 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.8467 | 0.8647 | 0.8627 | 1.0070 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.2223 | 1.1556 | 0.3534 | 0.6188 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.4368 | 0.7884 | 0.9836 | 0.8820 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.6591 | 0.6055 | 1.1746 | 0.8492 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.2001 | 0.3613 | 0.3715 | 0.4148 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.2557 | 0.9961 | 1.5372 | 0.9882 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 1.4757 | 2.3639 | 3.4486 | 2.6600 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.4005 | 0.4427 | 0.7930 | 1.2242 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 3.4557 | 1.8449 | 3.7065 | 4.4477 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.9283 | 1.0206 | 1.3366 | 1.1411 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.5327 | 0.8384 | 0.4069 | 0.9455 |

### Hygiene

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.7966 | 1.1173 | 4.5840 | 1.6175 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 1.6182 | 0.6721 | 0.1526 | 0.6495 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.3403 | 1.2377 | 2.6558 | 0.9005 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 2.4931 | 0.6913 | 4.7569 | 0.8936 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.9920 | 0.9064 | 0.9991 | 0.7790 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.2170 | 1.5674 | 1.5918 | 2.0569 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.9512 | 1.1404 | 0.8513 | 0.6641 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.3511 | 0.8884 | 0.7406 | 0.8410 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.7738 | 1.0564 | 1.7030 | 1.4265 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.5274 | 0.8432 | 1.4114 | 0.8684 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.4063 | 0.7607 | 0.4345 | 1.5388 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.7838 | 0.8521 | 1.6576 | 0.9823 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.8045 | 1.1545 | 0.9917 | 0.6467 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.4439 | 1.3766 | 0.6434 | 0.8581 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.0478 | 1.5503 | 0.2669 | 0.9085 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 1.9841 | 1.3002 | 2.3102 | 1.2522 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.6941 | 1.0131 | 0.4496 | 2.0759 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 1.5520 | 0.8585 | 0.8664 | 1.0662 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.4149 | 0.7514 | 0.8296 | 0.5325 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 1.0174 | 0.8947 | 0.5128 | 0.9486 |

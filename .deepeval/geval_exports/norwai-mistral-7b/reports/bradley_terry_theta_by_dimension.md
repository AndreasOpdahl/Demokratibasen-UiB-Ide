# Bradley–Terry strengths (θ = exp(β), mean-centered β)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Faithfulness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 3.0506 | 1.1142 | 1618688.8833 | 28.1653 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 1.0002 | 0.5472 | 0.7518 | 2.1018 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.4794 | 1.3754 | 0.3096 | 0.4800 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 2.2674 | 0.8900 | 0.5991 | 0.7999 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.0219 | 0.7201 | 0.3431 | 0.7758 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.4828 | 1.1563 | 5.8516 | 1.7701 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.8988 | 0.5105 | 0.2765 | 1.3597 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.4348 | 1.0049 | 0.5945 | 0.7749 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.1011 | 1.1166 | 0.5181 | 0.5823 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.9780 | 0.7004 | 0.6398 | 1.0105 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.3149 | 0.9122 | 0.0747 | 0.0712 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.8944 | 1.1138 | 0.6404 | 0.6173 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.8997 | 0.8687 | 0.6241 | 0.9610 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 1.3080 | 0.7765 | 0.6716 | 0.7620 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.0286 | 1.0237 | 0.6267 | 0.6155 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.5576 | 1.6713 | 0.2206 | 1.1874 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.6947 | 0.6474 | 0.3856 | 1.6175 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.5652 | 2.7715 | 0.3685 | 0.8943 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 3.4200 | 1.8368 | 0.4321 | 1.1203 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.5586 | 1.0379 | 0.2382 | 1.3548 |

### Correctness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 2.8839 | 8.8178 | 5671177.5557 | 15.6995 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.9396 | 0.4824 | 0.3698 | 2.1161 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.2740 | 0.5510 | 0.4893 | 0.5436 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 2.2432 | 0.8126 | 0.2224 | 1.0329 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.0523 | 1.6905 | 0.4889 | 1.6109 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.1456 | 3.0662 | 5.6993 | 1.3840 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.1135 | 0.7919 | 0.4329 | 1.5705 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.4465 | 0.6444 | 0.5517 | 0.7105 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.5979 | 1.5274 | 0.7244 | 0.6946 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.0369 | 1.7249 | 0.2817 | 0.3826 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.3006 | 0.3304 | 0.0778 | 0.1609 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 1.1231 | 0.3253 | 0.3728 | 0.4814 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.8447 | 0.9047 | 0.3730 | 0.9081 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 1.4746 | 0.4377 | 0.1876 | 0.3192 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.4895 | 0.5860 | 0.6713 | 0.6302 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.5683 | 1.1733 | 0.2056 | 1.5237 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.0502 | 2.1026 | 0.7017 | 2.7741 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.8649 | 0.5041 | 0.8458 | 0.9899 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.8931 | 2.0487 | 0.7702 | 0.9751 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.5332 | 1.1856 | 0.2522 | 1.4047 |

### Completeness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 3.6018 | 6.2788 | 1470595.1185 | 35.5955 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 1.0468 | 2.3555 | 0.4758 | 1.6111 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.1021 | 1.0699 | 0.7724 | 2.7868 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.9259 | 1.8404 | 0.3708 | 1.0651 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.5561 | 1.0884 | 0.2751 | 1.0192 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.9610 | 3.8832 | 1.8071 | 0.8890 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.7481 | 0.9549 | 0.4494 | 0.8154 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.5848 | 1.0715 | 0.5446 | 0.7100 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.1150 | 0.1783 | 1.2415 | 11.6345 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.6893 | 0.5121 | 0.1555 | 0.4275 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.3002 | 0.4715 | 0.1640 | 0.5575 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.9504 | 0.6244 | 0.4179 | 0.3044 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 1.1812 | 1.5483 | 0.6816 | 0.4515 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.9672 | 0.9706 | 0.5666 | 0.4382 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.7761 | 0.4115 | 0.3379 | 0.2820 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.4980 | 0.4183 | 0.2485 | 0.6589 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.1427 | 0.7654 | 0.4188 | 0.6345 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.4478 | 2.1081 | 0.5693 | 0.8731 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 2.5073 | 1.2687 | 1.1954 | 0.8002 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.7782 | 0.5579 | 0.3355 | 0.8143 |

### Newsworthiness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 5.3242 | 8.2688 | 9.4487 | 9.4842 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 2.2972 | 4.3269 | 1.0032 | 1.6929 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.4416 | 0.7767 | 1.3731 | 1.1235 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 2.8142 | 1.0188 | 0.4865 | 2.5149 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.6156 | 5.3808 | 0.8457 | 1.0764 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.5163 | 1.7567 | 1.5417 | 0.6350 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.4215 | 0.6113 | 1.1181 | 0.7786 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.7147 | 1.1564 | 1.2407 | 2.4859 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.2342 | 0.4128 | 2.0348 | 1.8696 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.7890 | 0.4376 | 0.3871 | 0.3613 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.2102 | 0.3962 | 0.3006 | 0.2517 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 1.1766 | 1.7273 | 1.3976 | 1.4262 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.5516 | 0.6940 | 1.1038 | 0.7141 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.5340 | 0.4062 | 0.6633 | 0.2964 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 2.5245 | 0.1650 | 0.8915 | 0.6631 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 2.3649 | 0.3681 | 0.6318 | 0.4459 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.1939 | 0.5984 | 0.4442 | 0.5132 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 2.4629 | 1.4552 | 0.7261 | 1.7036 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 2.0940 | 1.9470 | 1.7978 | 1.4106 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 1.2140 | 1.4732 | 1.1034 | 0.8905 |

### Hygiene

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 2931481.3084 | 6.2843 | 2365650.0057 | 345273908639772.0000 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 1.1576 | 2.5448 | 0.2941 | 13239.4914 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.3433 | 0.8367 | 0.8436 | 0.0676 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.5129 | 0.2186 | 0.0866 | 0.0896 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.2842 | 2.0370 | 0.4140 | 0.0364 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.6825 | 1.8315 | 0.7637 | 0.0827 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.4734 | 0.7627 | 0.7160 | 0.2152 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.4937 | 0.2772 | 0.2211 | 0.0221 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.2263 | 0.2089 | 0.4345 | 0.2144 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.7498 | 1.1277 | 0.8154 | 0.6545 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.1492 | 0.8169 | 0.2238 | 0.0754 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.4560 | 0.3433 | 0.5453 | 0.0678 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.2873 | 0.5788 | 2.2745 | 0.0695 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.5130 | 1.2307 | 0.1601 | 0.0371 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.5046 | 1.5413 | 1.1200 | 0.0541 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.4316 | 1.3600 | 0.4675 | 0.2010 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.2257 | 2.3406 | 2.8105 | 0.4604 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.5636 | 0.4155 | 0.1073 | 0.0629 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.8846 | 2.0019 | 0.5617 | 0.1528 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.3001 | 2.2570 | 0.2816 | 0.0283 |

# Bradley–Terry strengths (θ = exp(β), mean-centered β)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Faithfulness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 2.2786 | 2.8504 | 3.6999 | 3.4551 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.7576 | 0.8869 | 1.3036 | 0.5716 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.7638 | 0.8620 | 0.8122 | 0.7654 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.0333 | 1.4358 | 1.1798 | 1.4305 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.4586 | 1.1033 | 0.7709 | 1.0632 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.2225 | 1.1131 | 0.9966 | 0.8305 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.5332 | 0.8490 | 1.2336 | 1.7100 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.4926 | 0.8179 | 0.8061 | 0.7573 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.2794 | 0.9241 | 1.2221 | 0.9843 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.1381 | 1.4719 | 1.4438 | 1.8301 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.3581 | 0.5431 | 0.5959 | 0.6050 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.9497 | 0.6515 | 0.7627 | 0.8592 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.7800 | 0.8256 | 0.5949 | 0.8642 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 1.4070 | 1.5700 | 1.4861 | 1.4813 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.9839 | 0.9929 | 0.8960 | 0.8713 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.7530 | 0.8356 | 0.7783 | 0.6575 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.8581 | 0.7292 | 1.0013 | 1.4383 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.5009 | 1.1119 | 1.0840 | 0.8206 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.8508 | 0.8369 | 0.5217 | 0.5035 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.9760 | 1.0674 | 1.0116 | 0.9910 |

### Correctness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 2.4409 | 3.2077 | 3.5644 | 3.0590 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.7356 | 0.5318 | 0.8557 | 0.5694 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.7058 | 1.0569 | 0.8039 | 0.8987 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.9296 | 1.1982 | 1.1033 | 0.9954 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.4192 | 1.0150 | 0.7661 | 1.1185 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.2426 | 0.9760 | 1.0830 | 0.8431 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.6315 | 1.2759 | 1.0398 | 1.2415 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.5571 | 0.8699 | 0.7648 | 0.7403 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.1646 | 0.9854 | 1.1878 | 1.0928 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.0244 | 1.4710 | 1.8438 | 1.6437 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.5061 | 0.7572 | 0.6661 | 0.9392 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 1.1561 | 0.6797 | 0.7769 | 0.8209 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.8324 | 0.7104 | 0.5813 | 0.8323 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 1.2059 | 1.3403 | 1.5661 | 1.4606 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.1584 | 1.0269 | 0.9829 | 1.0036 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.8449 | 0.6501 | 0.8873 | 0.6410 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.7443 | 0.8739 | 1.1364 | 1.4054 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.5432 | 1.2833 | 1.0861 | 0.9162 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.7069 | 0.7367 | 0.4288 | 0.4877 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.9077 | 1.0747 | 1.1768 | 1.0889 |

### Completeness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.7791 | 3.1693 | 1.9695 | 1.6359 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.7423 | 0.6344 | 1.0071 | 0.6600 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.8378 | 1.0239 | 0.6437 | 0.8725 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.9990 | 1.0134 | 1.4233 | 1.3553 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.3142 | 1.2831 | 0.9820 | 1.3651 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.2498 | 0.9523 | 0.8799 | 0.7920 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.2021 | 0.7258 | 0.9993 | 1.0599 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.4118 | 0.8062 | 0.7751 | 0.7840 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.8768 | 0.9558 | 1.3754 | 0.9099 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.1071 | 1.7989 | 1.2875 | 1.3155 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.2814 | 0.7370 | 0.7578 | 1.0057 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 1.1656 | 0.6748 | 0.9673 | 1.1217 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.9142 | 0.6146 | 0.5880 | 0.7187 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 1.3409 | 1.5800 | 1.9724 | 1.2485 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.2022 | 1.1177 | 1.1363 | 1.2213 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.9196 | 0.8790 | 0.8447 | 1.0007 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.9628 | 1.4447 | 1.4760 | 1.7409 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.6066 | 0.8368 | 0.7581 | 0.6701 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.0765 | 0.6273 | 0.5585 | 0.5893 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.9008 | 1.0837 | 0.9106 | 0.8563 |

### Newsworthiness

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 1.5780 | 2.8411 | 3.1313 | 2.4111 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 1.1116 | 0.7042 | 0.8871 | 0.6237 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.0393 | 1.6667 | 0.7795 | 1.0375 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.1009 | 1.2384 | 1.8306 | 1.5139 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.8025 | 0.9692 | 1.0162 | 1.1285 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.8158 | 0.7403 | 0.6302 | 0.9589 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.3360 | 0.9194 | 0.9644 | 1.3139 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.7995 | 0.5913 | 0.8095 | 0.7369 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.1191 | 1.1342 | 1.0317 | 1.0439 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.0285 | 1.3997 | 1.1908 | 1.2992 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.9470 | 0.9179 | 0.9837 | 0.8514 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 1.1760 | 0.6859 | 0.8383 | 0.8352 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.9714 | 0.5914 | 0.7409 | 0.6126 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.6893 | 1.1708 | 1.5586 | 0.9467 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.1607 | 0.9592 | 1.2539 | 1.0944 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 1.2365 | 0.9225 | 0.6192 | 0.8914 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.8258 | 1.2518 | 1.2676 | 1.0496 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 1.0940 | 0.9569 | 0.6272 | 0.7850 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.9414 | 0.7866 | 0.5986 | 0.7207 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.6800 | 1.0761 | 1.1673 | 1.2473 |

### Hygiene

| model | gpt-3.5-turbo_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 2.4151 | 5.0602 | 4.1384 | 4.4420 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 1.4565 | 1.4137 | 1.0423 | 0.6811 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.7820 | 1.0442 | 0.6116 | 0.5806 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.2391 | 1.4997 | 1.4157 | 1.8084 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.9008 | 1.0928 | 1.0753 | 1.2628 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.9988 | 1.0240 | 1.0145 | 1.0162 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.9026 | 0.8138 | 1.4582 | 1.3325 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.7521 | 0.7886 | 0.9094 | 0.6204 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.7810 | 1.2636 | 0.6308 | 0.6478 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.0708 | 1.3978 | 1.8070 | 1.7469 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.1677 | 0.9560 | 0.5141 | 0.8052 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 1.2705 | 0.8468 | 0.6727 | 0.6800 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.9084 | 0.7596 | 0.7193 | 0.9764 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 1.0514 | 0.9830 | 2.4640 | 1.9259 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.0861 | 1.0667 | 1.0441 | 1.0867 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.9096 | 0.3990 | 0.7045 | 0.7076 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.7234 | 0.9871 | 1.1044 | 1.1125 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.8523 | 0.6634 | 0.6823 | 0.8405 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.7022 | 0.4698 | 0.4543 | 0.4209 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.9478 | 0.8893 | 1.0522 | 0.8416 |

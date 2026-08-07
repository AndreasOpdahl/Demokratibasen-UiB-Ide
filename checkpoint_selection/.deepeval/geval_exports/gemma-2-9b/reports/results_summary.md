# G-Eval results summary

**Judges:** `gpt-5-mini`, `google/gemini-2.5-flash-preview-05-20`, `anthropic/claude-3-5-haiku-20241022`, `mistral-medium-latest`
**Dimensions:** `relevance`, `consistency`, `newsworthiness`, `hygiene`
**Documents in subset:** 65 distinct `doc_id`.
**Datapoints:** 8320 pairwise judgments total (520 rows per G-Eval table × 16 table(s), one per judge × dimension).
Equivalent to 520 pair comparisons × 4 dimensions × 4 judges.

Bradley–Terry: `GPT4o-mini` labels gold summaries (JSONL `reference`). Exported θ use mean-centered β (geom. mean θ = 1); odds vs any other model match the fitted BT model.

---

## 1. Pairwise win rates

### Relevance

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.844 | 0.781 | 0.688 | 0.750 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.227 | 0.318 | 0.409 | 0.318 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.659 | 0.670 | 0.727 | 0.716 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.478 | 0.478 | 0.609 | 0.522 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.646 | 0.500 | 0.500 | 0.583 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.300 | 0.300 | 0.500 | 0.400 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.455 | 0.545 | 0.500 | 0.455 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.458 | 0.292 | 0.500 | 0.354 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.750 | 0.900 | 0.800 | 0.750 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.453 | 0.531 | 0.453 | 0.484 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.661 | 0.679 | 0.661 | 0.607 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.463 | 0.407 | 0.593 | 0.556 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.559 | 0.574 | 0.559 | 0.588 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.385 | 0.308 | 0.346 | 0.308 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.543 | 0.614 | 0.629 | 0.586 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.342 | 0.355 | 0.408 | 0.395 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.470 | 0.530 | 0.470 | 0.485 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.429 | 0.457 | 0.400 | 0.529 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.521 | 0.479 | 0.458 | 0.646 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.420 | 0.340 |
| checkpoint-500-inputs-refs-preds-1000-examples | 0.382 | 0.529 | 0.529 | 0.412 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.479 | 0.458 | 0.438 | 0.438 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.486 | 0.557 | 0.500 | 0.400 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.523 | 0.500 | 0.614 | 0.659 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.656 | 0.594 | 0.531 | 0.547 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.382 | 0.294 | 0.265 | 0.353 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.621 | 0.552 | 0.638 | 0.569 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.391 | 0.478 | 0.435 | 0.370 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.625 | 0.594 | 0.688 | 0.641 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.379 | 0.455 | 0.303 | 0.409 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.364 | 0.364 | 0.364 | 0.455 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.460 | 0.480 | 0.540 | 0.460 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.477 | 0.523 | 0.523 | 0.614 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.329 | 0.343 | 0.229 | 0.271 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.550 | 0.600 | 0.600 | 0.550 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.400 | 0.383 | 0.333 | 0.317 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.519 | 0.462 | 0.519 | 0.481 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.556 | 0.472 | 0.556 | 0.500 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.611 | 0.639 | 0.583 | 0.583 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.519 | 0.426 | 0.259 | 0.463 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.457 | 0.565 |

### Consistency

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.656 | 0.781 | 0.656 | 0.719 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.318 | 0.318 | 0.227 | 0.545 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.614 | 0.591 | 0.580 | 0.659 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.587 | 0.522 | 0.478 | 0.435 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.646 | 0.521 | 0.604 | 0.583 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.300 | 0.433 | 0.267 | 0.500 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.523 | 0.545 | 0.568 | 0.545 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.479 | 0.396 | 0.500 | 0.417 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.800 | 0.850 | 0.850 | 0.800 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.344 | 0.375 | 0.312 | 0.391 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.679 | 0.661 | 0.786 | 0.750 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.481 | 0.426 | 0.444 | 0.481 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.529 | 0.544 | 0.515 | 0.618 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.346 | 0.308 | 0.346 | 0.423 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.600 | 0.614 | 0.657 | 0.629 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.408 | 0.329 | 0.408 | 0.342 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.606 | 0.606 | 0.576 | 0.515 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.443 | 0.371 | 0.443 | 0.443 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.625 | 0.542 | 0.625 | 0.521 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.460 | 0.440 | 0.420 | 0.520 |
| checkpoint-500-inputs-refs-preds-1000-examples | 0.412 | 0.441 | 0.412 | 0.529 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.458 | 0.479 | 0.458 | 0.500 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.500 | 0.486 | 0.400 | 0.443 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.432 | 0.318 | 0.386 | 0.386 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.562 | 0.594 | 0.484 | 0.453 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.324 | 0.382 | 0.294 | 0.235 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.552 | 0.534 | 0.483 | 0.466 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.435 | 0.565 | 0.326 | 0.478 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.750 | 0.703 | 0.688 | 0.609 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.394 | 0.545 | 0.439 | 0.561 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.545 | 0.455 | 0.409 | 0.545 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.440 | 0.500 | 0.380 | 0.300 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.477 | 0.568 | 0.682 | 0.705 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.314 | 0.329 | 0.400 | 0.443 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.500 | 0.450 | 0.750 | 0.400 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.333 | 0.433 | 0.567 | 0.383 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.462 | 0.462 | 0.462 | 0.462 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.542 | 0.542 | 0.597 | 0.514 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.417 | 0.556 | 0.611 | 0.472 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.463 | 0.463 | 0.426 | 0.389 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.609 | 0.478 | 0.522 | 0.457 |

### Newsworthiness

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.812 | 0.938 | 0.781 | 0.750 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.591 | 0.409 | 0.455 | 0.409 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.534 | 0.625 | 0.625 | 0.625 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.522 | 0.326 | 0.478 | 0.587 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.521 | 0.521 | 0.542 | 0.458 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.400 | 0.333 | 0.433 | 0.367 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.500 | 0.545 | 0.500 | 0.409 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.438 | 0.396 | 0.333 | 0.458 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.750 | 0.600 | 0.800 | 0.700 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.484 | 0.531 | 0.547 | 0.422 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.679 | 0.643 | 0.679 | 0.571 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.444 | 0.481 | 0.444 | 0.481 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.500 | 0.485 | 0.574 | 0.574 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.615 | 0.346 | 0.423 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.586 | 0.414 | 0.486 | 0.514 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.513 | 0.539 | 0.382 | 0.382 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.455 | 0.455 | 0.515 | 0.545 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.386 | 0.514 | 0.614 | 0.543 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.583 | 0.521 | 0.583 | 0.458 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.460 | 0.420 | 0.320 | 0.340 |
| checkpoint-500-inputs-refs-preds-1000-examples | 0.500 | 0.441 | 0.529 | 0.412 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.479 | 0.438 | 0.500 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.500 | 0.557 | 0.471 | 0.514 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.523 | 0.455 | 0.455 | 0.500 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.516 | 0.625 | 0.547 | 0.516 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.265 | 0.265 | 0.294 | 0.471 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.466 | 0.517 | 0.569 | 0.603 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.391 | 0.500 | 0.522 | 0.457 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.469 | 0.453 | 0.500 | 0.656 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.424 | 0.424 | 0.439 | 0.379 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.455 | 0.455 | 0.545 | 0.545 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.420 | 0.540 | 0.360 | 0.620 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.432 | 0.386 | 0.568 | 0.545 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.443 | 0.429 | 0.457 | 0.486 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.350 | 0.700 | 0.600 | 0.550 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.383 | 0.383 | 0.400 | 0.333 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.654 | 0.596 | 0.538 | 0.577 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.653 | 0.528 | 0.444 | 0.444 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.639 | 0.694 | 0.667 | 0.583 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.463 | 0.407 | 0.370 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.457 | 0.391 | 0.478 | 0.478 |

### Hygiene

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.781 | 0.656 | 0.625 | 0.719 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.409 | 0.409 | 0.364 | 0.273 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.716 | 0.648 | 0.648 | 0.614 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.543 | 0.565 | 0.565 | 0.435 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.479 | 0.521 | 0.562 | 0.458 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.400 | 0.400 | 0.200 | 0.267 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.477 | 0.523 | 0.545 | 0.364 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.438 | 0.562 | 0.542 | 0.625 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.650 | 0.700 | 0.700 | 0.700 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.562 | 0.547 | 0.438 | 0.469 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.607 | 0.679 | 0.625 | 0.661 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.389 | 0.463 | 0.481 | 0.407 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.500 | 0.559 | 0.485 | 0.676 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.500 | 0.385 | 0.423 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.529 | 0.486 | 0.529 | 0.486 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.421 | 0.434 | 0.329 | 0.421 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.561 | 0.500 | 0.576 | 0.485 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.557 | 0.400 | 0.457 | 0.457 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.604 | 0.562 | 0.562 | 0.458 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.360 | 0.440 | 0.560 | 0.560 |
| checkpoint-500-inputs-refs-preds-1000-examples | 0.471 | 0.324 | 0.588 | 0.412 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.396 | 0.417 | 0.417 | 0.396 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.514 | 0.486 | 0.514 | 0.571 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.455 | 0.523 | 0.409 | 0.432 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.531 | 0.516 | 0.406 | 0.438 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.412 | 0.382 | 0.412 | 0.412 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.603 | 0.586 | 0.569 | 0.552 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.478 | 0.435 | 0.391 | 0.370 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.656 | 0.703 | 0.641 | 0.547 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.455 | 0.500 | 0.364 | 0.545 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.591 | 0.591 | 0.682 | 0.636 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.320 | 0.420 | 0.580 | 0.520 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.523 | 0.500 | 0.705 | 0.659 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.414 | 0.429 | 0.371 | 0.343 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.550 | 0.600 | 0.500 | 0.550 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.350 | 0.467 | 0.383 | 0.450 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.538 | 0.462 | 0.577 | 0.577 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.417 | 0.403 | 0.444 | 0.528 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.472 | 0.417 | 0.528 | 0.639 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.315 | 0.352 | 0.389 | 0.389 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.543 | 0.457 | 0.587 | 0.522 |

---

## 2. Bradley–Terry strengths (θ)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Relevance

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 6.6025 | 4.5246 | 2.4649 | 3.3472 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.2243 | 0.3222 | 0.5815 | 0.3056 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 1.4070 | 1.3986 | 1.8188 | 1.5323 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.7738 | 0.6741 | 0.9401 | 0.8941 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 2.0667 | 1.1314 | 1.3323 | 1.7737 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.3330 | 0.2863 | 0.8047 | 0.5898 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.9645 | 1.3984 | 1.3153 | 1.1064 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.6377 | 0.3339 | 0.7060 | 0.5202 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 2.4292 | 8.5569 | 3.9855 | 2.4779 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.8739 | 1.1445 | 0.8649 | 0.8558 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 2.7836 | 3.0230 | 2.0315 | 1.8206 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.6265 | 0.6158 | 1.4993 | 1.2268 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.3320 | 1.4661 | 1.3884 | 1.4626 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.5382 | 0.4069 | 0.4801 | 0.3394 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.4663 | 1.7715 | 1.7174 | 1.2849 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.5516 | 0.6615 | 1.0052 | 0.8422 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.0834 | 1.3991 | 1.0311 | 1.0305 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.5539 | 0.5107 | 0.5056 | 1.0110 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.8142 | 0.7553 | 0.5733 | 1.7807 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 1.6239 | 1.6327 | 0.8202 | 0.6341 |
| checkpoint-500-inputs-refs-preds-1000-examples | 0.5791 | 0.9777 | 1.3332 | 0.8305 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 1.0763 | 0.9852 | 0.9289 | 0.8222 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.9275 | 1.3002 | 1.0176 | 0.8933 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 1.1632 | 1.1027 | 1.8069 | 2.1081 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 2.7086 | 2.1505 | 1.1472 | 1.1130 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.6807 | 0.3932 | 0.4459 | 0.5460 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 1.5642 | 1.3314 | 1.6927 | 1.3387 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.7413 | 0.9803 | 0.9454 | 0.6165 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 1.6857 | 1.4253 | 1.9023 | 1.6502 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.6461 | 0.7663 | 0.5125 | 0.7339 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.5436 | 0.5403 | 0.5284 | 0.9798 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.7207 | 0.7585 | 1.0228 | 0.8781 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.7522 | 1.0474 | 0.9365 | 1.8340 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.5342 | 0.5573 | 0.3297 | 0.3999 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.3804 | 1.4690 | 1.3144 | 1.2198 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.7698 | 0.7067 | 0.6069 | 0.4709 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.9527 | 0.6877 | 1.1058 | 0.8231 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 1.0796 | 0.7384 | 0.9742 | 0.7335 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.4978 | 1.5503 | 1.2816 | 1.3556 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 1.0945 | 0.6677 | 0.3568 | 0.9696 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 1.9473 | 1.9709 | 1.1610 | 1.7628 |

### Consistency

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 2.9013 | 5.3080 | 2.6553 | 3.5527 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.2935 | 0.3584 | 0.2279 | 0.9503 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 1.0750 | 1.2588 | 1.1640 | 1.3842 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 1.1642 | 1.0700 | 0.6259 | 0.5125 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.7889 | 0.8876 | 1.4093 | 1.4168 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.3364 | 0.5725 | 0.2366 | 0.8523 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.2644 | 1.3139 | 1.4696 | 1.7013 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.6993 | 0.4906 | 0.7241 | 0.5158 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 4.0835 | 4.8961 | 5.3789 | 3.5484 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.5910 | 0.6451 | 0.4764 | 0.6387 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 3.2448 | 2.4059 | 4.9214 | 3.4783 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.8501 | 0.6064 | 0.7499 | 0.9365 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.3509 | 1.2802 | 1.2055 | 1.9825 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.4411 | 0.3892 | 0.4485 | 0.9402 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.3758 | 1.6834 | 1.6740 | 1.5834 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.8785 | 0.5744 | 0.9089 | 0.6398 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.8156 | 1.8840 | 1.4796 | 1.1910 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.6730 | 0.4540 | 0.6005 | 0.4560 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.3728 | 0.8772 | 1.8449 | 1.0875 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 1.4327 | 1.2297 | 1.1546 | 1.2233 |
| checkpoint-500-inputs-refs-preds-1000-examples | 0.6569 | 0.5679 | 0.6357 | 1.0159 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.8598 | 0.9356 | 0.7867 | 1.0596 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.9531 | 0.7124 | 0.7563 | 0.8812 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.8024 | 0.3715 | 0.5451 | 0.6257 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 1.7689 | 1.9437 | 1.4224 | 1.0302 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.4561 | 0.6680 | 0.4344 | 0.2926 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 1.0957 | 1.4046 | 0.8834 | 1.0302 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.8349 | 1.7312 | 0.5725 | 0.9559 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 2.6004 | 2.2790 | 1.9906 | 1.0385 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.6373 | 1.2283 | 0.6532 | 1.1125 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.1399 | 0.7657 | 0.6141 | 1.5182 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.7518 | 0.8875 | 0.5522 | 0.3202 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.9660 | 1.0800 | 2.3617 | 2.3321 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.4609 | 0.5225 | 0.7388 | 0.8724 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.0682 | 0.6922 | 3.1699 | 0.6304 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.5394 | 0.9410 | 1.3903 | 0.6259 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.6620 | 0.7763 | 0.5946 | 0.9349 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.8902 | 1.1901 | 1.1929 | 0.9203 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.7162 | 1.3159 | 1.6250 | 0.8021 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 1.0311 | 0.9386 | 0.7630 | 0.4748 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 3.0298 | 1.6418 | 2.2599 | 1.5035 |

### Newsworthiness

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 5.2974 | 18.3919 | 4.3484 | 3.1861 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 1.3598 | 0.7742 | 0.6642 | 0.5926 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.9257 | 1.3105 | 1.1930 | 1.3706 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.8195 | 0.3523 | 0.7291 | 1.2697 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.3101 | 1.3041 | 1.3816 | 0.8902 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.7681 | 0.4838 | 0.8201 | 0.6298 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.1064 | 1.5188 | 1.2246 | 0.7734 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.5861 | 0.5808 | 0.5033 | 0.7421 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 3.0676 | 1.6972 | 3.0291 | 1.6734 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.9698 | 0.9923 | 1.0495 | 0.7449 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 2.4164 | 1.9562 | 2.3271 | 1.3977 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.7337 | 0.9215 | 0.7547 | 0.7667 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.0345 | 1.2103 | 1.3388 | 1.3552 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 1.0463 | 1.6151 | 0.4917 | 0.6942 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.5183 | 0.6708 | 0.9622 | 1.0640 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 1.1225 | 1.2424 | 0.7041 | 0.6641 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.1214 | 1.1548 | 1.1678 | 1.1881 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.4276 | 0.7987 | 1.1630 | 1.1866 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.9859 | 0.8493 | 1.4205 | 0.8110 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.8963 | 0.8706 | 0.5311 | 0.5285 |
| checkpoint-500-inputs-refs-preds-1000-examples | 0.9361 | 0.7358 | 1.0381 | 0.6621 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 1.4763 | 1.0127 | 0.9494 | 1.0995 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.0257 | 1.1827 | 1.0049 | 1.0588 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 1.0758 | 0.8427 | 0.8836 | 1.0271 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 1.0782 | 1.3738 | 0.9995 | 0.8271 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.3960 | 0.3579 | 0.3861 | 0.9654 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.8008 | 0.7325 | 1.1458 | 1.2707 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.7118 | 0.9964 | 1.0828 | 0.9725 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.7123 | 0.7605 | 0.7916 | 1.8879 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.6029 | 0.5450 | 0.7385 | 0.6731 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.8553 | 0.7815 | 1.3828 | 1.2244 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.4681 | 0.9709 | 0.5429 | 1.9587 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.7618 | 0.6645 | 1.4887 | 1.1053 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.9029 | 0.7522 | 0.8173 | 0.9792 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.5045 | 2.3072 | 1.4740 | 1.1320 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.5747 | 0.6661 | 0.6105 | 0.5337 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 2.1646 | 1.1178 | 1.0807 | 1.2155 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 1.6536 | 1.0729 | 0.7177 | 0.8673 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.5775 | 1.8010 | 1.6475 | 1.3930 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.8167 | 0.6475 | 0.6765 | 0.7694 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 1.1593 | 0.8998 | 1.2839 | 1.0024 |

### Hygiene

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 4.4718 | 2.1828 | 2.0786 | 2.8000 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.5203 | 0.5836 | 0.3974 | 0.3030 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 2.0824 | 1.4067 | 1.1845 | 1.1500 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.9085 | 0.9445 | 0.9136 | 0.5596 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.0288 | 1.1379 | 1.3826 | 0.8142 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.6495 | 0.4918 | 0.2299 | 0.2975 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.9405 | 1.3289 | 1.3487 | 0.7397 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.7801 | 1.2227 | 0.9903 | 1.6447 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.5467 | 2.2296 | 1.8140 | 2.3459 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 1.3503 | 1.3315 | 0.8398 | 0.9432 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.6598 | 2.1644 | 1.9966 | 2.3611 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.5640 | 0.8047 | 0.8649 | 0.6433 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.0139 | 1.5488 | 1.0685 | 2.2886 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.8334 | 1.0501 | 0.5146 | 0.8573 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.2918 | 1.0033 | 0.9323 | 0.8687 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.8196 | 0.8890 | 0.6003 | 0.8269 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.6089 | 1.1517 | 1.3459 | 1.0662 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.8787 | 0.5398 | 0.8391 | 0.7440 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.4563 | 1.0703 | 1.7502 | 1.0099 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.6964 | 1.0089 | 1.4182 | 1.3842 |
| checkpoint-500-inputs-refs-preds-1000-examples | 0.7442 | 0.4650 | 1.4432 | 0.6033 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.7703 | 0.8083 | 0.7816 | 0.7198 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.1369 | 0.9998 | 1.3602 | 1.5286 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.8626 | 1.0793 | 0.9203 | 0.8138 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 1.0846 | 1.1550 | 0.7396 | 0.9064 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.8010 | 0.5320 | 0.6337 | 0.6398 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 1.5643 | 1.5611 | 1.0017 | 1.3644 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 1.0530 | 0.7702 | 0.6368 | 0.5869 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 1.2837 | 1.9739 | 1.7606 | 1.1773 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.9457 | 1.2100 | 0.5828 | 1.4065 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.4362 | 1.4309 | 2.2710 | 1.9801 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.4443 | 0.6825 | 1.3477 | 1.0013 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 1.2454 | 1.0010 | 3.2182 | 2.2527 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.8284 | 0.7106 | 0.5877 | 0.5364 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.0663 | 1.3987 | 1.0217 | 1.1954 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.6025 | 0.9522 | 0.5872 | 0.7318 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 1.0365 | 0.6941 | 0.9027 | 0.9053 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.7942 | 0.5727 | 0.6382 | 0.8889 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.7615 | 0.6819 | 1.1135 | 1.6197 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.4607 | 0.5595 | 0.7128 | 0.6066 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 1.5491 | 1.2853 | 1.6878 | 1.7194 |

---

## Export layout

- `json/` — pairwise rows and per-judge G-Eval tables (JSON)
- `tables/` — CSV summaries (win rates, Bradley–Terry)
- `reports/` — Markdown / LaTeX for reading and papers

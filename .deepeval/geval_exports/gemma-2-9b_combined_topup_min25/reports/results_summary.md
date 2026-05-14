# G-Eval results summary

**Judges:** `gpt-5-mini`, `google/gemini-2.5-flash-preview-05-20`, `anthropic/claude-3-5-haiku-20241022`, `mistral-medium-latest`
**Dimensions:** `relevance`, `consistency`, `newsworthiness`, `hygiene`
**Documents in subset:** 65 distinct `doc_id`.
**Datapoints:** 9712 pairwise judgments total (607 rows per G-Eval table × 16 table(s), one per judge × dimension).
Equivalent to 607 pair comparisons × 4 dimensions × 4 judges.

Bradley–Terry: `GPT4o-mini` labels gold summaries (JSONL `reference`). Exported θ use mean-centered β (geom. mean θ = 1); odds vs any other model match the fitted BT model.

---

## 1. Pairwise win rates

### Relevance

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.844 | 0.781 | 0.688 | 0.750 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.269 | 0.250 | 0.288 | 0.269 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.659 | 0.670 | 0.727 | 0.716 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.481 | 0.481 | 0.577 | 0.500 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.640 | 0.520 | 0.520 | 0.600 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.269 | 0.250 | 0.385 | 0.346 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.404 | 0.558 | 0.538 | 0.462 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.463 | 0.315 | 0.556 | 0.407 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.635 | 0.692 | 0.712 | 0.654 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.456 | 0.529 | 0.456 | 0.485 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.661 | 0.679 | 0.661 | 0.607 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.466 | 0.414 | 0.586 | 0.552 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.559 | 0.574 | 0.559 | 0.588 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.404 | 0.365 | 0.385 | 0.385 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.556 | 0.625 | 0.639 | 0.597 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.346 | 0.372 | 0.423 | 0.397 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.486 | 0.527 | 0.432 | 0.459 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.444 | 0.444 | 0.389 | 0.514 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.519 | 0.500 | 0.463 | 0.611 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.537 | 0.426 | 0.352 |
| checkpoint-500-inputs-refs-preds-1000-examples | 0.440 | 0.520 | 0.560 | 0.480 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.429 | 0.393 | 0.393 | 0.411 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.486 | 0.556 | 0.500 | 0.403 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.519 | 0.519 | 0.615 | 0.673 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.632 | 0.574 | 0.515 | 0.544 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.463 | 0.370 | 0.407 | 0.463 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.629 | 0.565 | 0.629 | 0.581 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.352 | 0.426 | 0.370 | 0.333 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.606 | 0.576 | 0.667 | 0.621 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.379 | 0.455 | 0.303 | 0.409 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.580 | 0.600 | 0.520 | 0.480 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.483 | 0.534 | 0.466 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.556 | 0.593 | 0.556 | 0.630 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.329 | 0.343 | 0.229 | 0.271 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.560 | 0.620 | 0.560 | 0.620 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.387 | 0.371 | 0.323 | 0.306 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.518 | 0.464 | 0.518 | 0.482 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.579 | 0.500 | 0.566 | 0.513 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.540 | 0.560 | 0.600 | 0.580 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.517 | 0.431 | 0.276 | 0.466 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.537 | 0.537 | 0.519 | 0.593 |

### Consistency

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.656 | 0.781 | 0.656 | 0.719 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.308 | 0.288 | 0.269 | 0.500 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.614 | 0.591 | 0.580 | 0.659 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.558 | 0.500 | 0.500 | 0.462 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.620 | 0.500 | 0.580 | 0.560 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.269 | 0.365 | 0.231 | 0.404 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.538 | 0.577 | 0.596 | 0.538 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.463 | 0.407 | 0.444 | 0.426 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.596 | 0.654 | 0.654 | 0.654 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.338 | 0.353 | 0.324 | 0.368 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.679 | 0.661 | 0.786 | 0.750 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.483 | 0.397 | 0.448 | 0.483 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.529 | 0.544 | 0.515 | 0.618 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.385 | 0.365 | 0.308 | 0.365 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.611 | 0.625 | 0.667 | 0.639 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.423 | 0.346 | 0.423 | 0.359 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.568 | 0.568 | 0.581 | 0.486 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.444 | 0.375 | 0.431 | 0.431 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.630 | 0.593 | 0.630 | 0.519 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.481 | 0.463 | 0.519 |
| checkpoint-500-inputs-refs-preds-1000-examples | 0.460 | 0.480 | 0.520 | 0.600 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.429 | 0.446 | 0.411 | 0.482 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.500 | 0.486 | 0.403 | 0.444 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.481 | 0.365 | 0.423 | 0.404 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.544 | 0.603 | 0.471 | 0.441 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.463 | 0.426 | 0.463 | 0.278 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.565 | 0.548 | 0.516 | 0.484 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.389 | 0.519 | 0.296 | 0.426 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.742 | 0.682 | 0.682 | 0.591 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.394 | 0.545 | 0.439 | 0.561 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.620 | 0.620 | 0.540 | 0.600 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.431 | 0.500 | 0.431 | 0.345 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.556 | 0.630 | 0.685 | 0.685 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.314 | 0.329 | 0.400 | 0.443 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.540 | 0.540 | 0.600 | 0.620 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.323 | 0.419 | 0.548 | 0.387 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.464 | 0.464 | 0.464 | 0.464 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.566 | 0.566 | 0.618 | 0.526 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.440 | 0.480 | 0.580 | 0.500 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.466 | 0.466 | 0.414 | 0.397 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.630 | 0.500 | 0.537 | 0.463 |

### Newsworthiness

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.812 | 0.938 | 0.781 | 0.750 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.442 | 0.404 | 0.404 | 0.346 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.534 | 0.625 | 0.625 | 0.625 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.519 | 0.346 | 0.462 | 0.538 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.540 | 0.540 | 0.560 | 0.480 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.365 | 0.327 | 0.385 | 0.365 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.442 | 0.500 | 0.481 | 0.365 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.481 | 0.407 | 0.389 | 0.500 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.635 | 0.577 | 0.731 | 0.615 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.485 | 0.529 | 0.544 | 0.426 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.679 | 0.643 | 0.679 | 0.571 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.448 | 0.483 | 0.448 | 0.483 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.500 | 0.485 | 0.574 | 0.574 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.481 | 0.481 | 0.404 | 0.423 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.597 | 0.431 | 0.500 | 0.528 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.513 | 0.526 | 0.385 | 0.385 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.486 | 0.486 | 0.527 | 0.554 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.403 | 0.500 | 0.597 | 0.556 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.556 | 0.537 | 0.574 | 0.444 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.426 | 0.426 | 0.296 | 0.315 |
| checkpoint-500-inputs-refs-preds-1000-examples | 0.540 | 0.420 | 0.560 | 0.480 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.482 | 0.482 | 0.411 | 0.482 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.500 | 0.556 | 0.472 | 0.514 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.519 | 0.481 | 0.462 | 0.500 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.515 | 0.618 | 0.544 | 0.515 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.370 | 0.407 | 0.370 | 0.537 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.484 | 0.532 | 0.581 | 0.613 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.352 | 0.444 | 0.463 | 0.407 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.455 | 0.470 | 0.500 | 0.667 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.424 | 0.424 | 0.439 | 0.379 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.560 | 0.600 | 0.600 | 0.620 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.431 | 0.466 | 0.362 | 0.638 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.519 | 0.481 | 0.593 | 0.593 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.443 | 0.429 | 0.457 | 0.486 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.520 | 0.680 | 0.600 | 0.540 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.371 | 0.371 | 0.387 | 0.339 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.643 | 0.589 | 0.536 | 0.571 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.618 | 0.513 | 0.447 | 0.434 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.600 | 0.640 | 0.600 | 0.540 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.466 | 0.414 | 0.379 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.444 | 0.370 | 0.463 | 0.463 |

### Hygiene

| model | gpt-5-mini_win_rate | google/gemini-2.5-flash-preview-05-20_win_rate | anthropic/claude-3-5-haiku-20241022_win_rate | mistral-medium-latest_win_rate |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 0.781 | 0.656 | 0.625 | 0.719 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.423 | 0.442 | 0.346 | 0.250 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.716 | 0.648 | 0.648 | 0.614 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.519 | 0.538 | 0.577 | 0.385 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 0.480 | 0.520 | 0.560 | 0.480 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.423 | 0.212 | 0.231 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.442 | 0.500 | 0.538 | 0.404 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.463 | 0.556 | 0.556 | 0.667 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 0.635 | 0.577 | 0.615 | 0.673 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.559 | 0.544 | 0.441 | 0.471 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 0.607 | 0.679 | 0.625 | 0.661 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.362 | 0.431 | 0.483 | 0.414 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.500 | 0.559 | 0.485 | 0.676 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.500 | 0.481 | 0.404 | 0.462 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 0.542 | 0.500 | 0.542 | 0.500 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.423 | 0.436 | 0.346 | 0.423 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 0.541 | 0.486 | 0.554 | 0.500 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.556 | 0.403 | 0.444 | 0.444 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.537 | 0.556 | 0.556 | 0.463 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.352 | 0.426 | 0.537 | 0.519 |
| checkpoint-500-inputs-refs-preds-1000-examples | 0.480 | 0.380 | 0.600 | 0.440 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.429 | 0.429 | 0.429 | 0.393 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 0.514 | 0.486 | 0.514 | 0.556 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.462 | 0.538 | 0.442 | 0.442 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.529 | 0.515 | 0.412 | 0.412 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.333 | 0.352 | 0.444 | 0.444 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.597 | 0.597 | 0.597 | 0.548 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.463 | 0.426 | 0.352 | 0.333 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.636 | 0.697 | 0.636 | 0.530 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.455 | 0.500 | 0.364 | 0.545 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 0.620 | 0.600 | 0.640 | 0.640 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.362 | 0.414 | 0.552 | 0.517 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 0.537 | 0.519 | 0.722 | 0.704 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.414 | 0.429 | 0.371 | 0.343 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.500 | 0.640 | 0.560 | 0.680 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.371 | 0.484 | 0.371 | 0.435 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.536 | 0.464 | 0.554 | 0.554 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.434 | 0.421 | 0.461 | 0.553 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.540 | 0.480 | 0.540 | 0.600 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.328 | 0.362 | 0.397 | 0.414 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 0.537 | 0.444 | 0.574 | 0.519 |

---

## 2. Bradley–Terry strengths (θ)

Gold summaries use the label `GPT4o-mini` (text from JSONL `reference`). β is shifted so mean(β)=0 over models (geometric mean of θ is 1); pairwise win odds are unchanged. Sparse data can push some θ toward 0 (separation).

### Relevance

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 6.7290 | 4.9209 | 2.4951 | 3.3176 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.2947 | 0.2513 | 0.3439 | 0.2837 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 1.4152 | 1.4032 | 1.8762 | 1.6107 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.7623 | 0.6815 | 0.8647 | 0.8105 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 2.0420 | 1.3778 | 1.5140 | 2.0169 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.3041 | 0.2321 | 0.4893 | 0.4400 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.7553 | 1.5457 | 1.5630 | 1.1562 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.6502 | 0.3670 | 0.8928 | 0.5930 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.3740 | 2.0476 | 2.5643 | 1.7739 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.8053 | 1.0582 | 0.8119 | 0.8151 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 2.5911 | 2.8874 | 2.0303 | 1.8076 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.6343 | 0.6507 | 1.5442 | 1.2343 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.3172 | 1.5823 | 1.4911 | 1.5303 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.6620 | 0.5785 | 0.5813 | 0.5511 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.4444 | 1.7891 | 1.6632 | 1.3286 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.5186 | 0.6266 | 1.0085 | 0.7956 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.1160 | 1.3937 | 0.8642 | 0.9298 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.6213 | 0.5136 | 0.4883 | 0.9078 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 0.9366 | 0.9587 | 0.6748 | 1.5121 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 1.3116 | 1.4916 | 0.7280 | 0.5726 |
| checkpoint-500-inputs-refs-preds-1000-examples | 0.7560 | 1.0506 | 1.6366 | 1.1066 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.8283 | 0.7133 | 0.7195 | 0.7358 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.0400 | 1.5534 | 1.1765 | 0.9555 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 1.3015 | 1.4895 | 2.2378 | 2.5560 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 2.1491 | 1.8069 | 1.0037 | 1.0418 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.8066 | 0.4834 | 0.7122 | 0.8258 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 1.5786 | 1.4071 | 1.6633 | 1.4172 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.6250 | 0.8115 | 0.7490 | 0.5428 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 1.4499 | 1.2189 | 1.6589 | 1.4354 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.6609 | 0.8125 | 0.5366 | 0.7520 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.2104 | 1.3702 | 0.9717 | 0.8601 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.8458 | 0.7376 | 0.9228 | 0.8218 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 1.0556 | 1.4721 | 1.2098 | 1.7710 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.5766 | 0.6179 | 0.3598 | 0.4242 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.1521 | 1.4280 | 1.0441 | 1.4167 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.7049 | 0.6144 | 0.5232 | 0.4400 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.9019 | 0.6621 | 0.9419 | 0.7963 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 1.1868 | 0.8275 | 1.0610 | 0.8259 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.0258 | 1.1041 | 1.3511 | 1.2675 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 1.0406 | 0.6590 | 0.3718 | 0.9043 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 1.8861 | 1.9792 | 1.3426 | 1.7958 |

### Consistency

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 2.7093 | 4.9844 | 2.8542 | 3.2711 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.3290 | 0.3309 | 0.2765 | 0.8591 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 1.1348 | 1.2074 | 1.3045 | 1.4010 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 1.0421 | 0.9110 | 0.7065 | 0.6357 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.7189 | 0.9367 | 1.3952 | 1.4501 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.2968 | 0.4411 | 0.2253 | 0.5645 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 1.3108 | 1.4667 | 1.6840 | 1.5391 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.6330 | 0.4974 | 0.5606 | 0.5546 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.3474 | 1.5487 | 1.6221 | 1.6527 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.5177 | 0.5280 | 0.4462 | 0.5574 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 3.1058 | 2.3658 | 4.3675 | 3.3649 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.8699 | 0.5776 | 0.8336 | 0.9960 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.3563 | 1.3764 | 1.1728 | 1.8093 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.6113 | 0.5754 | 0.4107 | 0.6630 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.3966 | 1.7045 | 1.7692 | 1.5871 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.7864 | 0.5213 | 0.8507 | 0.6242 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.4641 | 1.4643 | 1.5592 | 0.9932 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.6629 | 0.4808 | 0.5806 | 0.4940 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.5748 | 1.3161 | 1.8691 | 1.0684 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 1.3316 | 1.2504 | 1.0472 | 1.0451 |
| checkpoint-500-inputs-refs-preds-1000-examples | 0.8441 | 0.7651 | 1.1649 | 1.5852 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.7580 | 0.7705 | 0.6587 | 0.9411 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.1299 | 0.8861 | 0.8057 | 0.9096 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 1.1078 | 0.5719 | 0.7795 | 0.7756 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 1.5005 | 1.9209 | 1.1867 | 0.8727 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.7497 | 0.7133 | 0.8529 | 0.3830 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 1.1735 | 1.4465 | 1.0638 | 1.1344 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.6984 | 1.3175 | 0.5318 | 0.8209 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 2.4133 | 1.8670 | 1.9231 | 1.0351 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.6457 | 1.2230 | 0.7108 | 1.2014 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.3883 | 1.4462 | 0.9242 | 1.5889 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.6933 | 0.8607 | 0.6855 | 0.4339 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 1.3337 | 1.5877 | 2.3043 | 2.0564 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.5114 | 0.5768 | 0.8381 | 0.9047 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.0840 | 1.0271 | 1.2415 | 1.5090 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.5038 | 0.8457 | 1.3148 | 0.5996 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 0.6685 | 0.7280 | 0.6170 | 0.8646 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 1.0412 | 1.2240 | 1.4838 | 0.9705 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 0.7916 | 0.9102 | 1.4353 | 0.9882 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.9918 | 0.8644 | 0.7098 | 0.5468 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 2.8468 | 1.6229 | 1.9482 | 1.3267 |

### Newsworthiness

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 5.3160 | 19.1179 | 4.5277 | 3.1959 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.7102 | 0.6273 | 0.5854 | 0.4770 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 0.8979 | 1.3026 | 1.2101 | 1.3224 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.8637 | 0.4208 | 0.6999 | 1.0297 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.5805 | 1.5076 | 1.6041 | 1.0352 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.5971 | 0.4923 | 0.6150 | 0.6103 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.8536 | 1.2094 | 1.2039 | 0.6606 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.7326 | 0.5932 | 0.6064 | 0.8680 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.6116 | 1.2828 | 2.4043 | 1.3231 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 0.9835 | 1.0354 | 1.0593 | 0.7747 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 2.2959 | 1.8143 | 2.1254 | 1.3356 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.7403 | 0.8657 | 0.8050 | 0.8105 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 0.9639 | 1.0132 | 1.3746 | 1.3226 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.8804 | 0.8774 | 0.6553 | 0.6835 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.5789 | 0.7484 | 1.0105 | 1.1370 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 1.0565 | 1.1352 | 0.7113 | 0.6644 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.1893 | 1.3140 | 1.2399 | 1.2297 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.4990 | 0.7359 | 1.0831 | 1.2717 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.0687 | 1.0487 | 1.3967 | 0.8386 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.7209 | 0.8086 | 0.4294 | 0.4401 |
| checkpoint-500-inputs-refs-preds-1000-examples | 1.2554 | 0.7340 | 1.3671 | 0.9174 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 1.2308 | 1.0075 | 0.7987 | 1.0129 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.1806 | 1.3557 | 1.0670 | 1.1543 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 1.3045 | 1.0691 | 1.0895 | 1.1894 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 0.9585 | 1.2462 | 0.8919 | 0.7732 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.5866 | 0.6340 | 0.5536 | 1.0995 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 0.9172 | 0.8418 | 1.2439 | 1.3880 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.5785 | 0.7959 | 0.9036 | 0.7488 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 0.6629 | 0.7421 | 0.7522 | 1.9349 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.6618 | 0.5848 | 0.7753 | 0.7005 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.2936 | 1.4704 | 1.6020 | 1.5552 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.5192 | 0.6986 | 0.5272 | 2.0905 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 1.0745 | 0.9962 | 1.5331 | 1.3880 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.9091 | 0.7747 | 0.8542 | 0.9890 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 1.0290 | 2.0597 | 1.4278 | 1.1238 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.4915 | 0.5680 | 0.5532 | 0.5258 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 1.9244 | 1.2146 | 1.0410 | 1.2313 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 1.4092 | 1.0127 | 0.7574 | 0.7901 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.3630 | 1.5141 | 1.3432 | 1.1295 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.8510 | 0.6935 | 0.6535 | 0.7538 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 1.0209 | 0.7498 | 1.0955 | 0.9110 |

### Hygiene

| model | gpt-5-mini_theta | google/gemini-2.5-flash-preview-05-20_theta | anthropic/claude-3-5-haiku-20241022_theta | mistral-medium-latest_theta |
| --- | --- | --- | --- | --- |
| GPT4o-mini | 4.1977 | 2.0733 | 1.9610 | 3.0320 |
| checkpoint-1000-gen1-inputs-refs-preds-1000-examples | 0.6589 | 0.7414 | 0.4566 | 0.3141 |
| checkpoint-1000-inputs-refs-preds-1000-examples | 1.9203 | 1.3774 | 1.1641 | 1.1015 |
| checkpoint-10000-gen1-inputs-refs-preds-1000-examples | 0.8620 | 0.9076 | 1.0193 | 0.4757 |
| checkpoint-10000-inputs-refs-preds-1000-examples | 1.0460 | 1.1803 | 1.4553 | 0.9831 |
| checkpoint-1500-gen1-inputs-refs-preds-1000-examples | 0.9441 | 0.5880 | 0.2359 | 0.2532 |
| checkpoint-1500-inputs-refs-preds-1000-examples | 0.8080 | 1.1394 | 1.3310 | 0.9470 |
| checkpoint-2000-gen1-inputs-refs-preds-1000-examples | 0.8494 | 1.1956 | 1.0575 | 1.9598 |
| checkpoint-2000-inputs-refs-preds-1000-examples | 1.4438 | 1.1933 | 1.3944 | 1.7619 |
| checkpoint-2500-gen1-inputs-refs-preds-1000-examples | 1.3593 | 1.3183 | 0.8610 | 0.9702 |
| checkpoint-2500-inputs-refs-preds-1000-examples | 1.6657 | 2.1051 | 1.9516 | 2.2639 |
| checkpoint-3000-gen1-inputs-refs-preds-1000-examples | 0.4968 | 0.6772 | 0.9006 | 0.7020 |
| checkpoint-3000-inputs-refs-preds-1000-examples | 1.0032 | 1.4285 | 1.0441 | 2.3706 |
| checkpoint-3500-gen1-inputs-refs-preds-1000-examples | 0.9506 | 1.0170 | 0.6089 | 0.9073 |
| checkpoint-3500-inputs-refs-preds-1000-examples | 1.3716 | 1.1063 | 1.0222 | 0.8841 |
| checkpoint-4000-gen1-inputs-refs-preds-1000-examples | 0.7750 | 0.7719 | 0.6027 | 0.8023 |
| checkpoint-4000-inputs-refs-preds-1000-examples | 1.4296 | 1.0582 | 1.2371 | 1.1888 |
| checkpoint-4500-gen1-inputs-refs-preds-1000-examples | 0.9041 | 0.5561 | 0.7828 | 0.7153 |
| checkpoint-4500-inputs-refs-preds-1000-examples | 1.0673 | 1.0546 | 1.6082 | 1.0163 |
| checkpoint-500-gen1-inputs-refs-preds-1000-examples | 0.6081 | 0.8693 | 1.1679 | 1.0546 |
| checkpoint-500-inputs-refs-preds-1000-examples | 0.8172 | 0.5775 | 1.5566 | 0.7237 |
| checkpoint-5000-gen1-inputs-refs-preds-1000-examples | 0.8997 | 0.8658 | 0.8702 | 0.6997 |
| checkpoint-5000-inputs-refs-preds-1000-examples | 1.0826 | 1.0392 | 1.4159 | 1.5435 |
| checkpoint-5500-gen1-inputs-refs-preds-1000-examples | 0.9337 | 1.2428 | 1.0951 | 0.9627 |
| checkpoint-5500-inputs-refs-preds-1000-examples | 1.0526 | 1.1020 | 0.6842 | 0.7120 |
| checkpoint-6000-gen1-inputs-refs-preds-1000-examples | 0.5024 | 0.4711 | 0.6561 | 0.6701 |
| checkpoint-6000-inputs-refs-preds-1000-examples | 1.5561 | 1.7022 | 1.2493 | 1.3859 |
| checkpoint-6500-gen1-inputs-refs-preds-1000-examples | 0.9554 | 0.7356 | 0.5543 | 0.4981 |
| checkpoint-6500-inputs-refs-preds-1000-examples | 1.2897 | 1.9975 | 1.6540 | 1.0522 |
| checkpoint-7000-gen1-inputs-refs-preds-1000-examples | 0.9568 | 1.2356 | 0.6456 | 1.4124 |
| checkpoint-7000-inputs-refs-preds-1000-examples | 1.6575 | 1.4776 | 1.5854 | 1.7335 |
| checkpoint-7500-gen1-inputs-refs-preds-1000-examples | 0.5429 | 0.6860 | 1.1759 | 0.9615 |
| checkpoint-7500-inputs-refs-preds-1000-examples | 1.0657 | 0.9850 | 3.1962 | 2.5152 |
| checkpoint-8000-gen1-inputs-refs-preds-1000-examples | 0.8149 | 0.7410 | 0.6005 | 0.5383 |
| checkpoint-8000-inputs-refs-preds-1000-examples | 0.9123 | 1.6216 | 1.1539 | 1.9901 |
| checkpoint-8500-gen1-inputs-refs-preds-1000-examples | 0.6430 | 0.9997 | 0.5350 | 0.6327 |
| checkpoint-8500-inputs-refs-preds-1000-examples | 1.1830 | 0.7965 | 0.9202 | 0.8814 |
| checkpoint-9000-gen1-inputs-refs-preds-1000-examples | 0.7660 | 0.5979 | 0.6618 | 0.9404 |
| checkpoint-9000-inputs-refs-preds-1000-examples | 1.0711 | 0.9188 | 1.2360 | 1.3841 |
| checkpoint-9500-gen1-inputs-refs-preds-1000-examples | 0.4954 | 0.6027 | 0.7335 | 0.6338 |
| checkpoint-9500-inputs-refs-preds-1000-examples | 1.4677 | 1.1187 | 1.5237 | 1.5093 |

---

## Export layout

- `json/` — pairwise rows and per-judge G-Eval tables (JSON)
- `tables/` — CSV summaries (win rates, Bradley–Terry)
- `reports/` — Markdown / LaTeX for reading and papers

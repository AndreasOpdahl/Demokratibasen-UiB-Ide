# Tokenizers and c200k Conversion Factors

This project now supports giving **length goals in c200k/o200k token space** and converting them to each model tokenizer during evaluation.

- Conversion formula: `model_tokens ~= round(c200k_tokens * factor)`
- Source table used at runtime: `model_fine_tuning_olivia/scripts/utils/c200k_length_conversion_table.json`
- These factors are approximations (sampled from training-summary texts).

## Model Factors (c200k -> model tokenizer)

| Model | Tokenizer (observed) | Factor |
|---|---|---:|
| `gemma-2b` | `GemmaTokenizer` | `0.9961` |
| `gemma-7b-it` | `GemmaTokenizer` | `0.9961` |
| `gemma-2-9b` | `GemmaTokenizer` | `0.9961` |
| `eurollm-9b-instruct` | `LlamaTokenizer` | `0.9816` |
| `norwai-mistral-7b-instruct` | `LlamaTokenizer` | `0.8138` |
| `normistral-7b` | fast backend (`TokenizersBackend`) | `0.7329` |
| `normistral-7b-instruct` | fast backend (`TokenizersBackend`) | `0.7329` |
| `normistral-11b` | fast backend (`TokenizersBackend`) | `0.7472` |
| `normistral-11b-long` | fast backend (`TokenizersBackend`) | `0.7559` |
| `viking-13b` | fast backend (`TokenizersBackend`) | `0.7513` |
| `viking-7b` | fast backend (`TokenizersBackend`) | `0.7513` |
| `llama-3.1-8b-instruct` | fast backend (`TokenizersBackend`) | `1.1559` |
| `norskgpt-llama3-8b` | fast backend (`TokenizersBackend`) | `1.1559` |
| `llama-2-13b-chat-norwegian` | `LlamaTokenizer` | `1.2727` |
| `nb-gpt-j-6b` | `GPT2Tokenizer` | `1.3299` |

## Quick Examples

- c200k target `86` tokens:
  - `gemma-7b-it`: `~86`
  - `normistral-7b`: `~63`
  - `llama-3.1-8b-instruct`: `~99`
  - `nb-gpt-j-6b`: `~114`

- c200k range `40-180`:
  - `gemma-7b-it`: `40-179`
  - `normistral-7b`: `29-132`
  - `llama-3.1-8b-instruct`: `46-208`
  - `nb-gpt-j-6b`: `53-239`

## CLI Usage

Use these flags in evaluation:

- `--c200k_min_new_tokens=N`
- `--c200k_max_new_tokens=N`

They are converted automatically per model to:

- `min_new_tokens`
- `max_output_summary_tokens`

## Notes

- `GPT2Tokenizer` is **not** c200k/o200k.
- Recompute factors when dataset distribution changes with:
  - `model_fine_tuning_olivia/scripts/utils/analyze_c200k_length_conversion.py`

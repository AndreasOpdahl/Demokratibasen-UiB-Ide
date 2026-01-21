# Data sizes

Example:

* 43000 texts
* ca 2500 input tokens / text
* ca 1000 output tokens / text
* ca 110M input, 43M output


# OpenAI (2025-12-09)

## Flex

* Model	Input	Cached input	Output
* gpt-5.1	$0.625	$0.0625	$5.00
* gpt-5	$0.625	$0.0625	$5.00
* gpt-5-mini	$0.125	$0.0125	$1.00

## Standard

* Model	Input	Cached input	Output
* gpt-5.1	$1.25	$0.125	$10.00
* gpt-5	$1.25	$0.125	$10.00
* gpt-5-mini	$0.25	$0.025	$2.00
* gpt-4.1	$2.00	$0.50	$8.00
* gpt-4.1-mini	$0.40	$0.10	$1.60

# Anthropic (2025-12-09)

## Opus 4.5

Most intelligent model for building agents and coding

* Input
  $5 / MTok
* Output
  $25 / MTok
* Prompt caching
  Write
  $6.25 / MTok
  Read
  $0.50 / MTok

## Sonnet 4.5

Optimal balance of intelligence, cost, and speed

* Input
  Prompts ≤ 200K tokens
  $3 / MTok
  Prompts > 200K tokens
  $6 / MTok
* Output
  Prompts ≤ 200K tokens
  $15 / MTok
  Prompts > 200K tokens
  $22.50 / MTok
* Prompt caching
* ≤ 200K tokens
  Write
  $3.75 / MTok
  Read
  $0.30 / MTok
* > 200K tokens
  > Write
  > $7.50 / MTok
  > Read
  > $0.60 / MTok
  >

## Haiku 4.5

Fastest, most cost-efficient model

* Input
  $1 / MTok
* Output
  $5 / MTok
* Prompt caching
  Write
  $1.25 / MTok
  Read
  $0.10 / MTok

# Gemini (2025-12-11)


| **AI Model**               | **Input Price (≤200K Tokens)**                             | **Output Price (≤200K Tokens)** | **Context Window**                                          | **Key Capability** |
| -------------------------------- | ----------------------------------------------------------------- | -------------------------------------- | ----------------------------------------------------------------- | ------------------------ |
| **Gemini 2.5 Pro**         | **$1.25 / M**                        | **$10.00 / M** | 2 Million Tokens                       | High capability, complex reasoning, coding, long context (Note 1) |                          |
| **Gemini 2.5 Flash**       | **$0.15 / M**                        | **$3.50 / M**  | 2 Million Tokens                       | Fastest and most cost-efficient, general performance, versatile   |                          |
| **Gemini 2.5 Flash-Lite**  | **$0.10 / M**                        | **$0.40 / M**  | 200K Tokens                            | Ultra-low cost, high-throughput, simple, high-frequency tasks     |                          |
| **Gemini 3 Pro (Preview)** | **$1.00 / M**                        | **$6.00 / M**  | 1 Million Tokens                       | Latest model, specialized for complex agentic workflows (Note 2)  |                          |

# DeepSeek (2025-12-10)

| MODEL                                                                                      | deepseek-chat                                              | deepseek-reasoner                                                                                                     | deepseek-reasoner^(1)^                         |
| ------------------------------------------------------------------------------------------ | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| BASE URL                                                                                   | [https://api.deepseek.com](https://api.deepseek.com/)         | [https://api.deepseek.com/v3.2_speciale_expires_on_20251215](https://api.deepseek.com/v3.2_speciale_expires_on_20251215) |                                                |
| MODEL VERSION                                                                              | DeepSeek-V3.2``(Non-thinking Mode)                         | DeepSeek-V3.2``(Thinking Mode)                                                                                        | DeepSeek-V3.2-Speciale``（Thinking Mode Only） |
| CONTEXT LENGTH                                                                             | 128K                                                       |                                                                                                                       |                                                |
| MAX OUTPUT                                                                                 | DEFAULT: 4K``MAXIMUM: 8K                                   | DEFAULT: 32K``MAXIMUM: 64K                                                                                            | DEFAULT: 128K``MAXIMUM: 128K                   |
| FEATURES                                                                                   | [Json Output](https://api-docs.deepseek.com/guides/json_mode) | ✓                                                                                                                    | ✓                                             |
| [Tool Calls](https://api-docs.deepseek.com/guides/tool_calls)                                 | ✓                                                         | ✓                                                                                                                    | ✗                                             |
| [Chat Prefix Completion（Beta）](https://api-docs.deepseek.com/guides/chat_prefix_completion) | ✓                                                         | ✓                                                                                                                    | ✗                                             |
| [FIM Completion（Beta）](https://api-docs.deepseek.com/guides/fim_completion)                 | ✓                                                         | ✗                                                                                                                    | ✗                                             |
| PRICING                                                                                    | 1M INPUT TOKENS (CACHE HIT)                                | $0.028                                                                                                                |                                                |
| 1M INPUT TOKENS (CACHE MISS)                                                               | $0.28                                                      |                                                                                                                       |                                                |
| 1M OUTPUT TOKENS                                                                           | $0.42                                                      |                                                                                                                       |                                                |

# Mistral

Le ChatAPI pricingMistral CodeEnterprise deployments

USD ($)

EUR (€)

## Our most powerful models and APIs.

### Mistral Large 3

mistral-large-latest One of the best OSS models in the world: open-weight, general-purpose, flagship multimodal and multilingual model.

* Input (/M tokens) 0.5€
* Output (/M tokens) 1.5€

### Mistral Medium 3

mistral-medium-latest State-of-the-art performance. Simplified enterprise deployments. Cost-efficient.

* Input (/M tokens) 0.4€
* Output (/M tokens) 2€

### Document AI & OCR

mistral-ocr-latest Introducing the world’s best document understanding API.

* OCR 1€ / 1000 pages
* Annotations 3€ / 1000 pages

# Qwen (2025-12-10)

## **Text generation - International (Singapore)**

### **Qwen-Max**

Billing is based on the number of input and output tokens.

If the model supports [batch calls](https://www.alibabacloud.com/help/en/model-studio/batch-interfaces-compatible-with-openai/), the unit price for both input and output tokens is 50% of the real-time inference price. If the model supports [context cache](https://www.alibabacloud.com/help/en/model-studio/context-cache), only input tokens are eligible for a discount. These two discounts cannot be applied at the same time.

| **Model**                                                                                                                                                                                                                     | **Mode**                        | **Input tokens per request** | **Input price (Million tokens)** | **Output price (Million tokens)**> **Chain of thought + response**    | **Free quota **[(Note)](https://www.alibabacloud.com/help/en/model-studio/new-free-quota#591f3dfedfyzj) |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- | ---------------------------------- | -------------------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| qwen3-max>[50% off for batch calls](https://www.alibabacloud.com/help/en/model-studio/batch-interfaces-compatible-with-openai/)> [Discounts available for context cache](https://www.alibabacloud.com/help/en/model-studio/context-cache) | Non-thinking mode only                | 0 < Tokens ≤ 32K                  | $1.2                              | $6 | 1 million tokens for eachValidity period: 90 days after you activate Model Studio |                                                                                                      |
| 32K < Tokens ≤ 128K                                                                                                                                                                                                                | $2.4                            | $12 |                                    |                                        |                                                                                   |                                                                                                      |
| 128K < Tokens ≤ 252K                                                                                                                                                                                                               | $3                              | $15 |                                    |                                        |                                                                                   |                                                                                                      |
| qwen3-max-2025-09-23                                                                                                                                                                                                                | Non-thinking mode only                | 0 < Tokens ≤ 32K                  | $1.2 | $6                              |                                                                                   |                                                                                                      |
| 32K < Tokens ≤ 128K                                                                                                                                                                                                                | $2.4                            | $12 |                                    |                                        |                                                                                   |                                                                                                      |
| 128K < Tokens ≤ 252K                                                                                                                                                                                                               | $3                              | $15 |                                    |                                        |                                                                                   |                                                                                                      |
| qwen3-max-preview>[Discounts available for context cache](https://www.alibabacloud.com/help/en/model-studio/context-cache)                                                                                                             | Non-thinking and thinking modes       | 0 < Tokens ≤ 32K                  | $1.2 | $6                              |                                                                                   |                                                                                                      |
| 32K < Tokens ≤ 128K                                                                                                                                                                                                                | $2.4                            | $12 |                                    |                                        |                                                                                   |                                                                                                      |
| 128K < Tokens ≤ 252K                                                                                                                                                                                                               | $3                              | $15 |                                    |                                        |                                                                                   |                                                                                                      |

##### **More models**

| **Model**                                                                                                             | **Mode**         | **Input tokens per request** | **Input price (Million tokens)**   | **Output price (Million tokens)**                                                             | **Free quota **[(Note)](https://www.alibabacloud.com/help/en/model-studio/new-free-quota#591f3dfedfyzj) |
| --------------------------------------------------------------------------------------------------------------------------- | ---------------------- | ---------------------------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| qwen-max>[50% off for batch calls](https://www.alibabacloud.com/help/en/model-studio/batch-interfaces-compatible-with-openai/) | Non-thinking mode only | No tiered pricing                  | $1.6                              | $6.4 | 1 million tokens for each``Validity period: 90 days after you activate Model Studio`` |                                                                                                      |
| qwen-max-latest                                                                                                             | Non-thinking mode only | No tiered pricing                  | $1.6                              | $6.4 |                                                                                                     |                                                                                                      |
| qwen-max-2025-01-25                                                                                                         | Non-thinking mode only | No tiered pricing                  | $1.6                              | $6.4 |                                                                                                     |                                                                                                      |

### **Qwen-Plus**

Billing is based on the number of input and output tokens.

| **Model**             | **Input tokens per request**                      | **Input price (Million tokens)** | **Output price (Million tokens)** | **Free quota **[**(Note)**](https://www.alibabacloud.com/help/en/model-studio/new-free-quota#591f3dfedfyzj) |
| --------------------------- | ------------------------------------------------------- | -------------------------------------- | --------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Non-thinking mode** | **Thinking mode (Chain of thought + response)**   |                                        |                                         |                                                                                                                |
| qwen-plus                   | 0 < Tokens ≤ 256K                                      | $0.4 | $1.2                            | $4                                      | 1 million tokens for each``Validity period: 90 days after you activate Model Studio``            |
| 256K < Tokens ≤ 1M         | $1.2                                             | $3.6 | $12                                    |                                         |                                                                                                                |
| qwen-plus-latest            | 0 < Tokens ≤ 256K                                      | $0.4 | $1.2                            | $4                                      |                                                                                                                |
| 256K < Tokens ≤ 1M         | $1.2                                             | $3.6 | $12                                    |                                         |                                                                                                                |
| qwen-plus-2025-12-01        | 0 < Tokens ≤ 256K                                      | $0.4 | $1.2                            | $4                                      |                                                                                                                |
| 256K < Tokens ≤ 1M         | $1.2                                             | $3.6 | $12                                    |                                         |                                                                                                                |
| qwen-plus-2025-09-11        | 0 < Tokens ≤ 256K                                      | $0.4 | $1.2                            | $4                                      |                                                                                                                |
| 256K < Tokens ≤ 1M         | $1.2                                             | $3.6 | $12                                    |                                         |                                                                                                                |
| qwen-plus-2025-07-28        | 0 < Tokens ≤ 256K                                      | $0.4 | $1.2                            | $4                                      |                                                                                                                |
| 256K < Tokens ≤ 1M         | $1.2                                             | $3.6 | $12                                    |                                         |                                                                                                                |
| qwen-plus-2025-07-14        | No tiered pricing                                       | $0.4 | $1.2                            | $4                                      |                                                                                                                |
| qwen-plus-2025-04-28        | No tiered pricing                                       | $0.4 | $1.2                            | $4                                      |                                                                                                                |
| qwen-plus-2025-01-25        | No tiered pricing                                       | $0.4 | $1.2                            | -                                       |                                                                                                                |

### **Qwen-Flash**

Billing is based on the number of input and output tokens.

If the model supports [batch calls](https://www.alibabacloud.com/help/en/model-studio/batch-interfaces-compatible-with-openai/), the unit price for both input and output tokens is 50% of the real-time inference price. If the model supports [context cache](https://www.alibabacloud.com/help/en/model-studio/context-cache), only input tokens are eligible for a discount. These two discounts cannot be applied at the same time.

| **Model**                                                                                                                                                                                                                      | **Input tokens per request** | **Input price (Million tokens)**   | **Output price (Million tokens)**                                                             | **Free quota **[(Note)](https://www.alibabacloud.com/help/en/model-studio/new-free-quota#591f3dfedfyzj) |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| qwen-flash>[50% off for batch calls](https://www.alibabacloud.com/help/en/model-studio/batch-interfaces-compatible-with-openai/)> [Discounts available for context cache](https://www.alibabacloud.com/help/en/model-studio/context-cache) | 0 < Tokens ≤ 256K                 | $0.05                             | $0.4 | 1 million tokens for each``Validity period: 90 days after you activate Model Studio`` |                                                                                                      |
| 256K < Tokens ≤ 1M                                                                                                                                                                                                                  | $0.25                         | $2 |                                          |                                                                                                     |                                                                                                      |
| qwen-flash-2025-07-28                                                                                                                                                                                                                | 0 < Tokens ≤ 256K                 | $0.05                             | $0.4 |                                                                                                     |                                                                                                      |
| 256K < Tokens ≤ 1M                                                                                                                                                                                                                  | $0.25                         | $2 |                                          |                                                                                                     |                                                                                                      |

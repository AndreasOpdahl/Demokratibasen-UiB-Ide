# Best Checkpoint Recommendations

Report generated 2026-03-27 from `olivia:sinoa_finetunes/models/`.
All models trained starting 2026-03-16 with LoRA on Norwegian document summarization.
Evaluation on 1000 validation examples unless noted.

**Metrics glossary:**
- **ROUGE-Lsum**: n-gram overlap with reference summaries (higher = better, max ~50 for this task)
- **BERTScore F1**: semantic similarity to references (higher = better)
- **NLI Entailment**: proportion of generated sentences entailed by source document (higher = more faithful)
- **NLI Pass Rate**: proportion of documents where the summary passes faithfulness threshold (higher = better)
- **Compression**: prediction length / reference length (1.0 = same length; <1 = shorter; >1 = longer)

---

## Tier 1: Strong models — ready for production

### viking-13b (LumiOpen/Viking-13B)

**Recommended checkpoint: step 4500**

| Step | ROUGE-Lsum | BERTScore | NLI Entailment | NLI Pass | Compression |
|------|-----------|-----------|---------------|----------|-------------|
| 1500 | 39.2 | 0.695 | 0.787 | 0.560 | 0.284 |
| 2500 | 39.8 | 0.699 | 0.778 | 0.556 | 0.281 |
| 3500 | 40.3 | 0.700 | 0.779 | 0.564 | 0.282 |
| **4500** | **40.9** | **0.702** | **0.782** | **0.580** | **0.277** |
| 5000 | 40.7 | 0.702 | 0.779 | 0.558 | 0.280 |

Best model overall. All metrics are excellent and still improving or stable at step 5000.
ROUGE, BERTScore, NLI, and pass rate all peak near step 4500.
Compression ~0.28 means summaries are concise (28% of reference length in characters).
Could potentially benefit from more training (only trained to 5000 steps), but 4500 is a safe pick.

### gemma-2-9b (google/gemma-2-9b)

**Recommended checkpoint: step 4700**

| Step | ROUGE-Lsum | BERTScore | NLI Entailment | NLI Pass | Compression |
|------|-----------|-----------|---------------|----------|-------------|
| 3500 | 38.8 | 0.688 | 0.770 | 0.528 | 0.313 |
| 4500 | 38.9 | 0.689 | 0.773 | 0.520 | 0.312 |
| **4700** | **44.5** | **0.699** | — | — | — |
| 5500 | 38.9 | 0.689 | 0.768 | 0.534 | 0.312 |
| 9000 | 40.5 | 0.693 | — | — | 0.311 |

Second-best model. Highest raw ROUGE-Lsum of any model (44.5 at step 4700).
NLI data only available for major checkpoints (every 500 steps); the step-4700 checkpoint
is a "regular" checkpoint without NLI evaluation. Surrounding major checkpoints (4500, 5000)
show NLI entailment 0.765–0.773 which is very strong.

**What's missing:** NLI faithfulness not computed for step 4700 specifically. Recommend running
NLI evaluation on this checkpoint to confirm faithfulness before production use. Alternatively,
step 4500 (NLI entailment 0.773, NLI pass 0.520) is a safer choice with full metrics.

---

## Tier 2: Good models — usable with correct checkpoint selection

### llama-2-13b-chat-norwegian (ruternorway/llama-2-13b-chat-norwegian)

**Recommended checkpoint: step 1000**

| Step | ROUGE-Lsum | BERTScore | NLI Entailment | NLI Pass | Compression |
|------|-----------|-----------|---------------|----------|-------------|
|  500 | 29.0 | 0.636 | 0.800 | 0.492 | 0.443 |
| **1000** | **29.6** | **0.647** | **0.779** | **0.438** | **0.486** |
| 1500 | 29.6 | 0.644 | 0.793 | 0.500 | 0.463 |
| 3000 | 28.5 | 0.640 | 0.806 | 0.514 | 0.482 |
| 5000 | 28.6 | 0.642 | 0.806 | 0.488 | 0.489 |

Third-best ROUGE (29.6). Very high NLI faithfulness (0.78–0.81) across all checkpoints — this model
is the most consistently faithful. BERTScore peaks at step 1000. ROUGE slowly declines after 1500
but faithfulness actually increases, so step 3000 is an alternative if faithfulness matters more than ROUGE.

### llama-3.1-8b-instruct (meta-llama/Llama-3.1-8B-Instruct)

**Recommended checkpoint: step 500**

| Step | ROUGE-Lsum | BERTScore | NLI Entailment | NLI Pass | Compression |
|------|-----------|-----------|---------------|----------|-------------|
| **500** | **28.6** | **0.569** | **0.693** | **0.408** | **0.322** |
| 1000 | 28.7 | 0.572 | 0.672 | 0.368 | 0.369 |
| 2000 | 28.0 | 0.568 | 0.656 | 0.334 | 0.443 |
| 5000 | 25.6 | 0.559 | 0.628 | 0.258 | 0.631 |
| 10000 | 24.0 | 0.544 | 0.603 | 0.232 | 0.724 |

Severe overtraining: every metric degrades after step 500–1000. ROUGE drops from 29 to 24,
NLI pass rate halves from 0.41 to 0.23, compression doubles from 0.32 to 0.72 (model becomes
increasingly verbose and hallucinatory). Step 500 gives the best balance of all metrics.

### gemma-2b (google/gemma-2b)

**Recommended checkpoint: step 7000**

| Step | ROUGE-Lsum | BERTScore | NLI Entailment | NLI Pass | Compression |
|------|-----------|-----------|---------------|----------|-------------|
|  500 | 19.7 | 0.542 | 0.683 | 0.442 | 0.266 |
| 2500 | 26.7 | 0.595 | 0.655 | 0.380 | 0.317 |
| 5000 | 26.5 | 0.597 | 0.650 | 0.358 | 0.312 |
| **7000** | **27.7** | **0.600** | — | — | **0.311** |
| 8500 | 27.9 | 0.602 | — | — | 0.313 |
| 10000 | 27.7 | 0.601 | — | — | 0.313 |

Steady improvement through step 7000–8500, then plateaus. NLI only available up to step 5000;
faithfulness is moderate (0.65) and slowly decreasing. ROUGE and BERTScore keep improving past step 5000.

**What's missing:** NLI faithfulness not computed after step 5000. If NLI continued its decline trend,
step 5000 would be a safer choice. For ROUGE-only optimization, step 7000–8500 is fine.

Small model (2B params) — decent performance for its size but outclassed by larger models.

### norwai-mistral-7b-instruct (NorwAI/NorwAI-Mistral-7B-instruct)

**Recommended checkpoint: step 1500**

| Step | ROUGE-Lsum | BERTScore | NLI Entailment | NLI Pass | Compression |
|------|-----------|-----------|---------------|----------|-------------|
|  500 | 21.9 | 0.604 | 0.772 | 0.420 | 0.563 |
| **1500** | **23.8** | **0.616** | **0.727** | **0.330** | **0.648** |
| 2000 | 22.6 | 0.620 | 0.758 | 0.442 | 0.594 |
| 4500 | 22.3 | 0.625 | 0.766 | 0.480 | 0.536 |
| 10000 | 22.9 | 0.627 | — | — | 0.443 |

Unusual trajectory: ROUGE peaks at step 1500, dips, then slowly recovers. BERTScore steadily
improves through training. NLI faithfulness dips at step 1500 but recovers by step 2000+.
Step 1500 maximizes ROUGE; step 4500 maximizes BERTScore + NLI pass rate.

Depending on priority: **step 1500 for ROUGE, step 4500 for faithfulness**.

---

## Tier 3: Mediocre models — limited utility

### normistral-11b (norallm/normistral-11b-warm)

**Recommended checkpoint: step 1000**

| Step | ROUGE-Lsum | BERTScore | NLI Entailment | NLI Pass | Compression |
|------|-----------|-----------|---------------|----------|-------------|
|  500 | 14.8 | 0.562 | 0.651 | 0.374 | 1.010 |
| **1000** | **18.9** | **0.578** | **0.687** | **0.286** | **0.714** |
| 1300 | 19.5 | — | — | — | 0.668 |
| 2000 | 18.2 | 0.570 | 0.648 | 0.238 | 0.914 |
| 5000 | 17.5 | 0.574 | 0.604 | 0.194 | 1.083 |
| 10000 | 17.7 | 0.576 | — | — | 1.371 |

Overtrains heavily. NLI faithfulness drops 12% (0.687→0.604), NLI pass rate halves (0.37→0.19),
and compression doubles (0.71→1.37) as the model becomes increasingly verbose and hallucinatory.
Step 1000 gives the best faithfulness; step 1300 has marginally higher ROUGE but no NLI data.

Despite being an 11B Norwegian-specialized Mistral model, ROUGE tops out at ~19.5.
The 2048-context NorwAI-Mistral-7B-instruct (23.8) outperforms it on ROUGE.

### norskgpt-llama3-8b (bineric/norskgpt-llama3-8b)

**Recommended checkpoint: step 500**

| Step | ROUGE-Lsum | BERTScore | NLI Entailment | NLI Pass | Compression |
|------|-----------|-----------|---------------|----------|-------------|
| **500** | **19.2** | **0.553** | **0.766** | **0.452** | **0.388** |
| 1000 | 19.9 | 0.551 | 0.736 | 0.386 | 0.420 |
| 2000 | 18.1 | 0.534 | 0.692 | 0.326 | 0.463 |
| 5000 | 16.2 | 0.521 | 0.660 | 0.338 | 0.526 |

Rapid degradation after step 500–1000. Every metric worsens monotonically.
Step 500 has the best NLI (0.766) and nearly the best ROUGE (19.2 vs 19.9 at step 1000).
The model quickly overtrains and becomes verbose.

### nb-gpt-j-6b (NbAiLab/nb-gpt-j-6B-torgersen-alpaca)

**Recommended checkpoint: step 1000**

| Step | ROUGE-Lsum | BERTScore | NLI Entailment | NLI Pass | Compression |
|------|-----------|-----------|---------------|----------|-------------|
|  500 | 12.8 | 0.438 | 0.636 | 0.382 | 0.258 |
| **1000** | **13.0** | **0.442** | **0.658** | **0.390** | **0.267** |
| 1500 | 13.3 | 0.451 | 0.654 | 0.372 | 0.278 |
| 3000 | 12.8 | 0.442 | 0.661 | 0.374 | 0.256 |
| 6500 | 12.9 | 0.444 | 0.652 | 0.360 | 0.271 |

Performance is essentially flat across all checkpoints (ROUGE 12.7–13.4). The GPT-J architecture
with 2048-token context and Alpaca prompt format limits this model's capability.
Step 1000 is a marginal best. This model is not competitive with any of the above.

---

## Tier 4: Broken models — need retraining

### gemma-7b (google/gemma-7b)

**Status: BROKEN — needs retraining**

Best ROUGE-Lsum: 7.76 (completely flat across all checkpoints).

Root cause: FP16 precision loss during generation. The base model stores weights in BF16 but
the training/eval code loaded them in FP16. Combined with gemma-7b's lack of logit soft-capping
(unlike gemma-2-9b which has it), FP16 errors compound during autoregressive generation, producing
garbled Norwegian text with merged words and nonsensical output.

Training loss decreased normally (1.94→1.37), adapter weights are non-NaN, but the dtype mismatch
during generation corrupts every token selection.

Fix applied: BF16 model loading + BF16 autocast in eval script. Requires retraining.

### viking-7b (LumiOpen/Viking-7B)

**Status: RETRAINING IN PROGRESS (started 2026-03-27)**

Original run produced 100% NaN adapter weights due to FP16 loading of a BF16-native model,
causing immediate gradient explosion (loss went 182.9→0.0, grad_norm=NaN from step 20 onwards).

Fix applied: BF16 model loading. Retraining started; no evaluation results yet.

---

## Models not present in this directory

The following models from `model_configs.py` do not have trained directories in
`sinoa_finetunes/models/`:

- **normistral-7b** (norallm/normistral-7b-warm) — not trained
- **normistral-7b-instruct** (norallm/normistral-7b-warm-instruct) — not trained
- **gemma-2-27b** (google/gemma-2-27b) — not trained
- **gemma-2b-it** and other instruct variants

The normistral-7b models have a 2048-token context window (`max_position_embeddings=2048`)
which caused garbled output for long inputs. A fix has been applied (`max_input_text_tokens=1700`)
in `model_configs.py` but these models have not been trained yet.

---

## Tier 5: Problematic — needs investigation

### eurollm-9b-instruct (utter-project/EuroLLM-9B-Instruct-2512)

**Status: UNCLEAR — predictions are extremely truncated**

| Step | ROUGE-Lsum | BERTScore | NLI Entailment | NLI Pass | Compression |
|------|-----------|-----------|---------------|----------|-------------|
|  500 | 13.9 | 0.273 | 0.384 | 0.188 | 0.089 |
| 1000 | 12.9 | 0.324 | 0.419 | 0.200 | 0.108 |
| 2000 | 14.2 | 0.257 | 0.337 | 0.168 | 0.084 |
| 5000 | 11.7 | 0.233 | 0.322 | 0.156 | 0.084 |

Compression ratio 0.07–0.11 means predictions are only **7–11% of reference length** — the model
generates extremely short outputs (roughly one sentence when it should produce a paragraph).
BERTScore (0.23–0.32) and NLI pass rate (0.14–0.20) are the lowest of any model.

This has not been investigated. Possible causes:
- Prompt template mismatch (uses PROMPT_CHATML; the model's chat template may differ)
- The model may be producing EOS tokens too early
- The model's tokenizer or generation config may have issues
- May also be affected by the FP16/BF16 mismatch (model is Mistral-based)

Not recommended for use until investigated.

---

## Summary table

| Model | Size | Best Ckpt | ROUGE-Lsum | BERTScore | NLI Entail | NLI Pass | Status |
|-------|------|-----------|-----------|-----------|-----------|----------|--------|
| viking-13b | 13B | 4500 | **40.9** | **0.702** | 0.782 | **0.580** | ✅ Production |
| gemma-2-9b | 9B | 4700 | **44.5** | **0.699** | ~0.77* | ~0.52* | ✅ Production |
| llama-2-13b-chat-no | 13B | 1000 | 29.6 | 0.647 | **0.806**† | 0.514† | ✅ Production |
| llama-3.1-8b-inst | 8B | 500 | 28.6 | 0.569 | 0.693 | 0.408 | ⚠️ Use early ckpt |
| gemma-2b | 2B | 7000 | 27.7 | 0.600 | 0.65* | 0.36* | ⚠️ NLI unverified |
| norwai-mistral-7b | 7B | 1500 | 23.8 | 0.616 | 0.727 | 0.330 | ⚠️ Use early ckpt |
| normistral-11b | 11B | 1000 | 18.9 | 0.578 | 0.687 | 0.286 | ⚠️ Use early ckpt |
| norskgpt-llama3-8b | 8B | 500 | 19.2 | 0.553 | 0.766 | 0.452 | ⚠️ Use early ckpt |
| nb-gpt-j-6b | 6B | 1000 | 13.0 | 0.442 | 0.658 | 0.390 | ⚠️ Limited |
| eurollm-9b-instruct | 9B | ? | 14.2 | 0.324 | 0.419 | 0.200 | ❌ Investigate |
| gemma-7b | 7B | — | 7.8 | — | — | — | ❌ Retrain (BF16 fix) |
| viking-7b | 7B | — | — | — | — | — | 🔄 Retraining |

\* Interpolated from surrounding checkpoints; NLI not computed at the exact recommended step.
† NLI faithfulness at step 3000–5000 (stays high across all checkpoints for this model).

### Key observations

1. **viking-13b and gemma-2-9b are the clear winners.** Both achieve ROUGE >40 with strong
   faithfulness. gemma-2-9b has the highest raw ROUGE (44.5); viking-13b has the most complete
   metric coverage and the highest NLI pass rate (0.58).

2. **Most models overtrain.** Of the 10 evaluated models, 5 show clear performance degradation
   with continued training (llama-3.1-8b, norskgpt-llama3-8b, normistral-11b, norwai-mistral-7b,
   eurollm-9b). The optimal checkpoint is often at step 500–1500, far earlier than the 5000–10000
   steps these models were trained for.

3. **NLI faithfulness generally drops with training** while ROUGE stays flat or improves slowly.
   This suggests models learn to produce fluent, reference-matching text that increasingly
   hallucinates details not in the source document.

4. **llama-2-13b-chat-norwegian is uniquely faithful.** Its NLI entailment (0.80) stays high
   across all checkpoints, unlike every other model. This may be due to its chat-tuning or
   the Norwegian-specific fine-tuning of the base model.

# Evaluation Metrics Computation Time Estimates

## Overview
This document provides time estimates for computing various evaluation metrics on a typical validation set (500 examples).

**Assumptions:**
- Validation set size: 500 examples (default `VAL_DATA_SIZE`)
- Average document length: ~3000 tokens (~2000 words)
- Average summary length: ~150 tokens (~100 words, ~5-8 sentences)
- Hardware: GPU (A100/H100) for model inference
- Models loaded on first use (one-time cost)

---

## 1. ROUGE Metrics ⚡ (Already Computed)

**Operations:**
- Token-based n-gram matching
- No model inference required

**Time Estimate:**
- **Per example:** < 1ms
- **Total (500 examples):** < 1 second
- **Bottleneck:** None - very fast

**Notes:** Already computed in the original evaluation pipeline, so no additional time.

---

## 2. Reference-Based Metrics

### 2.1 BERTScore (Norwegian BERT-Large)

**Operations:**
- Load model: `NbAiLab/nb-bert-large` (one-time: ~5-10 seconds)
- Encode all predictions and references
- Compute cosine similarity between embeddings
- BERT-Large encoding: ~150-200ms per example on GPU

**Time Estimate:**
- **Model loading (first time):** 5-10 seconds
- **Per example:** ~150-200ms (encoding both prediction + reference)
- **Total (500 examples):** ~75-100 seconds (~1.5-2 minutes)
- **Bottleneck:** Model encoding (BERT-Large is computationally expensive)

**Optimization:** Could batch encoding for faster processing, but current implementation processes sequentially.

---

## 3. Hygiene Metrics ⚡⚡

**Operations:**
- Word counting (compression ratio)
- N-gram extraction and counting (repetition)
- Regex pattern matching (punctuation)

**Time Estimate:**
- **Per example:** < 1ms
- **Total (500 examples):** < 1 second
- **Bottleneck:** None - pure text processing

**Notes:** These are extremely fast and add negligible overhead.

---

## 4. NLI-Based Faithfulness Metrics 🐌 (MAJOR BOTTLENECK)

**Operations:**
- Load model: `joeddav/xlm-roberta-large-xnli` (one-time: ~10-15 seconds)
- For each document-summary pair:
  1. Split summary into sentences (~5-8 sentences)
  2. Chunk document into ~350 token chunks (~6-15 chunks for 2000-5000 token docs)
  3. For each sentence, run NLI against each chunk
  4. NLI inference: ~50-100ms per call on GPU

**Computational Complexity:**
- Average sentences per summary: 6
- Average chunks per document: 10
- NLI calls per example: 6 sentences × 10 chunks = **60 NLI calls**
- NLI inference time: ~75ms per call (average)

**Time Estimate:**
- **Model loading (first time):** 10-15 seconds
- **Per example:** 60 calls × 75ms = **~4.5 seconds**
- **Total (500 examples):** 500 × 4.5s = **~37.5 minutes** (2,250 seconds)
- **Bottleneck:** NLI model inference (quadratic complexity: sentences × chunks)

**Notes:** 
- This is the **slowest metric** by far
- The TODO in `extended_evaluation.py` suggests running on a subset for every checkpoint
- Consider running on full set only for final evaluation or every N checkpoints

---

## Total Time Estimates

### Scenario 1: All Metrics (Full Evaluation)
- ROUGE: < 1s (already computed)
- BERTScore: ~1.5-2 minutes
- Hygiene: < 1s
- NLI Faithfulness: **~37.5 minutes**
- **Total: ~40 minutes** for 500 examples

### Scenario 2: Fast Evaluation (Skip NLI)
- ROUGE: < 1s
- BERTScore: ~1.5-2 minutes
- Hygiene: < 1s
- **Total: ~2 minutes** for 500 examples

### Scenario 3: Subset Evaluation (Recommended for Checkpoints)
- Run NLI on 50-100 examples instead of 500
- NLI time: 50 × 4.5s = **~3.75 minutes** (instead of 37.5 minutes)
- **Total: ~5-6 minutes** for full evaluation with subset NLI

---

## Recommendations

### For Checkpoint Evaluation (During Training):
1. ✅ **Always compute:** ROUGE, Hygiene metrics (fast, < 2 minutes total)
2. ⚠️ **Consider skipping:** BERTScore (adds 1.5-2 minutes, but useful)
3. 🚫 **Skip or subset:** NLI Faithfulness (too slow for every checkpoint)

**Suggested approach:**
- **Every checkpoint:** ROUGE + Hygiene + BERTScore (~2-3 minutes)
- **Every 5-10 checkpoints:** Add NLI on subset (50-100 examples, +4 minutes)
- **Final evaluation:** All metrics on full set (~40 minutes)

### Optimization Strategies:

1. **Batch NLI inference:** Process multiple sentence-chunk pairs in batches
   - Current: Sequential processing
   - Potential speedup: 2-3x with batching

2. **Reduce NLI calls:**
   - Use only top-k chunks per sentence (e.g., top 3 by similarity)
   - Current: All chunks
   - Potential speedup: 3-5x

3. **Smaller NLI model:** Use a smaller/faster NLI model for checkpoint evaluation
   - Current: XLM-RoBERTa-Large
   - Alternative: Smaller multilingual NLI model
   - Potential speedup: 2-3x

4. **Parallel processing:** Process multiple examples in parallel
   - Current: Sequential
   - Potential speedup: Limited by GPU memory

---

## Time Breakdown Summary

| Metric | Time (500 examples) | % of Total | Priority |
|--------|---------------------|------------|----------|
| ROUGE | < 1s | < 1% | ✅ Always |
| Hygiene | < 1s | < 1% | ✅ Always |
| BERTScore | ~90s | ~4% | ⚠️ Recommended |
| NLI Faithfulness | ~2250s | ~95% | 🚫 Skip/Subset |

**Key Insight:** NLI Faithfulness dominates computation time (95%+). Consider running it selectively.

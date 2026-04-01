# Pairwise evaluation method

This document describes the pairwise G-Eval workflow implemented in the `pairwise_eval` package: how data are prepared, how comparisons are built, how judgments are recorded, and how **win rates** and **Bradley–Terry** strengths are computed. Wording matches the current code (configuration in `pairwise_eval/config.py`).

---

## 1. Goal

For each document (source text), we compare **summaries produced by different systems** under the same setting. Instead of scoring each summary in isolation, we use **pairwise comparisons**: a judge sees two summaries (A vs B) and picks a winner (or a tie) along evaluation dimensions such as faithfulness, correctness, and completeness. Aggregating many such choices yields **win rates** per model and, optionally, **Bradley–Terry** latent strengths.

---

## 2. Data: long-form table

All systems under evaluation—including the **gold reference summary**—appear as rows in a single long table (`long_df`). Each row has (among others):

- **`doc_id`**: document identifier  
- **`source_text`**: input article  
- **`model_id`**: which system produced the summary (checkpoint filename stem for JSONL models, or a fixed id for the gold summary)  
- **`summary_text`**: the summary being compared  
- **`reference_summary`**: the gold reference text for that document (metadata; repeated for every model on that doc)

**Real data (`Data/eval/*.jsonl`):**

1. **Checkpoint rows** — One JSONL file per model; each line is one document. Rows are stacked so `model_id` is the file stem and `summary_text` comes from the `prediction` field. Files are checked for aligned `input_text` across models.  
2. **Gold summary as a model** — One extra row per `doc_id` is appended. Its `summary_text` is the JSONL **`reference`** field. Across files, `reference` is assumed identical for the same line index, so any row for that document suffices to read it (the implementation uses the first row per `doc_id`).

The configurable label for that gold-summary “model” is `REFERENCE_SUMMARY_MODEL_ID` (currently **`GPT4o-mini`** in `config.py`). That name is for tables and exports only; the text still comes from the dataset’s `reference` column, not from an external API call.

**Toy data** follows the same pattern: finetuned-style models plus one gold row per document via the same append step.

**Subset of documents:** `MAX_DOCUMENTS` in `config.py` limits evaluation to the first \(N\) distinct `doc_id` values in first-seen order (`long_df_head_documents`). Set it to `None` to use every loaded document.

---

## 3. Building pairwise comparisons

For each document, let \(M\) be the set of `model_id`s that have a row for that doc.

- All unordered pairs \(\{a,b\}\) with \(a,b \in M\), \(a \neq b\), are enumerated.  
- Up to **`n_pairs`** pairs are drawn **uniformly at random without replacement** (default `n_pairs = 4`; capped by the number of available pairs).  
- For each selected pair, **left vs right** order is **randomized** (50/50) so positional bias is averaged out.

Each pair row stores `doc_id`, `left`, `right`, and the corresponding `summary_text` fields (`sumleft`, `sumright`). The same pair sampling is reused for every evaluation dimension and judge: only the judgment outcome changes per \((\text{judge}, \text{dimension})\).

**Implication:** With many models, a small `n_pairs` means each document only sees a **subset** of possible matchups; some models (including the gold-summary id) may appear rarely or not at all on a given doc in that sample.

---

## 4. Judgments (G-Eval tables)

**Dimensions** (default): `faithfulness`, `correctness`, `completeness`.

**Judges:** The tuple `JUDGES` in `config.py` lists judge identifiers. Each entry is both the key for G-Eval tables/exports and, when using the local LLM path, the **`model`** string passed to the chat API (e.g. `google/gemma-3-4b`). You can add several models to compare judge agreement; `HUMAN_JUDGES` / `LLM_JUDGES` split win-rate pooling when humans are used later.

For every pair row and every \((\text{judge}, \text{dimension})\), the pipeline attaches document context (`source_text`, `reference_summary`) and records a judgment. The result columns include:

- **`choice_side`**: `left`, `right`, or `tie`  
- **`chosen`**: the winning `model_id` when there is a strict winner; missing when tied  
- **`rationale`**: optional text (e.g. from an LLM judge)

**Tie rule for downstream metrics:** A row counts as a tie if `choice_side == "tie"` **or** `chosen` is missing.

The CLI can call a **local LLM** (`make_local_llm_evaluate_fn` → `geval_local_judge`) or **`mock_evaluate_pair`** (random A/B/tie with **`MOCK_TIE_PROB`**). Aggregation (win rates, Bradley–Terry) is the same either way.

For the mock judge, RNG is **deterministic per** \((\text{judge}, \text{dimension})\) from `DEFAULT_GEVAL_BASE_SEED`.

**Resumable checkpoints:** If `GEVAL_CHECKPOINT_DIR` in `config.py` is set, `build_geval_tables` writes **one JSON line per new judgment** (with `flush` + `fsync`) under `.deepeval/geval_judgment_checkpoints/`, one file per \((\text{judge}, \text{dimension})\). On restart, judgments whose **stable key** (judge, dimension, doc, left/right model ids, and a hash of both summary texts) already appear in that file are **skipped**; only missing keys call the judge. Set `GEVAL_CHECKPOINT_DIR = None` to disable. Delete the checkpoint files (or a single `.jsonl`) to force a full re-run for that slice.

---

## 5. Win rates

For each model \(m\):

- **Opportunities** — Each time \(m\) appears as `left` or `right` in a comparison counts as one opportunity (for that judge and dimension, or pooled as defined below).  
- **Wins** — If the judge picks a strict winner, the winner gets **+1** win. If the comparison is a **tie**, **both** sides get **+0.5** wins (so the total wins awarded per comparison remain 1.0, split between the two).

\[
\text{win\_rate}(m) = \frac{\text{wins}(m)}{\text{opportunities}(m)}
\]

where `wins(m)` includes fractional contributions from ties.

**Exported “paper” table:** Human judges in `HUMAN_JUDGES` are pooled over all dimensions; LLM judges in `LLM_JUDGES` are pooled the same way (`win_rate_human`, `win_rate_llm_pooled`). If there are no human judges, the human columns are empty/NaN.

**Per-dimension tables:** Same tie logic, reported separately for each dimension and each judge.

---

## 6. Bradley–Terry model

Pairwise results are summarized with a **Bradley–Terry** model on directed win counts.

### 6.1 Win count matrix

For a fixed \((\text{judge}, \text{dimension})\) and a fixed ordering of models `model_order`, build matrix \(W\) where \(W_{ij}\) counts how often \(i\) was chosen over \(j\) when \(i\) and \(j\) were the two sides:

- **Strict win for \(i\):** \(W_{ij} \mathrel{+}= 1\)  
- **Tie:** \(W_{ij} \mathrel{+}= 0.5\) and \(W_{ji} \mathrel{+}= 0.5\)

So ties contribute symmetrically to off-diagonals (a common pragmatic approximation when ties are rare).

### 6.2 Fitting

Let \(\beta_k\) be log-strength parameters. The likelihood uses standard BT terms for each directed count \(W_{ij}\) with \(i \neq j\). The model is **identified** during optimization by fixing **one** index `ref_idx` so \(\beta_{\text{ref_idx}} = 0\) (default reference: `REFERENCE_SUMMARY_MODEL_ID` when present in `model_order`, else the first model). Optimization uses **L-BFGS-B** (`scipy.optimize.minimize`).

### 6.3 Reported \(\theta\) (strengths)

After MLE, the code **re-centers** \(\beta\) by subtracting the **mean** across models: \(\beta_k \leftarrow \beta_k - \frac{1}{K}\sum_j \beta_j\), then reports \(\theta_k = \exp(\beta_k)\).

- **Pairwise win probabilities implied by BT are unchanged** by this shift (only differences \(\beta_i - \beta_j\) matter).  
- **Geometric mean** of the reported \(\theta\) over models is **1** for that fit.

Exports include long and wide tables of \(\beta\), \(\theta\), optimizer success, and the number of pairwise comparisons underlying each fit.

---

## 7. Exports

Running the main pipeline writes artifacts under **`.deepeval/geval_exports/`** (JSON pairwise and G-Eval tables, CSV summaries, Markdown/LaTeX reports). See `pairwise_eval/io_export.py` for layout (`json/`, `tables/`, `reports/`).

---

## 8. Limitations and design choices (honest scope)

1. **Subsampled pairs** — `n_pairs` per document limits coverage; estimates are conditional on which matchups were sampled.  
2. **Ties in Bradley–Terry** — Splitting ties 0.5/0.5 into \(W\) is standard in tools but is still an approximation to a full tie model.  
3. **Mock judge** — Default runs are not human or LLM judgments until `mock_evaluate_pair` is replaced.  
4. **Same pairs for all dimensions** — Efficiency and comparability; if a judge should see different pairs per dimension, the sampling would need to change.

---

## 9. Code map

| Stage | Module / entry |
|--------|----------------|
| Data stacking + gold row | `pairwise_eval/data.py` |
| Pair sampling | `pairwise_eval/pairs.py` |
| Judging + G-Eval tables | `pairwise_eval/judging.py` |
| Win rates | `pairwise_eval/win_rates.py` |
| Bradley–Terry | `pairwise_eval/bradley_terry.py` |
| CLI | `pairwise_eval/__main__.py` |
| Defaults | `pairwise_eval/config.py` |

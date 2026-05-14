# Running pairwise G-Eval experiments

This guide is for running the pipeline on **new summarization checkpoints** (one eval folder per base model under `Data/eval/`), then auditing API failures, retrying failed judgments, and opening the HTML viewers.

All commands assume the **repository root** as the current working directory unless noted.

---

## Prerequisites

1. **Python environment** with dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. **API keys** for whichever cloud judges you enable in `pairwise_eval/config.py` (see below). Put them in a repo-root `.env` file (optional; keys are read from the environment). Typical variables:

   - `OPENAI_API_KEY` — OpenAI judges (`gpt-5-mini`, etc.)
   - `GOOGLE_API_KEY` or `GEMINI_API_KEY` — Gemini judges (`google/...`)
   - `ANTHROPIC_API_KEY` — Anthropic judges (`anthropic/...`)
   - `MISTRAL_API_KEY` — Mistral judges (`mistral-medium-latest`, etc.)

3. **Eval data layout**: under `Data/eval/<your_model_name>/`, one `*.jsonl` per checkpoint (file stem = `model_id`). Each JSONL must have aligned rows across files (same `input_text` order); the loader assigns `doc_id` as `doc_1`, `doc_2`, … per row index. See `pairwise_eval.data.stack_eval_jsonl_checkpoints_long_df`.

4. **G-Eval prompts**: for each dimension name in `EVAL_DIMENSIONS`, there must be a template file:

   `Data/prompts/geval/<dimension>.txt`

---

## What to edit in `pairwise_eval/config.py`

These values are read when you run `python -m pairwise_eval` (or `python pairwise_eval/__main__.py`). **Save the file before launching**; there is no CLI override for most of these.

### `EVAL_DIMENSIONS`

Tuple of dimension names (strings). Each name must match a prompt file `Data/prompts/geval/<name>.txt`.

Example:

```python
EVAL_DIMENSIONS: tuple[str, ...] = (
    "relevance",
    "consistency",
    "newsworthiness",
    "hygiene",
)
```

**Cost:** the pipeline runs **one LLM judgment per (pair row × judge × dimension)**. More dimensions ⇒ proportionally more API calls.

---

### `JUDGES`

Tuple of judge identifiers. Each id is also used for routing to the correct API:

- **OpenAI** — id must be in `OPENAI_JUDGE_IDS` (default: `{"gpt-5-mini"}`).
- **Gemini** — id must look like `google/<…>` and be in `GEMINI_JUDGE_IDS`.
- **Anthropic** — id must look like `anthropic/<…>` and be in `ANTHROPIC_JUDGE_IDS`.
- **Mistral** — id in `MISTRAL_JUDGE_IDS` (uses the same JSON shape as OpenAI against `MISTRAL_CHAT_COMPLETIONS_URL`).

Any judge string **not** in one of those sets is treated as a **local** LM Studio–style judge (`LOCAL_LLM_CHAT_URL`); that path is not used when all judges are cloud APIs.

If you add a new judge string, you must:

1. Add it to `JUDGES`, and  
2. Add it to the correct `*_JUDGE_IDS` frozenset, and  
3. Set the matching API key / optional `*_JUDGE_TO_API_MODEL` remap (Gemini/Anthropic often need a stable key → current API model id).

---

### `MAX_DOCUMENTS`

```python
MAX_DOCUMENTS: int | None = None  # example: use full corpus
MAX_DOCUMENTS: int | None = 50   # example: first 50 distinct doc_id (first-seen order)
```

- **`None`**: use every document present in the stacked eval JSONLs.
- **`N` (int ≥ 1)**: keep only the first **N** distinct `doc_id` values (order = first appearance in the long dataframe after stacking).

The number in `reports/results_summary.md` (**“Documents in subset”**) is `long_df["doc_id"].nunique()` at export time. It will be **≤ `MAX_DOCUMENTS`** and never larger than the number of documents actually present in the eval files.

---

### `N_PAIRS_PER_DOCUMENT`

How many random (or balanced) **unordered** model pairs to sample **per document** for pairwise comparison, capped by `C(n_models, 2)`.

Larger values ⇒ more pair rows ⇒ more judgments. Typical values: `4`–`8`.

---

### `BALANCED_PAIR_SAMPLING`

- **`False`**: each document independently samples `N_PAIRS_PER_DOCUMENT` pairs uniformly at random (legacy behavior).
- **`True`**: greedy global balancing so each model / unordered pair gets more even coverage across documents (recommended for new runs).

---

### Other useful settings (optional)

| Setting | Role |
|--------|------|
| `EVAL_DATA_DIR` | `None` → auto `Data/eval`. Or set to an alternate eval root (path under `REPO_ROOT` if relative). |
| `GEVAL_EXPORT_DIRNAME` | Subfolder name under `.deepeval/` for exports (default `geval_exports`). Change if you want a separate export tree. |
| `GEVAL_CHECKPOINT_DIR` | Where append-only judgment JSONL checkpoints live; default `.deepeval/geval_judgment_checkpoints`. Use a different path to avoid mixing two studies. |
| `EXTEND_PAIRS_TABLE_JSON` | Usually `None`. Set only if you intentionally want to reuse a specific `pairs_table.json` path; the CLI also auto-loads `<export_leaf>/json/pairs_table.json` when present. |
| `REFERENCE_SUMMARY_MODEL_ID` | Must match how gold rows are labeled (default `GPT4o-mini`); JSONL file stems must not use this id. |
| `GEMINI_MAX_REQUESTS_PER_MINUTE` | Proactive spacing between **Gemini-only** calls; `0` disables. Other providers use retry backoff on errors, not this throttle. |

---

## Quick toggles in `pairwise_eval/__main__.py`

At the top of the file (not in `config.py`):

| Variable | Meaning |
|----------|---------|
| `USE_TOY_DATA` | `False` for real data; `True` only for a tiny built-in smoke test. |
| `USE_LLM_JUDGE` | `True` → real cloud/local LLM judges; `False` → mock judge (fast, no API keys). |

---

## Running the pipeline

From the repo root:

```bash
python -m pairwise_eval
```

or:

```bash
python pairwise_eval/__main__.py
```

**Behavior:**

- If `Data/eval/` (or `EVAL_DATA_DIR`) contains **`*.jsonl` directly**, one run writes to `.deepeval/<GEVAL_EXPORT_DIRNAME>/` (flat layout).
- If it contains **only subfolders** with `*.jsonl` (e.g. `Data/eval/viking-13b/`, `Data/eval/norwai-mistral-7b/`), the pipeline runs **once per subfolder** and writes:

  - `.deepeval/<GEVAL_EXPORT_DIRNAME>/<folder_name>/` — exports (JSON, tables, reports)  
  - `.deepeval/geval_judgment_checkpoints/<folder_name>/` — resumable JSONL checkpoints (when `GEVAL_CHECKPOINT_DIR` is set)

**Resume:** re-running the same model is safe: existing judgment keys in the JSONL files are skipped; only missing (judge × dimension × pair) work is issued.

---

## Audit: list `[api_error]` failures (does not fix anything)

**Script:** `scripts/audit_geval_api_failures.py`

**All judgment leaves** under the default checkpoint root (or whatever `GEVAL_CHECKPOINT_DIR` is in config):

```bash
python scripts/audit_geval_api_failures.py
```

**One model (“leaf”) only** — pass the folder that contains `*.jsonl` files:

```bash
python scripts/audit_geval_api_failures.py \
  --checkpoint-dir .deepeval/geval_judgment_checkpoints/viking-13b
```

Output: grouped counts by dimension × judge × failure category (e.g. `timeout`, `http_400`, `errno_89`).

---

## Retry: re-call the LLM for failed lines (fixes checkpoints, then refreshes exports)

**Script:** `scripts/retry_geval_checkpoint_503.py`

It re-runs judgments for lines whose rationale is `[api_error]` in selected **categories**, rewrites matching lines **in place** in the same JSONL files, then (by default) rebuilds exports so `results_summary.md`, win-rate tables, etc. match.

**Important:**

- Retry uses **`pairwise_eval.config`** (`MAX_DOCUMENTS`, `N_PAIRS_PER_DOCUMENT`, seeds, `EVAL_DATA_DIR`, …). These should match the run that produced the checkpoints, or stable keys may not line up with pair rows.
- The default retry set includes common HTTP errors and `timeout` / `connection_error` / `json_parse`, but **not** every possible label. For example **`errno_89`** must be added explicitly.

**Dry run** (no API calls, no writes):

```bash
python scripts/retry_geval_checkpoint_503.py --dry-run
```

**All leaves**, default error buckets **plus** `errno_89`:

```bash
python scripts/retry_geval_checkpoint_503.py --only-errors all,errno_89
```

**One leaf only:**

```bash
python scripts/retry_geval_checkpoint_503.py \
  --checkpoint-dir .deepeval/geval_judgment_checkpoints/viking-13b \
  --only-errors all,errno_89
```

**Retry without regenerating exports** (only JSONL on disk):

```bash
python scripts/retry_geval_checkpoint_503.py \
  --checkpoint-dir .deepeval/geval_judgment_checkpoints/norwai-mistral-7b \
  --no-export
```

After a successful retry, run the **audit** again on the same `--checkpoint-dir` to confirm failures are gone.

---

## HTML viewers (export summaries and prefix curves)

Run from repo root. Outputs are written under `.deepeval/geval_exports/images/` in these examples (create the folder if needed: `mkdir -p .deepeval/geval_exports/images`).

### 1. Win rates / Bradley–Terry viewer (all export leaves in one page)

```bash
python Other/view_geval_export_results.py \
  --all_export_leaves \
  -o .deepeval/geval_exports/images/geval_export_viewer_all_leaves.html
```

**Subset of models** (explicit leaf names under `.deepeval/geval_exports/`):

```bash
python Other/view_geval_export_results.py \
  --export_leaf viking-13b norwai-mistral-7b \
  -o .deepeval/geval_exports/images/geval_export_viewer_subset.html
```

### 2. Prefix (cumulative-doc) viewer — one HTML, dropdown to pick model

**Default** (one page; dropdown lists every discoverable export leaf under `.deepeval/geval_exports/`):

```bash
python Other/view_geval_prefix_interactive.py \
  -o .deepeval/geval_exports/images/geval_prefix_interactive.html
```

**Subset of models** (dropdown order follows your list):

```bash
python Other/view_geval_prefix_interactive.py \
  --export-leaves viking-13b norwai-mistral-7b \
  -o .deepeval/geval_exports/images/geval_prefix_subset.html
```

**Single model** (dropdown hidden):

```bash
python Other/view_geval_prefix_interactive.py \
  --export_leaf viking-13b \
  -o .deepeval/geval_exports/images/geval_prefix_interactive_viking-13b.html
```

Open the generated `.html` files in a normal browser.

---

## See also

- `run.txt` — older notes and extra examples (some paths may refer to legacy layouts; prefer the commands above for this repo).
- Docstrings in `pairwise_eval/__main__.py`, `pairwise_eval/config.py`, and the `scripts/*.py` files for edge cases.

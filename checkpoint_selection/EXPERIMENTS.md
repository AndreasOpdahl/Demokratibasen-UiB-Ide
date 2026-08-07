# Running pairwise G-Eval experiments

This guide is for running the pipeline on **new summarization checkpoints** (one eval folder per base model under `DATA_ROOT/eval/`), then auditing API failures, retrying failed judgments, and opening the HTML viewers.

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

3. **Eval data layout**: under `DATA_ROOT/eval/<your_model_name>/`, one `*.jsonl` per checkpoint (file stem = `model_id`). Each JSONL must have aligned rows across files (same `input_text` order); the loader assigns `doc_id` as `doc_1`, `doc_2`, … per row index. See `pairwise_eval.data.stack_eval_jsonl_checkpoints_long_df`.

   `DATA_ROOT` (see `pairwise_eval.config.DATA_ROOT`) is no longer inside the repo (moved 2026-06). It
   defaults to `$ONEDRIVE/Shared/Demokratibasen-UiB-Ide/EvaluationDatasets/CheckpointSelection/Data_202606`;
   override with the `CHECKPOINT_SELECTION_DATA_DIR` env var if your OneDrive root or the dataset
   snapshot name differs.

4. **G-Eval prompts**: for each dimension name in `EVAL_DIMENSIONS`, there must be a template file:

   `DATA_ROOT/prompts/geval/<dimension>.txt`

---

## What to edit in `pairwise_eval/config.py`

These values are read when you run `python -m pairwise_eval` (or `python pairwise_eval/__main__.py`). **Save the file before launching**; there is no CLI override for most of these.

### `EVAL_DIMENSIONS`

Tuple of dimension names (strings). Each name must match a prompt file `DATA_ROOT/prompts/geval/<name>.txt`.

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
| `EVAL_DATA_DIR` | `None` → auto `DATA_ROOT/eval`. Or set to an alternate eval root (path under `DATA_ROOT` if relative). |
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

- If `DATA_ROOT/eval/` (or `EVAL_DATA_DIR`) contains **`*.jsonl` directly**, one run writes to `.deepeval/<GEVAL_EXPORT_DIRNAME>/` (flat layout).
- If it contains **only subfolders** with `*.jsonl` (e.g. `DATA_ROOT/eval/viking-13b/`, `DATA_ROOT/eval/norwai-mistral-7b/`), the pipeline runs **once per subfolder** and writes:

  - `.deepeval/<GEVAL_EXPORT_DIRNAME>/<folder_name>/` — exports (JSON, tables, reports)  
  - `.deepeval/geval_judgment_checkpoints/<folder_name>/` — resumable JSONL checkpoints (when `GEVAL_CHECKPOINT_DIR` is set)

**Resume:** re-running the same model is safe: existing judgment keys in the JSONL files are skipped; only missing (judge × dimension × pair) work is issued.

---

## Where G-Eval results are stored

All G-Eval artifacts live under `.deepeval/` at the repository root. The exact subfolder names come from `pairwise_eval/config.py`:

| Setting | Default | Role |
|---------|---------|------|
| `GEVAL_EXPORT_DIRNAME` | `geval_exports` | Aggregated exports (JSON, CSV, reports) |
| `GEVAL_CHECKPOINT_DIR` | `.deepeval/geval_judgment_checkpoints` | Raw, resumable judgment logs |

**Layout:**

- **Flat eval** (`DATA_ROOT/eval/*.jsonl` directly): one export leaf at `.deepeval/geval_exports/` and one checkpoint folder at `.deepeval/geval_judgment_checkpoints/`.
- **Per-model eval** (`DATA_ROOT/eval/<folder_name>/*.jsonl`): one **leaf** per subfolder, e.g. for the four-model human-candidate study:

  ```text
  .deepeval/geval_exports/2500-human-cadidates/
  .deepeval/geval_judgment_checkpoints/2500-human-cadidates/
  ```

The leaf name matches the eval subfolder name (`2500-human-cadidates`, `viking-13b`, etc.).

### `geval_judgment_checkpoints/` — raw judgments (source of truth)

This is the **append-only, resumable** store written while the pipeline runs. Re-runs and retry scripts read and update these files; exports are rebuilt from them afterward.

**One JSONL file per judge × dimension**, named like:

```text
gpt-5-mini__relevance.jsonl
google__gemini-2.5-flash-preview-05-20__hygiene.jsonl
anthropic__claude-3-5-haiku-20241022__consistency.jsonl
mistral-medium-latest__newsworthiness.jsonl
```

Each line is one pairwise judgment:

| Field | Meaning |
|-------|---------|
| `key` | Stable id for this comparison (judge, dimension, `doc_id`, left/right model ids, hash of summary texts) |
| `choice_side` | `"left"` or `"right"` (or tie handling when applicable) |
| `chosen` | Model id of the winning summary |
| `rationale` | Judge's explanation, or `[api_error]` on failure |

**Use checkpoints when you need to:**

- Resume an interrupted run (`python -m pairwise_eval` skips keys already present)
- Audit or retry failed API calls (`scripts/audit_geval_api_failures.py`, `scripts/retry_geval_checkpoint_503.py`)
- Inspect individual judgment rationales

For the full 2500-document human-candidate run, expect **16 files** (4 judges × 4 dimensions) with **15,000 lines each** (2500 docs × 6 unordered model pairs per doc).

### `geval_exports/` — aggregated analysis artifacts

Exports are **derived** from checkpoints (and the eval JSONLs). They are regenerated at the end of each pipeline run or retry. Start here for summaries, tables, and downstream tooling.

Each export leaf has three subfolders plus a manifest:

```text
manifest.json
json/
tables/
reports/
```

**`manifest.json`** — index of generated files and row counts.

**`json/`**

| File | Content |
|------|---------|
| `pairs_table.json` | Sampled pairwise comparisons: one row per (document, left model, right model) with source text and both summaries. Reused on resume so new runs do not resample pairs. |
| `summarization_long.json` | Long-format stack of all document × model summaries from the eval JSONLs. |
| `geval__<judge>__<dimension>.json` | One table per judge × dimension: merges `pairs_table` rows with that judge's choices and rationales. |

**`tables/`** — CSV aggregates, e.g. `win_rates_by_model.csv`, `bradley_terry_theta__<dimension>.csv`, `bradley_terry_long.csv`.

**`reports/`** — human-readable summaries:

- `results_summary.md` — scope (judges, dimensions, document count), win-rate tables, Bradley–Terry θ by dimension
- `win_rates_by_model.md`, `win_rates_by_dimension.md`, `bradley_terry_theta_by_dimension.md`
- `win_rates_by_model.tex` — LaTeX table for papers

**Downstream use:** `human_eval_batching/build_batches.py` reads G-Eval exports (under `.deepeval/geval_exports/2500-human-cadidates/`) to rank documents and pairs for human annotation. The HTML viewers (`Other/view_geval_export_results.py`, `Other/view_geval_prefix_interactive.py`) also read from `geval_exports/`.

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

## Human evaluation batches (Label Studio)

Human annotation batches are built with `human_eval_batching/build_batches.py`. The design is 24 batches × 24 blocks × 3 pairwise comparisons = **576 documents** total, released in three non-overlapping Label Studio projects.

### Label Studio labeling interface

Copy the labeling configuration from:

```text
human_eval_batching/labelstudio_block_interface.xml
```

Paste it into **Settings → Labeling Interface** when creating each Label Studio project.

### Generate the three batch sets

Run from the repository root. Project 1 can be generated as soon as you are ready to start annotation; projects 2 and 3 should be generated after the full 2500-document G-Eval run is complete, so later batches use the full ranking.

**Project 1 — batches 1–8 (192 documents):**

```powershell
python human_eval_batching/build_batches.py --batches 8 --output-dir human_eval_batching/frozen_projects/project_01_batches_01_08
```

**Project 2 — batches 9–20 (288 documents):**

```powershell
python human_eval_batching/build_batches.py --batches 12 --start-batch-number 9 --exclude-documents-from human_eval_batching/frozen_projects/project_01_batches_01_08 --output-dir human_eval_batching/frozen_projects/project_02_batches_09_20
```

**Project 3 — batches 21–24 (96 documents):**

```powershell
python human_eval_batching/build_batches.py --batches 4 --start-batch-number 21 --exclude-documents-from human_eval_batching/frozen_projects/project_01_batches_01_08 --exclude-documents-from human_eval_batching/frozen_projects/project_02_batches_09_20 --output-dir human_eval_batching/frozen_projects/project_03_batches_21_24
```

`--exclude-documents-from` keeps documents from appearing in more than one project.

### Import tasks into Label Studio

For each project, upload **`labelstudio_tasks.json`** from that project's frozen output folder:

| Project | Import file |
|---------|-------------|
| Batches 1–8 | `human_eval_batching/frozen_projects/project_01_batches_01_08/labelstudio_tasks.json` |
| Batches 9–20 | `human_eval_batching/frozen_projects/project_02_batches_09_20/labelstudio_tasks.json` |
| Batches 21–24 | `human_eval_batching/frozen_projects/project_03_batches_21_24/labelstudio_tasks.json` |

Use **`labelstudio_tasks.json`**, not `labelstudio_tasks_by_block.json`. The per-annotator file randomizes block order separately for each annotator (A–F).

More detail on outputs, selection criteria, and annotator assignments: `human_eval_batching/README.md`.

---

## See also

- `run.txt` — older notes and extra examples (some paths may refer to legacy layouts; prefer the commands above for this repo).
- Docstrings in `pairwise_eval/__main__.py`, `pairwise_eval/config.py`, and the `scripts/*.py` files for edge cases.

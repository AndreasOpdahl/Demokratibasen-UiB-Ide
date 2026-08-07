# Human Evaluation Batching

Standalone tooling for creating human annotation batches from the
`DATA_ROOT/eval/2500-human-cadidates` four-candidate evaluation set, where
`DATA_ROOT` defaults to
`$ONEDRIVE/Shared/Demokratibasen-UiB-Ide/EvaluationDatasets/CheckpointSelection/Data_202606`
(override with `CHECKPOINT_SELECTION_DATA_DIR`; see `build_batches.py`).

This directory is intentionally separate from the existing pairwise evaluation
and human annotation code. Regenerating these files should not modify the
current pipeline.

## Generate

From the repository root:

```powershell
python human_eval_batching/build_batches.py
```

By default this creates 576 document blocks:

- 24 batches
- 24 blocks per batch
- 3 summary comparisons per block
- 3 annotators per batch
- 6 annotators total: `A` through `F`

The first 8 batches are the 192-document pilot subset. All 24 batches are the
576-document expanded design.

Useful options:

```powershell
python human_eval_batching/build_batches.py --batches 8
python human_eval_batching/build_batches.py --batches 24 --seed 42
python human_eval_batching/build_batches.py --output-dir human_eval_batching/outputs_alt
```

The script builds a stable 24-batch master design by default, then writes only
the requested prefix. This means a pilot run with `--batches 8` has the same
batches 1-8 as a later run with `--batches 24`; the later run only adds batches
9-24. Documents are not reused across blocks or batches within one generated
output. The selected documents are also stratified by selection bucket before
batching, so each batch has a similar mix of criteria.

`--documents` is still accepted as a direct override, but `--batches` is the
preferred control because the design is organized around 24-block batches.

## Incremental Label Studio Projects

To start annotation before the full G-Eval run is finished, generate and freeze
the first project:

```powershell
python human_eval_batching/build_batches.py --batches 8 --output-dir human_eval_batching/frozen_projects/project_01_batches_01_08
```

Later, generate additional projects while excluding documents already used in
previous projects:

```powershell
python human_eval_batching/build_batches.py --batches 12 --start-batch-number 9 --exclude-documents-from human_eval_batching/frozen_projects/project_01_batches_01_08 --output-dir human_eval_batching/frozen_projects/project_02_batches_09_20

python human_eval_batching/build_batches.py --batches 4 --start-batch-number 21 --exclude-documents-from human_eval_batching/frozen_projects/project_01_batches_01_08 --exclude-documents-from human_eval_batching/frozen_projects/project_02_batches_09_20 --output-dir human_eval_batching/frozen_projects/project_03_batches_21_24
```

The `--exclude-documents-from` option accepts either a previous output directory
or a `documents.jsonl` file. This keeps the three Label Studio projects
non-overlapping while preserving batch numbering.

## Outputs

All generated files are written to `human_eval_batching/outputs/`.

### `documents.jsonl`

One row per selected document. Contains stable document ids, source metadata,
selection bucket, document-level LLM-selection features, and source text.

### `summaries.jsonl`

One row per document-summary candidate.

The four model ids are:

- `GPT4o-mini`
- `gemma-2-9b__checkpoint-2500-gen0-inputs-refs-preds-2500-examples`
- `viking-13b__checkpoint-3500-gen0-inputs-refs-preds-2500-examples`
- `gpt-4o-mini-elaborate__inputs-refs-preds-2500-examples`

## Pair-Level Selection

The script ranks individual document/model-pair comparisons before selecting
documents. For every pair `p = (d, m_i, m_j)`, it computes criterion scores from
the available G-Eval judgments:

- `low_agreement`: high vote entropy over judge decisions
- `high_agreement`: high top-choice share and low tie rate
- `gpt4o_challenged`: GPT4o-mini loses when it appears in the pair
- `elaborate_interesting`: the elaborate/newsworthiness-prompt summary wins

Scores are converted to rank scores across all available pairwise comparisons.
A pair is considered high-ranked for a criterion when it is in the top ranked
portion controlled by `PAIR_HIGH_RANK_CUTOFF` in `build_batches.py`.

Document priority is derived from the high-ranked pairs it contains: documents
with multiple high-ranked pairs receive a higher priority. After a document is
selected, its three block pairs are chosen to include as many of the
high-ranked pairs as possible, then balance constraints are used as tie-breakers.

This follows the idea that the document is selected because of specific
interesting pairwise comparisons, and those comparisons should be shown to the
human annotators.

### `blocks.jsonl`

One row per document block. Each block has one document and three summary
pairs. Pair construction enforces:

- every summary appears at least once in the block
- no summary appears more than twice in the block
- high-ranked pairwise comparisons for the selected document are included when
  feasible
- global model-pair exposure is used as a tie-breaker after high-ranked-pair
  inclusion
- global left/right exposure is balanced greedily

### `batches.jsonl`

One row per batch. Each batch contains 24 block ids.

### `assignments.jsonl`

One row per annotator-batch assignment. Each batch is assigned to three
annotators. Block order is randomized separately per annotator.

### `labelstudio_tasks.json`

One task per annotator-block assignment. Each task includes the source document
and three left/right summary comparisons. Model ids and summary ids are included
as metadata for analysis; do not expose those fields in the labeling UI if the
task should remain blinded.

### `labelstudio_tasks_by_block.json`

One task per block. Use this combined file when importing all generated batches
into one Label Studio project.

### `labelstudio_tasks_by_batch/`

One Label Studio import file per batch, using the same one-task-per-block format
as `labelstudio_tasks_by_block.json`. Use these files when creating separate
Label Studio projects per batch.

### `manifest.json`

Run metadata and balance checks:

- model pair counts
- left-position counts
- selection bucket counts
- annotator assignment counts
- annotator-pair co-assignment counts

## Notes

The old pair-level selection criteria are still computed at the pair level.
They are then lifted to the document level for block selection. The block
builder preserves the selected high-ranked pairs whenever this is compatible
with the block constraints.

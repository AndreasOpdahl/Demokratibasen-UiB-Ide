# Human annotation — LLM judge validation

Independent annotation files per dimension to validate that LLM judges are acceptable
proxies for human judgment. **Not** for picking the best model.

## Build the four files

```bash
python human_annotation/build_dataset.py -n 25
```

Creates `human_annotation/outputs/winners/`:

| File | Content |
|------|---------|
| `relevance.csv` / `.json` | 25 pairs, relevance only |
| `consistency.csv` / `.json` | 25 pairs, consistency only |
| `newsworthiness.csv` / `.json` | 25 pairs, newsworthiness only |
| `hygiene.csv` / `.json` | 25 pairs, hygiene only |
| `all.csv` / `.json` | All 100 items concatenated (4 dimensions) |
| `*_ls.csv` / `all_ls.csv` | Label Studio subset: `dimension`, `dimension_color`, `annotation_prompt`, `source_text`, `summary_left`, `summary_right` |
| `selection_metadata.json` | Selection stats + pair overlap across dims |

Each row has `item_id` like `rel-01`, `con-01`, … and a `dimension_color` hex code for UI tooling.

The four sets are **independent** — different pairs per dimension is expected and fine.

## CSV columns

Each CSV includes LLM judgments alongside the annotation content so you can compare directly:

| Column | Description |
|--------|-------------|
| `reference_summary` | Reference summary shown to LLM judges |
| `annotation_prompt` | G-Eval rubric for this dimension (no document/summary slots or JSON output spec) |
| `human_choice` | Empty until annotated — fill with `left`, `right`, or `tie` |
| `llm_majority` | Panel majority on this dimension |
| `llm_votes_left/right/tie` | Vote counts across the 4 LLM judges |
| `llm_vote_entropy` | Disagreement among judges (higher = more split) |
| `llm_vote__*` | Each judge's vote (`left` / `right` / `tie`) |

Rationales are in the JSON files only (`llm_judges`), not in the CSV.

## Selection buckets (per dimension)

- **low_agreement** (40%): LLM judges split on this dimension
- **tie_majority** (10%): at least half of judges voted `tie`
- **high_agreement** (15%): unanimous `left` or `right` (calibration)
- **reference_challenged** (20%): GPT4o-mini loses on this dimension
- **representative** (15%): random fill from remaining pool

## Data sources

- Judgments: `.deepeval/geval_judgment_checkpoints/winners/`
- Text: `.deepeval/geval_exports/winners/json/`

## After annotation

Fill `human_choice` with `left`, `right`, or `tie` in each CSV, then:

```bash
python human_annotation/analyze_agreement.py human_annotation/outputs/winners/
```

Prints accuracy and confusion matrices per judge per dimension file.

## Explore

Open `human_annotation/explore_selection.ipynb`.

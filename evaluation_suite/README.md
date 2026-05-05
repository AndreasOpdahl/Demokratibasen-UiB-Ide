# Evaluation Suite

This directory provides a cleaner, modular layout for evaluation workflows,
without modifying legacy scripts under `Other/` or `human evaluation/`.

## Layout

- `core/`
  - shared logic (e.g., weighted checkpoint mean computation from G-Eval JSON)
- `apps/`
  - user-facing CLI entry points
- `outputs/`
  - recommended location for generated artifacts

## Quick start

- Interactive prefix visualization (legacy-compatible entry point):
  - `python3 evaluation_suite/apps/view_geval_prefix_interactive.py --help`
- Human pair selection (legacy-compatible entry point):
  - `python3 evaluation_suite/apps/select_human_eval_pairs.py --help`
- Compute weighted top checkpoints directly:
  - `python3 evaluation_suite/apps/compute_top_checkpoints.py --help`


#!/usr/bin/env python3
"""Print a checkpoint's predictions labeled GOOD/BAD by hygiene outcome.

Given an ``all_eval_results`` folder, a checkpoint number, and an optional
generation (default 0), this reads:

  * ``checkpoint-<n>-gen<gen>-hygiene-bad-<suffix>.jsonl`` — line-aligned with
    the predictions; a non-empty line marks an example that FAILED hygiene.
  * ``checkpoint-<n>-gen<gen>-inputs-refs-preds-<suffix>.jsonl`` — holds the
    actual prediction text (the bad file stores ``prediction: null``).

Every prediction is written to stdout prefixed with ``\\n\\nGOOD:\\n`` or
``\\n\\nBAD:\\n``. Diagnostics go to stderr so stdout stays clean for piping.

Example:
    python scripts/show_hygiene_labeled_predictions.py \\
        shortlist/viking-13b-apptainer-fsdp/all_eval_results 3500 --gen 0
"""

import argparse
import glob
import json
import os
import sys


def _eprint(*args):
    print(*args, file=sys.stderr)


def _resolve_suffix(folder, checkpoint, gen, examples_suffix):
    """Determine the examples suffix (e.g. '2500-examples') for the bad file."""
    if examples_suffix:
        return examples_suffix
    pattern = os.path.join(
        folder, f"checkpoint-{checkpoint}-gen{gen}-hygiene-bad-*-examples.jsonl"
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        _eprint(f"ERROR: no hygiene-bad file found matching: {pattern}")
        sys.exit(1)
    if len(matches) > 1:
        _eprint(
            "ERROR: multiple hygiene-bad files match; pass --examples_suffix to choose one:"
        )
        for m in matches:
            base = os.path.basename(m)
            # extract the '<N>-examples' part
            tail = base.split("-hygiene-bad-", 1)[1]
            _eprint(f"  - {tail[:-len('.jsonl')]}  ({base})")
        sys.exit(1)
    base = os.path.basename(matches[0])
    return base.split("-hygiene-bad-", 1)[1][: -len(".jsonl")]


def _read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def main():
    parser = argparse.ArgumentParser(
        description="Print a checkpoint's predictions labeled GOOD/BAD by hygiene outcome.",
    )
    parser.add_argument("folder", help="Path to the all_eval_results folder.")
    parser.add_argument("checkpoint", type=int, help="Checkpoint step number (e.g. 3500).")
    parser.add_argument(
        "--gen", type=int, default=0, help="Generation to inspect (default: 0)."
    )
    parser.add_argument(
        "--examples_suffix",
        default=None,
        help="Examples suffix (e.g. '2500-examples'). Auto-detected when omitted.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        _eprint(f"ERROR: folder does not exist: {args.folder}")
        sys.exit(1)
    if args.gen < 0:
        _eprint("ERROR: --gen must be non-negative.")
        sys.exit(1)

    suffix = _resolve_suffix(args.folder, args.checkpoint, args.gen, args.examples_suffix)

    bad_path = os.path.join(
        args.folder, f"checkpoint-{args.checkpoint}-gen{args.gen}-hygiene-bad-{suffix}.jsonl"
    )
    irp_path = os.path.join(
        args.folder,
        f"checkpoint-{args.checkpoint}-gen{args.gen}-inputs-refs-preds-{suffix}.jsonl",
    )

    if not os.path.exists(bad_path):
        _eprint(f"ERROR: hygiene-bad file not found: {bad_path}")
        sys.exit(1)
    if not os.path.exists(irp_path):
        _eprint(f"ERROR: inputs-refs-preds file not found (needed for prediction text): {irp_path}")
        sys.exit(1)

    bad_lines = _read_lines(bad_path)
    irp_lines = _read_lines(irp_path)

    if len(bad_lines) != len(irp_lines):
        _eprint(
            f"WARNING: line count mismatch (bad={len(bad_lines)}, "
            f"inputs-refs-preds={len(irp_lines)}); aligning to the shorter file."
        )
    n = min(len(bad_lines), len(irp_lines))

    good_count = 0
    bad_count = 0
    for i in range(n):
        irp_raw = irp_lines[i].strip()
        if not irp_raw:
            continue  # blank source line (skipped during hygiene)
        try:
            pred = json.loads(irp_raw).get("prediction") or ""
        except json.JSONDecodeError:
            _eprint(f"WARNING: could not parse inputs-refs-preds line {i + 1}; skipping.")
            continue

        is_bad = bool(bad_lines[i].strip())  # non-empty bad line => failed hygiene
        if is_bad:
            bad_count += 1
            sys.stdout.write(f"\n\nBAD:\n{pred}")
        else:
            good_count += 1
            sys.stdout.write(f"\n\nGOOD:\n{pred}")

    sys.stdout.write("\n")
    _eprint(
        f"\n[summary] checkpoint-{args.checkpoint} gen{args.gen} {suffix}: "
        f"{good_count} GOOD, {bad_count} BAD, {good_count + bad_count} total."
    )


if __name__ == "__main__":
    main()

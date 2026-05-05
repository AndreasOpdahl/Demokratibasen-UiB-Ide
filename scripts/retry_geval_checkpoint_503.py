#!/usr/bin/env python3
"""Re-run G-Eval LLM calls for checkpoint lines whose rationale is a retryable ``[api_error]``.

By default retries **all common** ``[api_error]`` buckets: HTTP **503**, **429**, **400**
(including **http_400_context_window**), **200**, **500**, **502**, plus **timeout**,
**connection_error**, and **json_parse**.

Narrow with ``--only-errors`` (alias: ``--errors``, ``--categories``): comma-separated list,
e.g. ``503,429`` or ``http_400_context_window`` or ``timeout``. Use ``all`` for the same full
default set. You can also pass exact audit labels like ``exception_RuntimeError`` or ``msg_...``.

Rewrites matching lines **in place** in the same ``*.jsonl`` files (same ``key``, new judgment).
After any successful line updates, by default it **rebuilds G-Eval tables from checkpoints**
(using the mock judge only for missing keys — there should be none) and runs
``export_full_run`` so ``.deepeval/<GEVAL_EXPORT_DIRNAME>/`` reports (win rates, Bradley–Terry,
``results_summary.md``, etc.) match the retried judgments. Use ``--no-export`` to skip that step.

**Important:** This reloads eval data and rebuilds pair rows using the same settings as
``pairwise_eval.config`` (``MAX_DOCUMENTS``, ``N_PAIRS_PER_DOCUMENT``, ``DEFAULT_PAIR_SEED``,
``resolve_eval_data_dir``, etc.). If those differ from the run that produced the checkpoints,
stable keys may not match and rows will be skipped.

**Pairs table:** The main CLI calls ``build_pairs_table`` with ``prior_pairs_df`` when
``EXTEND_PAIRS_TABLE_JSON`` is set or when ``<export_leaf>/json/pairs_table.json`` exists.
This script loads the **same** prior table for each judgment leaf so stable ``key`` hashes
(sumleft/sumright) match; without that, retries often fail with "no matching pair row for key"
even when eval settings are correct.

When the default checkpoint root contains **only subfolders** with ``*.jsonl`` (same layout as
``python -m pairwise_eval`` per model), this script scans **every** such leaf folder, retries
lines in each, reloads ``Data/eval/<folder_name>/`` for matching keys, and refreshes exports
under ``.deepeval/<GEVAL_EXPORT_DIRNAME>/<folder_name>/`` for each touched model. You can still
point ``--checkpoint-dir`` at a single model folder to scope work to that run.

Usage::

    python scripts/retry_geval_checkpoint_503.py
    python scripts/retry_geval_checkpoint_503.py --checkpoint-dir .deepeval/geval_judgment_checkpoints/gemma-2b
    python scripts/retry_geval_checkpoint_503.py --checkpoint-dir .deepeval/other_checkpoints
    python scripts/retry_geval_checkpoint_503.py --dry-run
    python scripts/retry_geval_checkpoint_503.py --no-export
    python scripts/retry_geval_checkpoint_503.py --only-errors 503,429
    python scripts/retry_geval_checkpoint_503.py --errors http_400_context_window
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, FrozenSet

import numpy as np
import pandas as pd

# Repo root on path for ``pairwise_eval`` imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pairwise_eval import build_geval_tables, export_full_run  # noqa: E402
from pairwise_eval.config import (  # noqa: E402
    DEFAULT_GEVAL_BASE_SEED,
    DEFAULT_PAIR_SEED,
    EVAL_DIMENSIONS,
    EXTEND_PAIRS_TABLE_JSON,
    GEVAL_CHECKPOINT_DIR,
    GEVAL_EXPORT_DIRNAME,
    JUDGES,
    MAX_DOCUMENTS,
    N_PAIRS_PER_DOCUMENT,
    REFERENCE_SUMMARY_MODEL_ID,
    REPO_ROOT,
)
from pairwise_eval.data import (  # noqa: E402
    load_eval_jsonl_long_df,
    long_df_head_documents,
    resolve_eval_data_dir,
    resolve_eval_dir_for_judgment_leaf,
    resolve_export_dir_for_judgment_leaf,
)
from pairwise_eval.geval_checkpoint import (  # noqa: E402
    _judgment_jsonable,
    discover_checkpoint_leaf_dirs,
    judgment_stable_key,
)
from pairwise_eval.io_export import resolve_geval_export_dir  # noqa: E402
from pairwise_eval.judging import (  # noqa: E402
    attach_doc_context,
    mock_evaluate_pair,
    rng_for_judge_dimension,
)
from pairwise_eval.llm_judge import make_local_llm_evaluate_fn  # noqa: E402
from pairwise_eval.pairs import build_pairs_table, load_pairs_table_json  # noqa: E402


def _prior_pairs_df_for_judgment_leaf(leaf: Path) -> pd.DataFrame | None:
    """Same prior as ``pairwise_eval.__main__._extend_pairs_json_for_export_dir`` for this leaf."""
    if EXTEND_PAIRS_TABLE_JSON is not None:
        p = Path(EXTEND_PAIRS_TABLE_JSON)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.is_file():
            return load_pairs_table_json(p)
        return None
    export_dir = resolve_export_dir_for_judgment_leaf(leaf)
    auto = export_dir / "json" / "pairs_table.json"
    if auto.is_file():
        return load_pairs_table_json(auto)
    return None


def _default_checkpoint_dir() -> Path:
    if GEVAL_CHECKPOINT_DIR is not None:
        return Path(GEVAL_CHECKPOINT_DIR)
    return REPO_ROOT / ".deepeval" / "geval_judgment_checkpoints"


def _leaf_is_per_model_judgment_subfolder(leaf: Path) -> bool:
    """True if ``leaf`` is a direct child of the configured G-Eval checkpoint root (per-model layout)."""
    if GEVAL_CHECKPOINT_DIR is None:
        return False
    try:
        return leaf.parent.resolve() == Path(GEVAL_CHECKPOINT_DIR).resolve()
    except OSError:
        return False


# Default: common HTTP failures + transport/parsing issues (see :func:`_categorize_api_error`).
_HTTP_TRANSIENT: FrozenSet[str] = frozenset(
    {
        "http_503",
        "http_429",
        "http_400",
        "http_400_context_window",
        "http_200",
        "http_500",
        "http_502",
    }
)
_NAMED_NON_HTTP: FrozenSet[str] = frozenset({"timeout", "connection_error", "json_parse"})
DEFAULT_RETRY_CATEGORIES: FrozenSet[str] = frozenset(_HTTP_TRANSIENT | {"timeout", "connection_error", "json_parse"})


def _parse_retry_categories(arg: str | None) -> FrozenSet[str]:
    """Parse ``--only-errors`` / ``--categories`` (comma-separated tokens)."""
    if arg is None or not str(arg).strip():
        return DEFAULT_RETRY_CATEGORIES
    raw = str(arg).strip()
    if raw.lower() in ("all", "*"):
        return DEFAULT_RETRY_CATEGORIES

    out: set[str] = set()
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        pl = p.lower()
        if pl in ("all", "*"):
            out |= DEFAULT_RETRY_CATEGORIES
            continue
        if pl in _NAMED_NON_HTTP:
            out.add(pl)
            continue
        if pl.startswith("http_") and len(pl) >= 8:
            out.add(pl)
            continue
        if pl.isdigit() and len(pl) == 3:
            out.add(f"http_{pl}")
            continue
        low = p.lower()
        if low.startswith("exception_") or low.startswith("errno_") or low.startswith("msg_"):
            out.add(p)
            continue
        raise ValueError(
            f"Invalid error token {part!r}. Examples: 503,429, http_400_context_window, timeout, "
            "connection_error, json_parse, all, or an exact label like exception_RuntimeError"
        )
    if not out:
        raise ValueError("--only-errors resulted in an empty set")
    return frozenset(out)


def _extract_http_status_code(body: str) -> str | None:
    patterns = [
        r"\bHTTP\s*/?\s*1\.[01]\s+(\d{3})\b",
        r"\bHTTP\s+(\d{3})\b",
        r"\b(\d{3})\s+Server Error\b",
        r"\b(\d{3})\s+Client Error\b",
        r"\b(\d{3})\s+Unknown Error\b",
        r'(?:"status"|status)\s*[:=]\s*"?(\d{3})\b',
        r'"code"\s*:\s*(\d{3})\b',
        r"'code'\s*:\s*(\d{3})\b",
        r"\bstatusCode[\"']?\s*[:=]\s*(\d{3})\b",
    ]
    for pat in patterns:
        m = re.search(pat, body, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _exception_label(body: str) -> str | None:
    m = re.search(r"\b([A-Za-z_][A-Za-z0-9_.]*Error)\s*:", body)
    if m:
        return f"exception_{m.group(1)}"
    return None


def _errno_label(body: str) -> str | None:
    m = re.search(r"\[Errno\s+(-?\d+)\]", body, re.IGNORECASE)
    if m:
        return f"errno_{m.group(1)}"
    return None


def _categorize_api_error(rationale: str) -> str:
    if "[api_error]" not in rationale:
        return "not_api_error"
    body = rationale.split("[api_error]", 1)[-1].strip()
    bl = body.lower()

    code = _extract_http_status_code(body)

    if code == "400" and (
        "context" in bl
        or "maximum context" in bl
        or ("token" in bl and ("maximum" in bl or "limit" in bl or "long" in bl))
        or "too many tokens" in bl
    ):
        return "http_400_context_window"
    if code is not None:
        return f"http_{code}"

    if "timeout" in bl or "timed out" in bl:
        return "timeout"
    if "connection" in bl or "econnrefused" in bl:
        return "connection_error"
    if "json" in bl and ("decode" in bl or "parse" in bl or "not json" in bl):
        return "json_parse"

    el = _exception_label(body)
    if el:
        return el
    en = _errno_label(body)
    if en:
        return en

    slug = re.sub(r"[^\w]+", "_", body[:72].strip())[:48].strip("_").lower()
    if slug:
        return f"msg_{slug}"
    return "msg_empty"


def _scan_retryable_lines(
    checkpoint_dir: Path,
    categories: FrozenSet[str],
) -> tuple[dict[Path, list[tuple[int, str, dict[str, Any]]]], Counter[str]]:
    """Map each JSONL path to retryable lines; second value is category → line count."""
    by_path: dict[Path, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    cat_counts: Counter[str] = Counter()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {checkpoint_dir}")

    for path in sorted(checkpoint_dir.glob("*.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rat = rec.get("rationale")
            if not isinstance(rat, str):
                continue
            cat = _categorize_api_error(rat)
            if cat not in categories:
                continue
            k = rec.get("key")
            if not isinstance(k, str):
                continue
            cat_counts[cat] += 1
            by_path[path].append((i, k, rec))
    return dict(by_path), cat_counts


def _load_eval_pairs_and_context(
    *,
    eval_dir: Path | None = None,
    pair_seed: int,
    max_documents: int | None,
    n_pairs_per_document: int,
    judgment_leaf: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(pairs_df, long_df, ctx)`` with ``ctx = attach_doc_context(pairs, long_df)``.

    When ``judgment_leaf`` is set, loads the same ``prior_pairs_df`` as the main pipeline
    (``EXTEND_PAIRS_TABLE_JSON`` or ``<export>/json/pairs_table.json``) so checkpoint ``key``
    rows still match.
    """
    root = eval_dir if eval_dir is not None else resolve_eval_data_dir()
    long_df = load_eval_jsonl_long_df(root)
    long_df = long_df_head_documents(long_df, max_documents)
    rng = np.random.default_rng(pair_seed)
    prior: pd.DataFrame | None = None
    if judgment_leaf is not None:
        prior = _prior_pairs_df_for_judgment_leaf(judgment_leaf)
        if prior is not None:
            src = (
                str(EXTEND_PAIRS_TABLE_JSON)
                if EXTEND_PAIRS_TABLE_JSON is not None
                else str(resolve_export_dir_for_judgment_leaf(judgment_leaf) / "json" / "pairs_table.json")
            )
            print(f"  prior_pairs_df: {len(prior)} row(s) from {src}", flush=True)
    pairs = build_pairs_table(
        long_df, n_pairs=n_pairs_per_document, rng=rng, prior_pairs_df=prior
    )
    ctx = attach_doc_context(pairs, long_df)
    return pairs, long_df, ctx


def _refresh_exports(
    *,
    pairs_df: pd.DataFrame,
    long_df: pd.DataFrame,
    checkpoint_dir: Path,
    export_dir: Path | None,
) -> Path:
    """Reload judgments from ``checkpoint_dir`` and write full export tree (same as main pipeline)."""
    models = sorted(long_df["model_id"].unique())
    boot = (export_dir / "json") if export_dir is not None else None
    geval = build_geval_tables(
        pairs_df,
        long_df,
        dimensions=EVAL_DIMENSIONS,
        judges=JUDGES,
        evaluate_fn=mock_evaluate_pair,
        checkpoint_dir=checkpoint_dir,
        checkpoint_bootstrap_json_dir=boot,
    )
    out = export_dir if export_dir is not None else resolve_geval_export_dir()
    return export_full_run(
        geval,
        pairs_df,
        long_df,
        models,
        ref_model=REFERENCE_SUMMARY_MODEL_ID,
        export_dir=out,
    )


def _find_row_for_key(
    ctx: pd.DataFrame, judge_id: str, dimension: str, target_key: str
) -> pd.Series | None:
    for _, row in ctx.iterrows():
        if judgment_stable_key(judge_id, dimension, row) == target_key:
            return row
    return None


def _rewrite_file(
    path: Path,
    key_to_judgment: dict[str, dict[str, object]],
    *,
    dry_run: bool,
) -> int:
    """Replace lines whose ``key`` is in ``key_to_judgment``. Returns number of lines replaced."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    replaced = 0
    new_lines: list[str] = []
    for line in lines:
        raw = line.strip()
        if not raw:
            new_lines.append(line)
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            new_lines.append(line)
            continue
        k = rec.get("key")
        if isinstance(k, str) and k in key_to_judgment:
            payload = {"key": k, **_judgment_jsonable(key_to_judgment[k])}
            new_lines.append(json.dumps(payload, ensure_ascii=False))
            replaced += 1
        else:
            new_lines.append(line)
    out = "\n".join(new_lines)
    if text.endswith("\n"):
        out += "\n"
    if not dry_run and replaced:
        path.write_text(out, encoding="utf-8")
    return replaced


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Retry LLM G-Eval for checkpoint lines with selected [api_error] categories; "
            "update JSONL in place. Default categories: common HTTP errors + timeout / "
            "connection / json_parse (use --only-errors to narrow)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --dry-run\n"
            "  %(prog)s --only-errors 503,429\n"
            "  %(prog)s --errors http_400_context_window\n"
            "  %(prog)s --categories all\n"
            "  %(prog)s --only-errors timeout,connection_error\n"
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory with *.jsonl checkpoints (default: GEVAL_CHECKPOINT_DIR)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List actions only; do not call APIs or write files",
    )
    parser.add_argument(
        "--pair-seed",
        type=int,
        default=DEFAULT_PAIR_SEED,
        help=f"RNG seed for pair sampling (default: {DEFAULT_PAIR_SEED}, must match original run)",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Override MAX_DOCUMENTS from config (default: use config value)",
    )
    parser.add_argument(
        "--n-pairs-per-document",
        type=int,
        default=None,
        help=f"Override N_PAIRS_PER_DOCUMENT (default: config = {N_PAIRS_PER_DOCUMENT})",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Do not regenerate .deepeval/<GEVAL_EXPORT_DIRNAME>/ after updating checkpoints",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help=(
            "Force this export root for every refreshed model (overwrites same tree each time). "
            "Omit for per-model exports under <repo>/.deepeval/<GEVAL_EXPORT_DIRNAME>/<model>/ "
            "when judgment folders match Data/eval/<model>/."
        ),
    )
    parser.add_argument(
        "-e",
        "--only-errors",
        "--errors",
        "--categories",
        dest="retry_errors",
        metavar="LIST",
        type=str,
        default=None,
        help=(
            "Comma-separated error buckets to retry (subset). Numeric HTTP codes (503,429), "
            "full names (http_400_context_window, timeout, connection_error, json_parse), "
            "keyword 'all' for the full default set, or exact audit labels (exception_*, errno_*, "
            "msg_*). "
            "Default when omitted: "
            + ",".join(sorted(DEFAULT_RETRY_CATEGORIES))
        ),
    )
    args = parser.parse_args()

    try:
        retry_cats = _parse_retry_categories(args.retry_errors)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2

    ck = args.checkpoint_dir
    if ck is None:
        ck = _default_checkpoint_dir()
    else:
        ck = ck.expanduser().resolve()

    max_docs = MAX_DOCUMENTS if args.max_documents is None else args.max_documents
    n_pairs = N_PAIRS_PER_DOCUMENT if args.n_pairs_per_document is None else args.n_pairs_per_document

    if not ck.is_dir():
        print(f"Not a directory: {ck}", file=sys.stderr)
        return 1

    leaves = discover_checkpoint_leaf_dirs(ck)

    if not leaves:
        print(
            f"No *.jsonl under {ck.resolve()} (neither top-level nor immediate subfolders).",
            file=sys.stderr,
        )
        return 1

    by_path: dict[Path, list[tuple[int, str, dict[str, Any]]]] = {}
    cat_counts: Counter[str] = Counter()
    for leaf in leaves:
        try:
            bp, cc = _scan_retryable_lines(leaf, retry_cats)
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            return 1
        for p, items in bp.items():
            by_path[p] = items
        cat_counts.update(cc)

    total_lines = sum(len(v) for v in by_path.values())
    if total_lines == 0:
        scanned = ", ".join(leaf.name for leaf in leaves)
        print(
            f"No matching [api_error] lines for categories {sorted(retry_cats)} "
            f"(scanned {len(leaves)} leaf folder(s): {scanned})"
        )
        return 0

    print(f"Checkpoint root: {ck.resolve()}")
    print(f"Leaf folders: {', '.join(leaf.name for leaf in leaves)}")
    print(f"Retry categories: {sorted(retry_cats)}")
    print(f"Lines to retry: {total_lines} across {len(by_path)} file(s)")
    print("By category:")
    for c in sorted(cat_counts.keys()):
        print(f"  {c}: {cat_counts[c]}")
    if args.dry_run:
        for p, items in sorted(by_path.items()):
            print(f"  {p.parent.name}/{p.name}: {len(items)} line(s)")
        print("Dry run: no API calls, no writes.")
        return 0

    if args.export_dir is not None and len(leaves) > 1:
        print(
            "[warn] --export-dir is set with multiple checkpoint leaf folders: each model refresh "
            "writes to the same path — use a single --checkpoint-dir or omit --export-dir.",
            file=sys.stderr,
        )

    by_leaf_files: dict[Path, dict[Path, list[tuple[int, str, dict[str, Any]]]]] = defaultdict(dict)
    for path, items in by_path.items():
        by_leaf_files[path.parent][path] = items

    evaluate_fn = make_local_llm_evaluate_fn(verbose=True)
    key_to_judgment: dict[str, dict[str, object]] = {}
    skipped: list[str] = []

    for leaf in sorted(by_leaf_files.keys()):
        paths_dict = by_leaf_files[leaf]
        eval_dir = resolve_eval_dir_for_judgment_leaf(leaf)
        if _leaf_is_per_model_judgment_subfolder(leaf) and eval_dir.name != leaf.name:
            print(
                f"[warn] Judgment folder {leaf.name!r} but eval data loaded from {eval_dir} "
                f"(expected ``Data/eval/{leaf.name}/`` with at least one *.jsonl). "
                "Stable keys will often not match — align folder names or set EVAL_DATA_DIR.",
                file=sys.stderr,
            )
        _, _, ctx = _load_eval_pairs_and_context(
            eval_dir=eval_dir,
            pair_seed=args.pair_seed,
            max_documents=max_docs,
            n_pairs_per_document=n_pairs,
            judgment_leaf=leaf,
        )
        print(
            f"Context [{leaf.name}]: eval={eval_dir} subset_rows={len(ctx)} "
            f"(pair_seed={args.pair_seed}, max_documents={max_docs!r}, n_pairs={n_pairs})",
            flush=True,
        )

        unique_keys: dict[str, None] = {}
        for items in paths_dict.values():
            for _, k, _ in items:
                unique_keys[k] = None
        keys_in_order = list(unique_keys.keys())

        for key_str in keys_in_order:
            try:
                meta = json.loads(key_str)
            except json.JSONDecodeError:
                skipped.append(key_str[:80] + "...")
                continue
            judge_id = meta.get("j")
            dimension = meta.get("d")
            if not isinstance(judge_id, str) or not isinstance(dimension, str):
                skipped.append(key_str[:80] + "...")
                continue

            row = _find_row_for_key(ctx, judge_id, dimension, key_str)
            if row is None:
                print(
                    f"[skip] leaf={leaf.name!r} no matching pair row for key "
                    f"(j={judge_id!r} d={dimension!r}) — "
                    "check MAX_DOCUMENTS / pair seed / eval data match the original run.",
                    flush=True,
                )
                skipped.append(key_str[:120])
                continue

            rng = rng_for_judge_dimension(judge_id, dimension, base_seed=DEFAULT_GEVAL_BASE_SEED)
            judgment = evaluate_fn(row, dimension, judge_id, rng)
            key_to_judgment[key_str] = judgment
            cat = _categorize_api_error(str(judgment.get("rationale", "")))
            print(f"[done] {leaf.name} | {dimension} | {judge_id} | new rationale category: {cat}", flush=True)

    if skipped:
        print(f"Skipped {len(skipped)} key(s) (unparseable or no matching row).", flush=True)

    total_replaced = 0
    leaves_updated: set[Path] = set()
    for path, items in sorted(by_path.items()):
        subset = {k: key_to_judgment[k] for _, k, _ in items if k in key_to_judgment}
        if not subset:
            continue
        n = _rewrite_file(path, subset, dry_run=False)
        total_replaced += n
        if n:
            leaves_updated.add(path.parent)
        print(f"[write] {path.parent.name}/{path.name}: replaced {n} line(s)", flush=True)

    print(f"Finished. Replaced {total_replaced} line(s) total.")

    if args.no_export:
        print("Skipping export (--no-export).")
        return 0
    if total_replaced == 0:
        print("No checkpoint lines were updated; skipping export.")
        return 0

    print("Refreshing exports from checkpoints (no new LLM calls expected)...", flush=True)
    for leaf in sorted(leaves_updated):
        pairs_df, long_df, _ = _load_eval_pairs_and_context(
            eval_dir=resolve_eval_dir_for_judgment_leaf(leaf),
            pair_seed=args.pair_seed,
            max_documents=max_docs,
            n_pairs_per_document=n_pairs,
            judgment_leaf=leaf,
        )
        if args.export_dir is not None:
            exp_root = args.export_dir.expanduser().resolve()
        else:
            exp_root = resolve_export_dir_for_judgment_leaf(leaf)
        out = _refresh_exports(
            pairs_df=pairs_df,
            long_df=long_df,
            checkpoint_dir=leaf,
            export_dir=exp_root,
        )
        where = "forced --export-dir" if args.export_dir is not None else GEVAL_EXPORT_DIRNAME.strip()
        print(f"Exports [{leaf.name}] → {out.resolve()} ({where})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

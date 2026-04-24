#!/usr/bin/env python3
"""Evaluate per-example hygiene for an inputs-refs-preds JSONL file.

Produces per-example metrics, good-hygiene/bad-hygiene JSONL outputs, and
filter stats for both checkpoint predictions and (once) reference summaries.

Usage:
    python report_hygiene.py test_data/llama-3.1-8b-instruct-checkpoint-1000-inputs-refs-preds-1000-examples.jsonl
"""

import argparse
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")

# Lightweight package stub so we can import utils.metrics without the heavy
# utils/__init__.py (which pulls in peft, transformers, etc.).
import types
_utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils")
_pkg = types.ModuleType("utils")
_pkg.__path__ = [_utils_dir]
_pkg.__package__ = "utils"
sys.modules.setdefault("utils", _pkg)

from utils.metrics import hygiene, markup_contamination, has_bad_delimiters
from utils.light_norwegian_parser import NorwegianLightParser, load_lexicon

CRITERIA = [
    "rep_3gram", "compression_ratio", "pred_chars", "max_pred_ref_char_ratio",
    "ends_with_punct", "markup_ratio", "bad_delimiters", "punctuation_score",
    "complete_sentence_ratio", "known_word_ratio",
    "starts_with_complete_sent", "ends_with_complete_sent",
]


def parse_filename(path: str):
    """Parse ``<prefix>checkpoint-<N>-inputs-refs-preds-<suffix>.jsonl``.

    Also accepts filenames without prefix:
    ``checkpoint-<N>-inputs-refs-preds-<suffix>.jsonl``.

    Returns (prefix, checkpoint_tag, suffix) where *prefix* includes
    the trailing hyphen when present (e.g. ``"llama-3.1-8b-instruct-"``).
    """
    basename = os.path.basename(path)
    m = re.match(
        r"^(.*?)(checkpoint-\d+)-inputs-refs-preds-(.+)\.jsonl$",
        basename,
    )
    if not m:
        raise ValueError(
            f"Filename does not match "
            f"[<prefix>]checkpoint-<N>-inputs-refs-preds-<suffix>.jsonl: {basename}"
        )
    return m.group(1), m.group(2), m.group(3)


def evaluate_summary(doc, summary, ref, nlp_parser):
    """Compute all hygiene metrics for a single summary text."""
    h = hygiene(doc, summary)
    h["pred_chars"] = len(summary)
    h["pred_ref_char_ratio"] = (len(summary) / len(ref)) if len(ref) else None
    h.update(markup_contamination(summary))

    h["bad_delimiters"] = has_bad_delimiters(summary)

    lp = nlp_parser.analyze(summary)
    h["complete_sentence_ratio"] = lp["complete_ratio"]
    h["starts_with_complete_sent"] = lp["starts_with_complete_sent"]
    h["ends_with_complete_sent"] = lp["ends_with_complete_sent"]
    h["punctuation_score"] = lp["punctuation_score"]
    h["known_word_ratio"] = lp["known_word_ratio"]
    return h


def check_criteria(h, args):
    """Return an ordered dict of criterion_name → passed (bool)."""
    cr = h["compression_ratio"]
    ratio = h["pred_ref_char_ratio"]
    return {
        "rep_3gram": h["rep_3gram"] <= args.max_rep_3gram,
        "compression_ratio": cr is not None and cr >= args.min_compression_ratio,
        "pred_chars": h["pred_chars"] >= args.min_pred_chars,
        "max_pred_ref_char_ratio": ratio is not None and ratio <= args.max_pred_ref_char_ratio,
        "ends_with_punct": bool(h["ends_with_punct"]),
        "markup_ratio": h["markup_ratio"] <= args.max_markup_ratio,
        "bad_delimiters": not h["bad_delimiters"],
        "punctuation_score": h["punctuation_score"] >= args.min_punctuation_score,
        "complete_sentence_ratio": h["complete_sentence_ratio"] >= args.min_complete_sentence_ratio,
        "known_word_ratio": h["known_word_ratio"] >= args.min_known_word_ratio,
        "starts_with_complete_sent": bool(h["starts_with_complete_sent"]),
        "ends_with_complete_sent": bool(h["ends_with_complete_sent"]),
    }


def new_counters():
    c = {name: 0 for name in CRITERIA}
    c["total"] = 0
    c["passed_all"] = 0
    return c


def print_report(label, counters, args):
    n = counters["total"]
    pct = lambda v: f"{100 * v / n:.1f}" if n else "0.0"

    print(f"\n=== {label} ===")
    print(f"Total examples:                        {n}")
    print(f"Passed (all criteria):                 {counters['passed_all']}/{n} ({pct(counters['passed_all'])}%)")
    print(f"  rep_3gram <= {args.max_rep_3gram}:                    {counters['rep_3gram']}/{n} ({pct(counters['rep_3gram'])}%)")
    print(f"  compression_ratio >= {args.min_compression_ratio}:          {counters['compression_ratio']}/{n} ({pct(counters['compression_ratio'])}%)")
    print(f"  pred_chars >= {args.min_pred_chars}:                   {counters['pred_chars']}/{n} ({pct(counters['pred_chars'])}%)")
    print(f"  pred_ref_char_ratio <= {args.max_pred_ref_char_ratio}:        {counters['max_pred_ref_char_ratio']}/{n} ({pct(counters['max_pred_ref_char_ratio'])}%)")
    print(f"  ends_with_punct:                     {counters['ends_with_punct']}/{n} ({pct(counters['ends_with_punct'])}%)")
    print(f"  markup_ratio <= {args.max_markup_ratio}:               {counters['markup_ratio']}/{n} ({pct(counters['markup_ratio'])}%)")
    print(f"  bad_delimiters == False:              {counters['bad_delimiters']}/{n} ({pct(counters['bad_delimiters'])}%)")
    print(f"  punctuation_score >= {args.min_punctuation_score}:         {counters['punctuation_score']}/{n} ({pct(counters['punctuation_score'])}%)")
    print(f"  complete_sentence_ratio >= {args.min_complete_sentence_ratio}:    {counters['complete_sentence_ratio']}/{n} ({pct(counters['complete_sentence_ratio'])}%)")
    print(f"  known_word_ratio >= {args.min_known_word_ratio}:           {counters['known_word_ratio']}/{n} ({pct(counters['known_word_ratio'])}%)")
    print(f"  starts_with_complete_sent:            {counters['starts_with_complete_sent']}/{n} ({pct(counters['starts_with_complete_sent'])}%)")
    print(f"  ends_with_complete_sent:              {counters['ends_with_complete_sent']}/{n} ({pct(counters['ends_with_complete_sent'])}%)")


def build_stats(input_file, counters, args):
    n = counters["total"]
    pct = lambda v: float(f"{100 * v / n:.1f}") if n else 0.0
    return {
        "input_file": os.path.abspath(input_file),
        "thresholds": {
            "max_rep_3gram": args.max_rep_3gram,
            "min_compression_ratio": args.min_compression_ratio,
            "min_pred_chars": args.min_pred_chars,
            "max_pred_ref_char_ratio": args.max_pred_ref_char_ratio,
            "max_markup_ratio": args.max_markup_ratio,
            "min_punctuation_score": args.min_punctuation_score,
            "min_complete_sentence_ratio": args.min_complete_sentence_ratio,
            "min_known_word_ratio": args.min_known_word_ratio,
        },
        "total": n,
        **{
            key: value
            for name in ["passed_all"] + CRITERIA
            for key, value in [
                (f"passed_{name}", counters[name]),
                (f"passed_{name}_pct", pct(counters[name])),
            ]
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Per-example hygiene evaluation.")
    parser.add_argument("input_file",
                        help="JSONL file matching <prefix>checkpoint-<N>-inputs-refs-preds-<suffix>.jsonl")
    parser.add_argument("--max_rep_3gram", type=float, default=0.2)
    parser.add_argument("--min_compression_ratio", type=float, default=1.0)
    parser.add_argument("--min_pred_chars", type=int, default=50)
    parser.add_argument("--max_pred_ref_char_ratio", type=float, default=1.5)
    parser.add_argument("--max_markup_ratio", type=float, default=0.01)
    parser.add_argument("--min_punctuation_score", type=float, default=-0.1)
    parser.add_argument("--min_complete_sentence_ratio", type=float, default=0.6)
    parser.add_argument("--min_known_word_ratio", type=float, default=0.9)
    parser.add_argument("--lexicon", type=str, default=None,
                        help="Path to external Norwegian lexicon file (one word per line).")
    parser.add_argument("--spacy-model", type=str, default="nb_core_news_md",
                        help="spaCy model for sentence parsing.")
    args = parser.parse_args()

    prefix, ckpt_tag, suffix = parse_filename(args.input_file)
    out_dir = os.path.dirname(args.input_file) or "."

    def out(tag, kind, ext):
        return os.path.join(out_dir, f"{prefix}{tag}-hygiene-{kind}-{suffix}.{ext}")

    def out_hygiene_variant(tag, variant):
        return os.path.join(out_dir, f"{prefix}{tag}-{variant}-hygiene-{suffix}.jsonl")

    ckpt_metrics_path   = out(ckpt_tag, "metrics", "jsonl")
    ckpt_good_path      = out_hygiene_variant(ckpt_tag, "good")
    ckpt_bad_path       = out_hygiene_variant(ckpt_tag, "bad")
    ckpt_stats_path     = out(ckpt_tag, "filter-stats", "json")

    ref_metrics_path    = out("reference", "metrics", "jsonl")
    ref_good_path       = out_hygiene_variant("reference", "good")
    ref_bad_path        = out_hygiene_variant("reference", "bad")
    ref_stats_path      = out("reference", "filter-stats", "json")

    _script_mtime = os.path.getmtime(__file__)
    need_ref = (not os.path.exists(ref_metrics_path)
                or os.path.getmtime(ref_metrics_path) < _script_mtime)

    lexicon = load_lexicon(args.lexicon) if args.lexicon else None
    nlp_parser = NorwegianLightParser(lexicon=lexicon, model=args.spacy_model)

    ckpt_counters = new_counters()
    ref_counters = new_counters() if need_ref else None

    ckpt_mf = open(ckpt_metrics_path, "w", encoding="utf-8")
    ckpt_gf = open(ckpt_good_path, "w", encoding="utf-8")
    ckpt_bf = open(ckpt_bad_path, "w", encoding="utf-8")
    ref_mf = open(ref_metrics_path, "w", encoding="utf-8") if need_ref else None
    ref_gf = open(ref_good_path, "w", encoding="utf-8") if need_ref else None
    ref_bf = open(ref_bad_path, "w", encoding="utf-8") if need_ref else None

    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    ckpt_mf.write("\n")
                    ckpt_gf.write("\n")
                    ckpt_bf.write("\n")
                    if ref_mf and ref_gf and ref_bf:
                        ref_mf.write("\n")
                        ref_gf.write("\n")
                        ref_bf.write("\n")
                    continue

                obj = json.loads(line)
                doc = obj["input_text"]
                pred = obj["prediction"]
                ref = obj["reference"]

                # --- Checkpoint (prediction) evaluation ---
                h = evaluate_summary(doc, pred, ref, nlp_parser)
                ckpt_mf.write(json.dumps(h, ensure_ascii=False) + "\n")

                criteria = check_criteria(h, args)
                passed = all(criteria.values())
                ckpt_counters["total"] += 1
                if passed:
                    ckpt_counters["passed_all"] += 1
                for name, ok in criteria.items():
                    if ok:
                        ckpt_counters[name] += 1
                ckpt_gf.write((line if passed else "") + "\n")
                if passed:
                    ckpt_bf.write("\n")
                else:
                    bad_obj = {
                        "input_text": doc,
                        "reference": ref,
                        "prediction": None,
                    }
                    ckpt_bf.write(json.dumps(bad_obj, ensure_ascii=False) + "\n")

                # --- Reference evaluation (only on first run) ---
                if need_ref and ref_mf and ref_gf and ref_bf and ref_counters is not None:
                    rh = evaluate_summary(doc, ref, ref, nlp_parser)
                    ref_mf.write(json.dumps(rh, ensure_ascii=False) + "\n")

                    rcriteria = check_criteria(rh, args)
                    rpassed = all(rcriteria.values())
                    ref_counters["total"] += 1
                    if rpassed:
                        ref_counters["passed_all"] += 1
                    for name, ok in rcriteria.items():
                        if ok:
                            ref_counters[name] += 1
                    ref_gf.write((line if rpassed else "") + "\n")
                    if rpassed:
                        ref_bf.write("\n")
                    else:
                        ref_bad_obj = {
                            "input_text": doc,
                            "reference": ref,
                            "prediction": None,
                        }
                        ref_bf.write(json.dumps(ref_bad_obj, ensure_ascii=False) + "\n")
    finally:
        ckpt_mf.close()
        ckpt_gf.close()
        ckpt_bf.close()
        if ref_mf:
            ref_mf.close()
        if ref_gf:
            ref_gf.close()
        if ref_bf:
            ref_bf.close()

    # --- Checkpoint report ---
    print_report(f"Checkpoint: {ckpt_tag}", ckpt_counters, args)
    print(f"\nMetrics:  {ckpt_metrics_path}")
    print(f"Good:     {ckpt_good_path}")
    print(f"Bad:      {ckpt_bad_path}")

    ckpt_stats = build_stats(args.input_file, ckpt_counters, args)
    with open(ckpt_stats_path, "w", encoding="utf-8") as sf:
        json.dump(ckpt_stats, sf, indent=2, ensure_ascii=False)
    print(f"Stats:    {ckpt_stats_path}")

    # --- Reference report ---
    if need_ref:
        print_report("Reference summaries", ref_counters, args)
        print(f"\nMetrics:  {ref_metrics_path}")
        print(f"Good:     {ref_good_path}")
        print(f"Bad:      {ref_bad_path}")

        ref_stats = build_stats(args.input_file, ref_counters, args)
        with open(ref_stats_path, "w", encoding="utf-8") as sf:
            json.dump(ref_stats, sf, indent=2, ensure_ascii=False)
        print(f"Stats:    {ref_stats_path}")
    else:
        print(f"\nReference hygiene already exists: {ref_metrics_path} (skipped)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze checkpoint-N inputs-refs-preds across all models.

Produces a report for deciding which models to continue vs rerun with correct prompt/tokens.
Checks: repetition, empty outputs, special token artifacts, prompt format consistency,
and content relevance (off-topic/hallucination indicators).

Usage:
  python analyze_checkpoint_5000_all_models.py              # default: checkpoint 5000
  python analyze_checkpoint_5000_all_models.py --checkpoint 6000
"""

import argparse
import json
import os
import re
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Expected prompt format per model (from PROMPT_TOKENIZATION_ANALYSIS.md)
# Key: model base name (from path, e.g. normistral-7b-apptainer-fsdp -> normistral-7b)
MODEL_PROMPT_EXPECTATIONS = {
    "normistral-7b": ("mistral", "apply_chat_template"),
    "normistral-11b": ("mistral", "apply_chat_template"),
    "normistral-7b-instruct": ("mistral", "apply_chat_template"),
    "norwai-mistral-7b-instruct": ("mistral", "apply_chat_template"),
    "norskgpt-llama3-8b": ("llama3", "apply_chat_template"),
    "llama-3.1-8b-instruct": ("llama3.1", "apply_chat_template"),
    "llama-2-13b-chat-norwegian": ("llama2", "manual_or_chat"),
    "eurollm-9b-instruct": ("chatml", "apply_chat_template"),
    "nb-gpt-j-6b": ("alpaca", "manual"),
    "gemma-2b": ("plain", "manual"),
    "gemma-7b-it": ("chatml", "apply_chat_template"),
    "gemma-2-9b": ("plain", "manual"),
    "viking-7b": ("plain", "manual"),
    "viking-13b": ("plain", "manual"),
}

# Special tokens that should NOT appear in model output (prompt format artifacts)
SPECIAL_TOKEN_PATTERNS = [
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"<\|endoftext\|>",
    r"<\|eot_id\|>",
    r"\[INST\]",
    r"\[\/INST\]",
    r"<s>",
    r"</s>",
    r"<\|begin_of_text\|>",
    r"<\|end_of_text\|>",
    r"<\|start_header_id\|>",
    r"<\|end_header_id\|>",
]


def ngram_repetition(text: str, n: int = 3) -> float:
    """3-gram repetition rate. Higher = more repetition."""
    tokens = re.findall(r"\d+(?:[.,]\d+)?|[\w/-]+|[^\w\s]", text.lower())
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    return (total - unique) / total if total > 0 else 0.0


def detect_prompt_format(prompt: str) -> str:
    """Detect which prompt format was used from the prompt string."""
    if not prompt:
        return "unknown"
    if "<|im_start|>" in prompt and "<|im_end|>" in prompt:
        return "chatml"
    if "[INST]" in prompt and "[/INST]" in prompt and "<s>" in prompt:
        return "mistral"
    if "Instruction:" in prompt and "Response:" in prompt:
        return "alpaca"
    # Llama3 vs plain - check for chat-style
    if "Du er en ekspert" in prompt and not any(x in prompt for x in ["<|im_start|>", "[INST]", "Instruction:"]):
        return "plain"
    if "<|begin_of_text|>" in prompt or "Llama" in prompt:
        return "llama3"
    return "unknown"


def has_special_tokens_in_output(pred: str) -> List[str]:
    """Return list of special token patterns found in prediction."""
    found = []
    for pat in SPECIAL_TOKEN_PATTERNS:
        if re.search(pat, pred):
            found.append(pat)
    return found


def off_topic_indicators(pred: str, input_snippet: str) -> List[str]:
    """Heuristic: indicators that prediction may be off-topic/hallucinating."""
    issues = []
    pred_lower = pred.lower()
    input_lower = input_snippet.lower()[:500]
    
    # Alphabet run (common collapse pattern)
    if "abcdefg" in pred_lower or "æøå" in pred_lower and pred.count("æ") > 3:
        issues.append("alphabet_repetition")
    
    # Hashtags (often from wrong template)
    if "#" in pred and sum(1 for c in pred if c == "#") >= 3:
        issues.append("hashtag_spam")
    
    # Pred way longer than typical summary (>500 words) - rambling
    words = len(re.findall(r"\w+", pred))
    if words > 500:
        issues.append("very_long_rambling")
    
    # Common wrong-document patterns
    if "tilstedeværelse" in pred_lower and "dato sted navn" in pred_lower:
        issues.append("meeting_attendance_template")
    if "smittevernloven" in pred_lower and "korona" in pred_lower and len(input_lower) < 500:
        issues.append("covid_template_mismatch")
    
    return issues


def analyze_file(filepath: str) -> Dict:
    """Analyze a single checkpoint-N-genG-inputs-refs-preds-X-examples.jsonl file."""
    model_name = Path(filepath).parent.parent.name  # e.g. normistral-7b-apptainer-fsdp
    model_base = model_name.replace("-apptainer-fsdp", "")
    
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    if not data:
        return {"model": model_base, "error": "empty_file", "n": 0}
    
    rep_scores = []
    empty_count = 0
    high_rep_count = 0
    special_token_count = 0
    off_topic_count = 0
    prompt_format = None
    special_tokens_seen = set()
    off_topic_samples = []
    
    for entry in data:
        pred = entry.get("prediction", "").strip()
        inp = entry.get("input_text", "")
        prompt = entry.get("prompt", "")
        
        if prompt_format is None and prompt:
            prompt_format = detect_prompt_format(prompt)
        
        # Repetition
        rep = ngram_repetition(pred, 3)
        rep_scores.append(rep)
        if rep >= 0.5:
            high_rep_count += 1
        
        # Empty
        words = len(re.findall(r"\w+", pred))
        if words == 0:
            empty_count += 1
        
        # Special tokens in output
        st = has_special_tokens_in_output(pred)
        if st:
            special_token_count += 1
            special_tokens_seen.update(st)
        
        # Off-topic
        ot = off_topic_indicators(pred, inp)
        if ot:
            off_topic_count += 1
            if len(off_topic_samples) < 3:
                off_topic_samples.append({"pred_snippet": pred[:200], "issues": ot})
    
    n = len(data)
    mean_rep = sum(rep_scores) / n if n else 0
    
    # Expected format
    expected = MODEL_PROMPT_EXPECTATIONS.get(model_base, (None, None))
    expected_format, expected_method = expected
    format_ok = (expected_format and prompt_format == expected_format) if expected_format else None
    
    return {
        "model": model_base,
        "model_dir": model_name,
        "n": n,
        "mean_rep_3gram": round(mean_rep, 4),
        "high_rep_pct": round(100 * high_rep_count / n, 1) if n else 0,
        "empty_pct": round(100 * empty_count / n, 1) if n else 0,
        "special_token_pct": round(100 * special_token_count / n, 1) if n else 0,
        "off_topic_pct": round(100 * off_topic_count / n, 1) if n else 0,
        "prompt_format_detected": prompt_format,
        "expected_format": expected_format,
        "format_match": format_ok,
        "special_tokens_seen": list(special_tokens_seen),
        "off_topic_samples": off_topic_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze checkpoint-N genG inputs-refs-preds across all models")
    parser.add_argument("--checkpoint", type=int, default=5000,
                        help="Checkpoint number to analyze (default: 5000)")
    args = parser.parse_args()
    checkpoint = args.checkpoint

    base = Path(__file__).resolve().parent.parent / "models"
    pattern = str(base / "*" / "all_eval_results" / f"checkpoint-{checkpoint}-gen*-inputs-refs-preds-*-examples.jsonl")
    files = sorted(glob.glob(pattern))
    
    print(f"Found {len(files)} checkpoint-{checkpoint} files\n")
    
    results = []
    for f in files:
        try:
            r = analyze_file(f)
            results.append(r)
        except Exception as e:
            results.append({"model": Path(f).parent.parent.name, "error": str(e), "n": 0})
    
    # Build report
    lines = [
        f"# Checkpoint-{checkpoint} Analysis: All Models",
        "",
        "Recommendations for **continue** vs **rerun** with correct prompt/tokens.",
        "",
        "## Summary Table",
        "",
        "| Model | N | Mean Rep | HighRep% | Empty% | S.Tok% | OffTopic% | Prompt | Expected | Match |",
        "|-------|---|----------|---------|--------|--------|------------|--------|----------|-------|",
    ]
    
    continue_models = []
    rerun_models = []
    
    for r in sorted(results, key=lambda x: x.get("model", "")):
        if "error" in r:
            lines.append(f"| {r.get('model','?')} | - | ERROR: {r['error']} |")
            rerun_models.append(r.get("model", "?"))
            continue
        
        m = r["model"]
        n = r["n"]
        mean_rep = r["mean_rep_3gram"]
        high_rep = r["high_rep_pct"]
        empty = r["empty_pct"]
        stok = r["special_token_pct"]
        off = r["off_topic_pct"]
        pf = r.get("prompt_format_detected", "?")
        exp = r.get("expected_format", "?")
        match = "✓" if r.get("format_match") is True else ("✗" if r.get("format_match") is False else "?")
        
        lines.append(f"| {m} | {n} | {mean_rep:.3f} | {high_rep}% | {empty}% | {stok}% | {off}% | {pf} | {exp} | {match} |")
        
        # Recommendation
        needs_rerun = (
            r.get("format_match") is False or
            high_rep > 20 or
            empty > 5 or
            stok > 0 or
            off > 30
        )
        if needs_rerun:
            rerun_models.append(m)
        else:
            continue_models.append(m)
    
    lines.extend([
        "",
        "## Recommendations",
        "",
        "### Continue (run more steps / monitor)",
        "",
    ])
    if continue_models:
        for m in continue_models:
            lines.append(f"- **{m}**")
    else:
        lines.append("- *(none clearly healthy)*")
    
    lines.extend([
        "",
        "### Rerun with correct prompt/tokens",
        "",
    ])
    if rerun_models:
        for m in rerun_models:
            lines.append(f"- **{m}**")
    else:
        lines.append("- *(none)*")
    
    lines.extend([
        "",
        "## Details (models with issues)",
        "",
    ])
    
    for r in results:
        if "error" in r:
            continue
        if (r.get("special_tokens_seen") or r.get("off_topic_samples") or
            r.get("format_match") is False or r.get("high_rep_pct", 0) > 15):
            lines.append(f"### {r['model']}")
            lines.append("")
            if r.get("format_match") is False:
                lines.append(f"- **Prompt mismatch**: detected `{r.get('prompt_format_detected')}` vs expected `{r.get('expected_format')}`")
            if r.get("special_tokens_seen"):
                lines.append(f"- **Special tokens in output**: {r['special_tokens_seen']}")
            if r.get("off_topic_samples"):
                for s in r["off_topic_samples"][:2]:
                    lines.append(f"- Off-topic sample (issues: {s['issues']}): `{s['pred_snippet'][:150]}...`")
            lines.append("")
    
    report = "\n".join(lines)
    print(report)
    
    analyses_dir = base.parent / "checkpoint_analyses"
    analyses_dir.mkdir(parents=True, exist_ok=True)
    out_path = analyses_dir / f"CHECKPOINT_{checkpoint}_ANALYSIS_REPORT.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()

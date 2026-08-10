#!/usr/bin/env python3
"""Generate heuristic summary quality analyses for all models with checkpoint-6000 data."""

import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models"
ANALYSES_DIR = BASE / "checkpoint_analyses"

MODELS = [
    "normistral-11b-apptainer-fsdp",
    "gemma-2-9b-apptainer-fsdp",
    "gemma-7b-apptainer-fsdp",
    "norwai-mistral-7b-instruct-apptainer-fsdp",
    "llama-3.1-8b-instruct-apptainer-fsdp",
    "llama-2-13b-chat-norwegian-apptainer-fsdp",
    "gemma-2b-apptainer-fsdp",
    "nb-gpt-j-6b-apptainer-fsdp",
]

# Patterns to detect failure modes (avoid Norwegian words: har, eller, etc.)
PATTERNS = {
    "alpaca_instruction": r"Hvilke ressurser har du tilgjengelig",
    "response_label": r"\bResponse\s*:",
    "hashtag_spam": r"#{3,}",
    "xx_placeholder": r"XX-XX XXXX|XXX\.|den XX\.|XX\.",
    "swedish_danish": r"\b(och|för|från|denna|blivit|angående|till att vara|för att|istället för)\b|für |af |till att vara",  # Swedish/Danish/German - avoid Norwegian "har","eller"
    "truncated_short": 50,  # chars - very short likely truncated
}


def load_samples(model: str, n: int = 15) -> list:
    fpath = MODELS_DIR / model / "all_eval_results" / "checkpoint-6000-gen0-inputs-refs-preds-1000-examples.jsonl"
    if not fpath.exists():
        return []
    lines = []
    with open(fpath) as f:
        for l in f:
            if l.strip():
                lines.append(json.loads(l))
    if not lines:
        return []
    # Sample: first 5, middle 3, 5 longest
    mid = len(lines) // 2
    samples = lines[:5] + lines[mid:mid+3] + sorted(lines, key=lambda x: len(x.get("prediction", "")), reverse=True)[:5]
    return samples[:n]


def load_eval_results(model: str) -> dict:
    fpath = MODELS_DIR / model / "all_eval_results" / "checkpoint-6000-gen0-eval-results-1000-examples.json"
    if not fpath.exists():
        return {}
    with open(fpath) as f:
        return json.load(f)


def analyze_predictions(samples: list) -> dict:
    issues = {"alpaca": 0, "response_label": 0, "hashtag": 0, "xx_placeholder": 0, "lang_mix": 0, "truncated": 0, "empty": 0}
    good_examples = []
    bad_examples = []
    for s in samples:
        pred = s.get("prediction", "").strip()
        ref = s.get("reference", "")[:200]
        inp = s.get("input_text", "")[:150]
        if not pred:
            issues["empty"] += 1
            continue
        if len(pred) < PATTERNS["truncated_short"]:
            issues["truncated"] += 1
        if re.search(PATTERNS["alpaca_instruction"], pred):
            issues["alpaca"] += 1
            bad_examples.append(("Alpaca instruction", inp, ref, pred))
        elif re.search(PATTERNS["response_label"], pred):
            issues["response_label"] += 1
        elif re.search(PATTERNS["hashtag_spam"], pred):
            issues["hashtag"] += 1
            bad_examples.append(("Hashtag spam", inp, ref, pred))
        elif re.search(PATTERNS["xx_placeholder"], pred):
            issues["xx_placeholder"] += 1
            bad_examples.append(("XX placeholder", inp, ref, pred))
        elif re.search(PATTERNS["swedish_danish"], pred, re.I):
            issues["lang_mix"] += 1
            bad_examples.append(("Language mix (SV/DA)", inp, ref, pred))
        else:
            if len(pred) > 100 and len(pred) < 2000:
                good_examples.append((inp, ref, pred))
    return {"issues": issues, "good": good_examples[:4], "bad": bad_examples[:4]}


def model_short_name(model: str) -> str:
    return model.replace("-apptainer-fsdp", "")


def generate_report(model: str) -> str:
    short = model_short_name(model)
    samples = load_samples(model)
    if not samples:
        return f"# {short} Checkpoint 6000: No data\n\nNo checkpoint-6000-gen0-inputs-refs-preds-1000-examples.jsonl found."
    eval_res = load_eval_results(model)
    analysis = analyze_predictions(samples)
    issues = analysis["issues"]
    good = analysis["good"]
    bad = analysis["bad"]

    lines = [
        f"# {short} Checkpoint 6000: Heuristic Summary Quality Analysis",
        "",
        "Analysis of predictions at checkpoint 6000, with Norwegian→English translations for readability.",
        "",
        "---",
        "",
        "## Metrics (checkpoint-6000-gen0-eval-results-1000-examples.json)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| ROUGE-Lsum | {eval_res.get('eval_rougeLsum', '?'):.1f} |" if isinstance(eval_res.get('eval_rougeLsum'), (int, float)) else "| ROUGE-Lsum | ? |",
        f"| BERTScore F1 (mean) | {eval_res.get('eval_reference_bertscore_f1_mean', '?'):.2f} |" if isinstance(eval_res.get('eval_reference_bertscore_f1_mean'), (int, float)) else "| BERTScore | ? |",
        f"| 3-gram repetition (hygiene) | {eval_res.get('eval_hygiene_mean_rep_3gram', '?'):.3f} |" if isinstance(eval_res.get('eval_hygiene_mean_rep_3gram'), (int, float)) else "| rep_3gram | ? |",
        f"| Compression ratio | {eval_res.get('eval_hygiene_mean_compression_ratio', '?'):.2f} |" if isinstance(eval_res.get('eval_hygiene_mean_compression_ratio'), (int, float)) else "| compression | ? |",
        "",
        "---",
        "",
        "## Failure Modes Detected (sampled)",
        "",
    ]
    if any(issues.values()):
        for k, v in issues.items():
            if v > 0:
                lines.append(f"- **{k}**: {v} of {len(samples)} sampled")
        lines.extend(["", "### Example failure cases", ""])
        for typ, inp, ref, pred in bad:
            lines.extend([
                f"**{typ}**",
                f"- Input: {inp[:120].replace(chr(10), ' ')}...",
                f"- Ref: {ref[:150]}...",
                f"- Pred: {pred[:300]}...",
                "",
            ])
    else:
        lines.append("No major failure patterns detected in sample.")
    lines.extend(["", "---", "", "## Good Examples (with translations)", ""])
    for inp, ref, pred in good:
        pred_trim = pred[:400] + "..." if len(pred) > 400 else pred
        ref_trim = ref[:250] + "..." if len(ref) > 250 else ref
        lines.extend([
            "**Norwegian (prediction):**",
            f"> {pred_trim}",
            "",
            "**English (reference snippet):**",
            f"> {ref_trim}",
            "",
        ])
    lines.extend([
        "",
        "---",
        "",
        "## Summary",
        "",
        f"- **Samples analyzed:** {len(samples)}",
        f"- **ROUGE-Lsum:** {eval_res.get('eval_rougeLsum', 'N/A')}",
        f"- **Main issues:** {', '.join(k for k,v in issues.items() if v>0) or 'None detected'}",
        "",
    ])
    return "\n".join(lines)


def main():
    for model in MODELS:
        short = model_short_name(model)
        report = generate_report(model)
        safe_name = short.upper().replace("-", "_").replace(".", "_")
        ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = ANALYSES_DIR / f"{safe_name}_CHECKPOINT_6000_SUMMARY_ANALYSIS.md"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Wrote {out_path.name}")


if __name__ == "__main__":
    main()

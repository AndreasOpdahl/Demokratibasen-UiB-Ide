"""
Interactive HTML viewer for checkpoint evaluation results.

Generates a self-contained HTML file with:
- A large interactive Plotly chart in the centre
- Model checkboxes along the top
- Evaluation run checkboxes (``all_eval_results/`` vs ``eval_results_min*_max*_tokens/``)
- Generation checkboxes (gen0, gen1, gen2, ...)
- Metric checkboxes on the right

Checkpoint result files must include a gen<N> fragment.

Usage:
    # View a single model
    python view_evaluation_results.py \
        --model_dir models/gemma-2-9b-apptainer-fsdp

    # View multiple models for comparison
    python view_evaluation_results.py \
        --model_dir models/gemma-2-9b-apptainer-fsdp models/normistral-7b-apptainer-fsdp

    # A direct all_eval_results folder is also accepted
    python view_evaluation_results.py \
        --model_dir models/*/all_eval_results

    # Custom output path
    python view_evaluation_results.py \
        --model_dir models/gemma-2-9b-apptainer-fsdp \
        --output viewer.html
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from visualise_checkpoint_results import (
    EVAL_RESULTS_TOKEN_SUBDIR_RE,
    discover_all_eval_run_ids,
    eval_run_display_label,
    extract_metrics,
)

def normalise_model_dir(path: str) -> str:
    """Accept ``model_dir/all_eval_results`` as well as plain ``model_dir``.

    Strips a trailing ``all_eval_results`` or ``eval_results_min*_max*_tokens``
    component so that ``--model_dir models/*/all_eval_results`` works the same
    as ``--model_dir models/*``.
    """
    clean = path.rstrip("/")
    basename = os.path.basename(clean)
    if basename == "all_eval_results" or EVAL_RESULTS_TOKEN_SUBDIR_RE.match(basename):
        return os.path.dirname(clean)
    return clean


BASELINE_MODEL_NAME = "gpt-4o-mini (baseline)"
DEFAULT_REFERENCE_FILE = (
    "~/OneDrive/Shared/Demokratibasen-UiB-Ide/TrainingDatasets/"
    "text_summary_dataset_202601/baseline_metrics/"
    "evaluation_149978_text_summary_examples_val_all.json"
)

EVAL_RESULTS_RE = re.compile(
    r"^(?:checkpoint|regular-checkpoint|major-checkpoint)-(?P<step>\d+)-"
    r"(?P<gen>gen(?P<gen_num>\d+))-"
    r"eval-results"
    r"-(?P<suffix>\d+-examples)"
    r"\.json$"
)

METRIC_GROUPS = {
    "ROUGE": {
        "rouge1": {"label": "ROUGE-1", "color": "#1f77b4", "dash": "dashdot"},
        "rouge2": {"label": "ROUGE-2", "color": "#2ca02c", "dash": "dashdot"},
        "rougeL": {"label": "ROUGE-L", "color": "#ff7f0e", "dash": "dashdot"},
        "rougeLsum": {"label": "ROUGE-Lsum", "color": "#d62728", "dash": "dot"},
    },
    "Reference": {
        "bertscore_f1": {"label": "BERTScore F1", "color": "#9467bd", "dash": "longdash"},
    },
    "Hygiene": {
        "compression_ratio": {"label": "Compression Ratio", "color": "#8c564b", "dash": "longdashdot"},
        "repetition_3gram": {"label": "3-gram Repetition", "color": "#e377c2", "dash": "longdashdot"},
        "ends_with_punct": {"label": "Ends w/ Punct", "color": "#7f7f7f", "dash": "longdashdot"},
        "pred_ref_tokens_ratio": {"label": "Pred/ref tokens", "color": "#d62728", "dash": "dash"},
        "mean_pred_tokens": {"label": "Predicted summary tokens", "color": "#3182bd", "dash": "dot"},
        "mean_ref_tokens": {"label": "Reference summary tokens", "color": "#31a354", "dash": "longdash"},
    },
    "Faithfulness": {
        "entailment_score": {"label": "Entailment Score", "color": "#17becf", "dash": "solid"},
        "outlier_rate": {"label": "Outlier Rate", "color": "#bcbd22", "dash": "solid"},
    },
}

MODEL_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def load_reference_metrics(reference_file: str) -> Dict[str, float]:
    """Load baseline reference metrics in the viewer's metric-key namespace."""
    path = os.path.expanduser(reference_file)
    if not os.path.isfile(path):
        print(f"Warning: reference file not found: {path}", file=sys.stderr)
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: could not read reference file {path}: {exc}", file=sys.stderr)
        return {}

    metrics: Dict[str, float] = {}

    reference_metrics = data.get("reference_metrics") if isinstance(data, dict) else None
    if isinstance(reference_metrics, dict):
        for src_key, dst_key in (
            ("rouge1", "rouge1"),
            ("rouge2", "rouge2"),
            ("rougeL", "rougeL"),
            ("rougeLsum", "rougeLsum"),
            ("bertscore_f1_mean", "bertscore_f1"),
        ):
            value = reference_metrics.get(src_key)
            if isinstance(value, (int, float)):
                metrics[dst_key] = float(value)

    hygiene_metrics = data.get("hygiene_metrics") if isinstance(data, dict) else None
    if isinstance(hygiene_metrics, dict):
        for src_key, dst_key in (
            ("mean_compression_ratio", "compression_ratio"),
            ("mean_rep_3gram", "repetition_3gram"),
            ("ratio_ends_with_punct", "ends_with_punct"),
        ):
            value = hygiene_metrics.get(src_key)
            if isinstance(value, (int, float)):
                metrics[dst_key] = float(value)

    faithfulness_metrics = data.get("faithfulness_metrics") if isinstance(data, dict) else None
    if isinstance(faithfulness_metrics, dict):
        value = faithfulness_metrics.get("mean_entailment_score")
        if isinstance(value, (int, float)):
            metrics["entailment_score"] = float(value)
        outlier_value = faithfulness_metrics.get("mean_ratio_outliers")
        if outlier_value is None:
            outlier_value = faithfulness_metrics.get("mean_outlier_rate")
        if isinstance(outlier_value, (int, float)):
            metrics["outlier_rate"] = float(outlier_value)

    # Also support files that already use checkpoint eval-result field names.
    extracted = extract_metrics({0: data})
    for key, pts in extracted.items():
        if pts:
            metrics.setdefault(key, float(pts[0][1]))

    return metrics


def normalize_examples_suffix(suffix: Optional[str]) -> str:
    """Return canonical examples suffix."""
    if not suffix:
        raise ValueError("Evaluation result filenames must include a <N>-examples suffix")
    if suffix.startswith("examples_"):
        count = suffix.replace("examples_", "", 1)
        if count.isdigit():
            return f"{count}-examples"
    return suffix


def examples_suffix_sort_key(suffix: str) -> Tuple[int, str]:
    match = re.match(r"^(\d+)-examples$", suffix)
    if match:
        return int(match.group(1)), suffix
    return 10**9, suffix


def load_evaluation_results_by_generation(model_dir: str, run_id: str) -> Dict[str, Dict[str, Dict[int, dict]]]:
    """Load eval results as examples_suffix -> gen_id -> checkpoint_step -> result."""
    results_dir = os.path.join(model_dir, run_id)
    if not os.path.isdir(results_dir):
        return {}

    out: Dict[str, Dict[str, Dict[int, dict]]] = {}
    for name in sorted(os.listdir(results_dir)):
        match = EVAL_RESULTS_RE.match(name)
        if not match:
            continue
        step = int(match.group("step"))
        gen_num = int(match.group("gen_num"))
        gen_id = f"gen{gen_num}"
        suffix = normalize_examples_suffix(match.group("suffix"))
        path = os.path.join(results_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: could not read {path}: {exc}", file=sys.stderr)
            continue
        out.setdefault(suffix, {}).setdefault(gen_id, {})[step] = data
    return out


def build_chart_data(model_dirs: List[str], reference_file: str = DEFAULT_REFERENCE_FILE) -> dict:
    """Load all models and evaluation runs; return JSON for the HTML viewer.

    Each model may have several folders: ``all_eval_results/`` and
    ``eval_results_min*_*max*_tokens/`` (same JSON layout). The viewer exposes
    one checkbox per distinct folder name found across the given model dirs.
    """
    model_dirs = list(dict.fromkeys(normalise_model_dir(d) for d in model_dirs))
    eval_run_ids = discover_all_eval_run_ids(model_dirs)
    if not eval_run_ids:
        eval_run_ids = ["all_eval_results"]

    models: Dict[str, dict] = {}
    all_metric_keys = set()
    all_suffixes = set()
    all_gens = set()
    all_steps = set()

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir.rstrip("/"))
        runs_out: Dict[str, dict] = {}

        for run_id in eval_run_ids:
            results_dir = os.path.join(model_dir, run_id)
            if not os.path.isdir(results_dir):
                continue

            loaded = load_evaluation_results_by_generation(model_dir, run_id)
            if not loaded:
                continue

            run_entry = {"suffixes": {}}
            for suffix, gens in sorted(loaded.items(), key=lambda item: examples_suffix_sort_key(item[0])):
                suffix_entry = {"gens": {}}
                for gen_id, results in sorted(gens.items(), key=lambda item: int(item[0][3:])):
                    extracted = extract_metrics(results)
                    if not extracted:
                        continue
                    gen_entry = {"metrics": {}}
                    for key, pts in extracted.items():
                        steps, vals = zip(*pts) if pts else ([], [])
                        gen_entry["metrics"][key] = {"steps": list(steps), "values": list(vals)}
                        all_metric_keys.add(key)
                        all_steps.update(steps)
                    suffix_entry["gens"][gen_id] = gen_entry
                    all_gens.add(gen_id)
                if suffix_entry["gens"]:
                    run_entry["suffixes"][suffix] = suffix_entry
                    all_suffixes.add(suffix)

            if run_entry["suffixes"]:
                runs_out[run_id] = run_entry

        if runs_out:
            models[model_name] = {"runs": runs_out}

    present_run_ids = set()
    for mentry in models.values():
        present_run_ids.update(mentry["runs"].keys())
    if not present_run_ids:
        present_run_ids.add("all_eval_results")

    reference_metrics = load_reference_metrics(reference_file) if reference_file else {}
    if reference_metrics:
        baseline_runs: Dict[str, dict] = {}
        baseline_suffixes = sorted(all_suffixes or {"1000-examples"}, key=examples_suffix_sort_key)
        for run_id in present_run_ids:
            baseline_runs[run_id] = {"suffixes": {}}
            for suffix in baseline_suffixes:
                baseline_runs[run_id]["suffixes"][suffix] = {
                    "gens": {
                        "gen0": {
                            "metrics": {
                                key: {"steps": [], "values": [value], "baseline": True}
                                for key, value in reference_metrics.items()
                            }
                        }
                    }
                }
                all_suffixes.add(suffix)
        models[BASELINE_MODEL_NAME] = {"runs": baseline_runs, "baseline": True}
        all_metric_keys.update(reference_metrics.keys())
        all_gens.add("gen0")

    eval_runs = [
        {"id": rid, "label": eval_run_display_label(rid)}
        for rid in sorted(present_run_ids, key=lambda n: (0 if n == "all_eval_results" else 1, n))
    ]
    suffixes = [
        {"id": s, "label": s}
        for s in sorted(all_suffixes, key=examples_suffix_sort_key)
    ]
    gens = [
        {"id": g, "label": g}
        for g in sorted(all_gens, key=lambda x: int(x[3:]))
    ]

    metric_catalog = {}
    for group_name, group_metrics in METRIC_GROUPS.items():
        for key, meta in group_metrics.items():
            if key in all_metric_keys:
                metric_catalog[key] = {**meta, "group": group_name}

    return {
        "models": models,
        "metric_catalog": metric_catalog,
        "eval_runs": eval_runs,
        "suffixes": suffixes,
        "gens": gens,
        "baseline_model_name": BASELINE_MODEL_NAME,
        "x_range": [min(all_steps), max(all_steps)] if all_steps else [0, 1],
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Checkpoint Evaluation Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         display: flex; flex-direction: column; height: 100vh; background: #fafafa; }
  header { padding: 10px 20px; background: #fff; border-bottom: 1px solid #ddd;
           display: flex; align-items: center; gap: 18px; flex-wrap: wrap; }
  header h1 { font-size: 16px; white-space: nowrap; }
  .model-bar { display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center; }
  .model-bar label { font-size: 13px; cursor: pointer; display: flex; align-items: center; gap: 4px; }
  .model-swatch { width: 12px; height: 12px; border-radius: 2px; display: inline-block; }
  .main { display: flex; flex: 1; overflow: hidden; }
  #chart { flex: 1; min-width: 0; }
  .sidebar { width: 210px; min-width: 180px; padding: 12px 14px; background: #fff;
             border-left: 1px solid #ddd; overflow-y: auto; }
  .sidebar h2 { font-size: 13px; text-transform: uppercase; color: #888; margin: 10px 0 4px; }
  .sidebar h2:first-child { margin-top: 0; }
  .sidebar label { display: flex; align-items: center; gap: 5px; font-size: 13px;
                   cursor: pointer; padding: 2px 0; }
  .sidebar .metric-swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
  .variant-toggle { padding: 4px 12px; background: #fff; border-bottom: 1px solid #ddd;
                    display: flex; gap: 16px; align-items: center; font-size: 13px; }
  .variant-toggle label { cursor: pointer; display: flex; align-items: center; gap: 4px; }
  .bulk-btn { font-size: 11px; cursor: pointer; background: #eee; border: 1px solid #ccc;
              border-radius: 3px; padding: 1px 6px; color: #555; }
  .bulk-btn:hover { background: #ddd; }
</style>
</head>
<body>

<header>
  <h1>Checkpoint Evaluation Viewer</h1>
  <div class="model-bar" id="model-bar"></div>
</header>

<div class="variant-toggle">
  <strong>Examples:</strong>
  <div class="model-bar" id="suffix-bar" style="flex:1"></div>
  <span style="margin-left:18px"><strong>Scale:</strong></span>
  <label><input type="checkbox" id="normalize"> Normalise per metric (0-1)</label>
</div>

<div class="variant-toggle">
  <strong>Generation:</strong>
  <div class="model-bar" id="gen-bar" style="flex:1"></div>
</div>

<div class="variant-toggle" id="eval-run-row">
  <strong>Evaluation run:</strong>
  <div class="model-bar" id="eval-run-bar" style="flex:1"></div>
</div>

<div class="main">
  <div id="chart"></div>
  <div class="sidebar" id="sidebar"></div>
</div>

<script>
const DATA = __DATA_PLACEHOLDER__;

const baselineModelName = DATA.baseline_model_name || "gpt-4o-mini (baseline)";
const modelNames = Object.keys(DATA.models).sort((a, b) => {
  if (a === baselineModelName) return 1;
  if (b === baselineModelName) return -1;
  return a.localeCompare(b);
});
const metricCatalog = DATA.metric_catalog;
const evalRuns = DATA.eval_runs || [];
const evalRunIds = evalRuns.map(r => r.id);
const suffixes = DATA.suffixes || [];
const suffixIds = suffixes.map(s => s.id);
const gens = DATA.gens || [];
const genIds = gens.map(g => g.id);
const xRange = DATA.x_range || [0, 1];

const MODEL_PALETTE = __PALETTE_PLACEHOLDER__;

const modelColors = {};
modelNames.forEach((m, i) => { modelColors[m] = MODEL_PALETTE[i % MODEL_PALETTE.length]; });

function evalRunLabel(runId) {
  const r = evalRuns.find(x => x.id === runId);
  return r ? r.label : runId;
}

// State: no models/metrics pre-selected; only 1000-example variant on by default
let selectedModels = new Set();
let selectedMetrics = new Set();
let selectedEvalRuns = new Set();
if (evalRunIds.includes("all_eval_results")) selectedEvalRuns.add("all_eval_results");
else if (evalRunIds.length) selectedEvalRuns.add(evalRunIds[0]);
let selectedSuffixes = new Set(suffixIds);
let selectedGens = new Set(genIds);
let normalizeMetrics = false;

// Helper: bulk-select buttons
function makeBulkButtons(setAll, clearAll) {
  const wrap = document.createElement("span");
  wrap.style.cssText = "display:inline-flex;gap:4px;margin-left:6px;";
  const btnAll = document.createElement("button");
  btnAll.className = "bulk-btn"; btnAll.textContent = "Select all";
  btnAll.addEventListener("click", setAll);
  const btnNone = document.createElement("button");
  btnNone.className = "bulk-btn"; btnNone.textContent = "Unselect all";
  btnNone.addEventListener("click", clearAll);
  wrap.appendChild(btnAll);
  wrap.appendChild(btnNone);
  return wrap;
}

// Build model checkboxes
const modelBar = document.getElementById("model-bar");
const modelCheckboxes = [];
modelNames.forEach(m => {
  const lbl = document.createElement("label");
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.checked = selectedModels.has(m);
  cb.addEventListener("change", () => { toggle(selectedModels, m, cb.checked); render(); });
  const sw = document.createElement("span");
  sw.className = "model-swatch";
  sw.style.background = modelColors[m];
  const shortName = m === baselineModelName ? m : m.replace(/-apptainer-fsdp$/, "");
  lbl.appendChild(cb);
  lbl.appendChild(sw);
  lbl.appendChild(document.createTextNode(" " + shortName));
  modelBar.appendChild(lbl);
  modelCheckboxes.push({ cb, key: m });
});
modelBar.appendChild(makeBulkButtons(
  () => { modelCheckboxes.forEach(({cb, key}) => { cb.checked = true; selectedModels.add(key); }); render(); },
  () => { modelCheckboxes.forEach(({cb, key}) => { cb.checked = false; selectedModels.delete(key); }); render(); }
));

function buildToggleBar(items, barId, selectedSet) {
  const bar = document.getElementById(barId);
  const checkboxes = [];
  items.forEach(item => {
    const lbl = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = selectedSet.has(item.id);
    cb.addEventListener("change", () => { toggle(selectedSet, item.id, cb.checked); render(); });
    lbl.appendChild(cb);
    lbl.appendChild(document.createTextNode(" " + item.label));
    bar.appendChild(lbl);
    checkboxes.push({ cb, key: item.id });
  });
  bar.appendChild(makeBulkButtons(
    () => { checkboxes.forEach(({cb, key}) => { cb.checked = true; selectedSet.add(key); }); render(); },
    () => { checkboxes.forEach(({cb, key}) => { cb.checked = false; selectedSet.delete(key); }); render(); }
  ));
}
buildToggleBar(suffixes, "suffix-bar", selectedSuffixes);
buildToggleBar(gens, "gen-bar", selectedGens);

// Evaluation run checkboxes (all_eval_results / eval_results_min*_max*_tokens / …)
const evalRunBar = document.getElementById("eval-run-bar");
const evalRunRow = document.getElementById("eval-run-row");
const evalRunCheckboxes = [];
if (evalRuns.length === 0) {
  evalRunRow.style.display = "none";
} else {
  evalRuns.forEach(run => {
    const lbl = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = selectedEvalRuns.has(run.id);
    cb.addEventListener("change", () => { toggle(selectedEvalRuns, run.id, cb.checked); render(); });
    lbl.appendChild(cb);
    lbl.appendChild(document.createTextNode(" " + run.label));
    evalRunBar.appendChild(lbl);
    evalRunCheckboxes.push({ cb, key: run.id });
  });
  evalRunBar.appendChild(makeBulkButtons(
    () => { evalRunCheckboxes.forEach(({cb, key}) => { cb.checked = true; selectedEvalRuns.add(key); }); render(); },
    () => { evalRunCheckboxes.forEach(({cb, key}) => { cb.checked = false; selectedEvalRuns.delete(key); }); render(); }
  ));
}

// Build metric checkboxes grouped
const sidebar = document.getElementById("sidebar");
const metricCheckboxes = [];
const groupOrder = ["Faithfulness", "Reference", "ROUGE", "Hygiene"];
const metricsByGroup = {};
for (const [key, meta] of Object.entries(metricCatalog)) {
  const g = meta.group;
  if (!metricsByGroup[g]) metricsByGroup[g] = [];
  metricsByGroup[g].push([key, meta]);
}

const metricBulkWrap = document.createElement("div");
metricBulkWrap.style.cssText = "margin-bottom:8px;";
metricBulkWrap.appendChild(makeBulkButtons(
  () => { metricCheckboxes.forEach(({cb, key}) => { cb.checked = true; selectedMetrics.add(key); }); render(); },
  () => { metricCheckboxes.forEach(({cb, key}) => { cb.checked = false; selectedMetrics.delete(key); }); render(); }
));
sidebar.appendChild(metricBulkWrap);

groupOrder.forEach(g => {
  if (!metricsByGroup[g]) return;
  const h = document.createElement("h2");
  h.textContent = g;
  sidebar.appendChild(h);
  metricsByGroup[g].forEach(([key, meta]) => {
    const lbl = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = selectedMetrics.has(key);
    cb.addEventListener("change", () => { toggle(selectedMetrics, key, cb.checked); render(); });
    const sw = document.createElement("span");
    sw.className = "metric-swatch";
    sw.style.background = meta.color;
    lbl.appendChild(cb);
    lbl.appendChild(sw);
    lbl.appendChild(document.createTextNode(" " + meta.label));
    sidebar.appendChild(lbl);
    metricCheckboxes.push({ cb, key });
  });
});

document.getElementById("normalize").addEventListener("change", e => { normalizeMetrics = e.target.checked; render(); });

function toggle(set, val, on) { if (on) set.add(val); else set.delete(val); }

function getMetricRange(metricKey) {
  let min = Infinity, max = -Infinity;
  for (const model of modelNames) {
    if (!selectedModels.has(model)) continue;
    const modelEntry = DATA.models[model] || {};
    const isBaseline = !!modelEntry.baseline || model === baselineModelName;
    const runs = modelEntry.runs || {};
    const runIdsForModel = isBaseline ? Object.keys(runs).slice(0, 1) : Array.from(selectedEvalRuns);
    for (const runId of runIdsForModel) {
      const runData = runs[runId];
      if (!runData) continue;
      const suffixData = runData.suffixes || {};
      for (const suffix of selectedSuffixes) {
        const suffixEntry = suffixData[suffix];
        if (!suffixEntry) continue;
        const genIdsForSuffix = isBaseline ? Object.keys(suffixEntry.gens || {}).slice(0, 1) : Array.from(selectedGens);
        for (const gen of genIdsForSuffix) {
          const genEntry = suffixEntry.gens[gen];
          if (!genEntry) continue;
          const d = genEntry.metrics[metricKey];
          if (!d) continue;
          for (const v of d.values) {
            if (v < min) min = v;
            if (v > max) max = v;
          }
        }
      }
    }
  }
  return { min, max };
}

function normalise(values, range) {
  const span = range.max - range.min;
  if (span === 0) return values.map(() => 0.5);
  return values.map(v => (v - range.min) / span);
}

function genNumber(genId) {
  const m = /^gen(\d+)$/.exec(genId || "");
  return m ? Number(m[1]) : 0;
}

function latestSelectedGenNumber() {
  const nums = Array.from(selectedGens).map(genNumber);
  return nums.length ? Math.max(...nums) : 0;
}

function render() {
  const traces = [];
  const latestGen = latestSelectedGenNumber();

  // Pre-compute ranges for normalisation
  const ranges = {};
  if (normalizeMetrics) {
    for (const metricKey of Object.keys(metricCatalog)) {
      if (!selectedMetrics.has(metricKey)) continue;
      ranges[metricKey] = getMetricRange(metricKey);
    }
  }

  for (const model of modelNames) {
    if (!selectedModels.has(model)) continue;
    const modelEntry = DATA.models[model] || {};
    const isBaseline = !!modelEntry.baseline || model === baselineModelName;
    const runs = modelEntry.runs || {};
    const baseColor = modelColors[model];
    const shortName = isBaseline ? model : model.replace(/-apptainer-fsdp$/, "");

    const runIdsForModel = isBaseline ? Object.keys(runs).slice(0, 1) : Array.from(selectedEvalRuns);
    for (const runId of runIdsForModel) {
      const runData = runs[runId];
      if (!runData) continue;
      const runTag = isBaseline ? "baseline" : evalRunLabel(runId);
      const suffixData = runData.suffixes || {};

      for (const suffix of selectedSuffixes) {
        const suffixEntry = suffixData[suffix];
        if (!suffixEntry) continue;
        const suffixLabel = (suffixes.find(x => x.id === suffix) || {}).label || suffix;
        const suffixOpacity = suffix === "500-examples" ? 0.55 : 1.0;
        const genIdsForSuffix = isBaseline ? Object.keys(suffixEntry.gens || {}).slice(0, 1) : Array.from(selectedGens);

        for (const gen of genIdsForSuffix) {
          const genEntry = suffixEntry.gens[gen];
          if (!genEntry) continue;
          const genOpacity = genNumber(gen) < latestGen ? 0.38 : 1.0;

          for (const metricKey of Object.keys(metricCatalog)) {
            if (!selectedMetrics.has(metricKey)) continue;
            const metric = genEntry.metrics[metricKey];
            if (!metric) continue;
            const meta = metricCatalog[metricKey];
            const metaLabel = meta.label;
            const metricDash = meta.dash || "solid";
            const range = ranges[metricKey];
            const rawValues = isBaseline ? [metric.values[0], metric.values[0]] : metric.values;
            const xVals = isBaseline ? xRange : metric.steps;
            const yVals = normalizeMetrics ? normalise(rawValues, range) : rawValues;
            const hoverText = rawValues.map((v, i) =>
              `${metaLabel}: ${v.toFixed(4)}<br>${gen} / ${suffixLabel} / ${runTag}` +
              (normalizeMetrics ? `<br>normalised: ${yVals[i].toFixed(3)}` : "")
            );
            traces.push({
              x: xVals, y: yVals,
              mode: isBaseline ? "lines" : "lines+markers",
              name: shortName + " / " + metaLabel + " [" + gen + ", " + suffixLabel + ", " + runTag + "]",
              line: { color: isBaseline ? meta.color : baseColor, width: isBaseline ? 2.0 : 2.5, dash: metricDash },
              marker: { size: isBaseline ? 0 : 6 },
              opacity: isBaseline ? 0.65 : suffixOpacity * genOpacity,
              legendgroup: model + "_" + runId + "_" + suffix + "_" + gen + "_" + metricKey,
              showlegend: true,
              text: hoverText,
              hoverinfo: "x+text+name",
            });
          }
        }
      }
    }
  }

  const yTitle = normalizeMetrics ? "Normalised Score (0 = metric min, 1 = metric max)" : "Score";
  const layout = {
    xaxis: { title: "Checkpoint Step" },
    yaxis: { title: yTitle, range: normalizeMetrics ? [-0.02, 1.02] : undefined },
    template: "plotly_white",
    legend: { font: { size: 11 }, tracegroupgap: 2 },
    margin: { t: 30, b: 50, l: 60, r: 10 },
  };

  Plotly.react("chart", traces, layout, { responsive: true });
}

render();

// Re-render on window resize so the chart fills available space
window.addEventListener("resize", () => { Plotly.Plots.resize(document.getElementById("chart")); });
</script>
</body>
</html>
"""


def generate_html(chart_data: dict) -> str:
    data_json = json.dumps(chart_data, ensure_ascii=False)
    palette_json = json.dumps(MODEL_PALETTE)
    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", data_json)
    html = html.replace("__PALETTE_PLACEHOLDER__", palette_json)
    return html


def main():
    parser = argparse.ArgumentParser(
        description="Interactive HTML viewer for checkpoint evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model_dir", type=str, nargs="+", required=True,
        help="Model directory(ies), or direct all_eval_results directory(ies).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output HTML file path (default: view_evaluation_results.html)",
    )
    parser.add_argument(
        "--reference-file",
        type=str,
        default=DEFAULT_REFERENCE_FILE,
        help=(
            "Reference/baseline evaluation JSON for horizontal baseline lines "
            f"(default: {DEFAULT_REFERENCE_FILE})"
        ),
    )
    args = parser.parse_args()

    chart_data = build_chart_data(args.model_dir, reference_file=args.reference_file)

    if not chart_data["models"]:
        print("Error: No evaluation data found.")
        sys.exit(1)

    output_path = args.output or "view_evaluation_results.html"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    html = generate_html(chart_data)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    n_models = len(chart_data["models"])
    n_metrics = len(chart_data["metric_catalog"])
    n_runs = len(chart_data.get("eval_runs") or [])
    n_gens = len(chart_data.get("gens") or [])
    print(f"\nViewer written to: {output_path}")
    print(f"  {n_models} model(s), {n_metrics} metric(s), {n_runs} evaluation run folder(s), {n_gens} generation(s)")
    print(f"  Open in a browser or use Cursor's Simple Browser (Ctrl+Shift+P > Simple Browser)")


if __name__ == "__main__":
    main()

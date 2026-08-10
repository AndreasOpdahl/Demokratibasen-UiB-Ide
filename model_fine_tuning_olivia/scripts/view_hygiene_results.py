"""
Interactive HTML viewer for checkpoint hygiene filter results.

Generates a self-contained HTML file with:
- A large interactive Plotly chart in the centre
- Model checkboxes along the top
- Generation checkboxes (gen0, gen1, gen2, ...)
- Hygiene metric checkboxes on the right

Reads hygiene filter stats from all_eval_results/ folders. Supported filenames:
- checkpoint-<N>-gen<M>-hygiene-filter-stats-1000-examples.json

Checkpoint files must include a gen<M> fragment.

Usage:
    # View a single model folder
    python view_hygiene_results.py \
        --model_dir models/gemma-2b-apptainer-fsdp

    # View multiple models for comparison
    python view_hygiene_results.py \
        --model_dir models/gemma-2b-apptainer-fsdp models/viking-13b-apptainer-fsdp

    # A direct all_eval_results folder is also accepted
    python view_hygiene_results.py \
        --model_dir models/gemma-2b-apptainer-fsdp/all_eval_results

    # Custom output path
    python view_hygiene_results.py \
        --model_dir models/*-apptainer-fsdp \
        --output hygiene_viewer.html
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from typing import Dict, List, Optional, Tuple


METRIC_GROUPS = {
    "Overall": {
        "passed_all": {"label": "Passed All", "color": "#1f77b4", "dash": "solid"},
    },
    "Core Hygiene": {
        "rep_3gram": {"label": "3-gram Repetition OK", "color": "#e377c2", "dash": "dashdot"},
        "compression_ratio": {"label": "Compression Ratio OK", "color": "#8c564b", "dash": "longdashdot"},
        "pred_chars": {"label": "Pred Chars OK", "color": "#3182bd", "dash": "dot"},
        "max_pred_ref_char_ratio": {"label": "Pred/ref Chars OK", "color": "#d62728", "dash": "dash"},
        "ends_with_punct": {"label": "Ends w/ Punct", "color": "#7f7f7f", "dash": "longdashdot"},
        "markup_ratio": {"label": "Markup Ratio OK", "color": "#bcbd22", "dash": "longdash"},
        "bad_delimiters": {"label": "No Bad Delimiters", "color": "#17becf", "dash": "dashdot"},
    },
    "Sentence Quality": {
        "punctuation_score": {"label": "Punctuation Score OK", "color": "#9467bd", "dash": "longdash"},
        "complete_sentence_ratio": {"label": "Complete Sentence Ratio OK", "color": "#2ca02c", "dash": "solid"},
        "known_word_ratio": {"label": "Known Word Ratio OK", "color": "#ff7f0e", "dash": "dot"},
        "starts_with_complete_sent": {"label": "Starts Complete", "color": "#98df8a", "dash": "dash"},
        "ends_with_complete_sent": {"label": "Ends Complete", "color": "#ff9896", "dash": "dash"},
    },
}

MODEL_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]

STATS_RE = re.compile(
    r"^(?P<prefix>.*?checkpoint-(?P<step>\d+)-)"
    r"(?P<gen>gen(?P<gen_num>\d+))-"
    r"hygiene-filter-stats-"
    r"(?P<suffix>\d+-examples)"
    r"\.json$"
)
REFERENCE_STATS_RE = re.compile(
    r"^reference-hygiene-filter-stats-"
    r"(?P<suffix>\d+-examples)"
    r"\.json$"
)
BASELINE_MODEL_NAME = "gpt-4o-mini (baseline)"


def normalise_model_dir(path: str) -> Tuple[str, str]:
    """Return (model_dir, results_dir).

    Accepts:
    - a model dir containing all_eval_results/
    - an all_eval_results/ dir directly
    - any directory that directly contains hygiene stats files (useful for tests)
    """
    clean = path.rstrip("/")
    if os.path.basename(clean) == "all_eval_results":
        return os.path.dirname(clean), clean
    all_eval = os.path.join(clean, "all_eval_results")
    if os.path.isdir(all_eval):
        return clean, all_eval
    return clean, clean


def model_display_name(model_dir: str) -> str:
    return os.path.basename(model_dir.rstrip("/"))


def read_stats_file(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: could not read {path}: {exc}", file=sys.stderr)
        return None


def load_hygiene_stats(model_dir_or_results_dir: str) -> Dict[str, Dict[str, Dict[int, dict]]]:
    """Load hygiene stats as suffix -> gen_id -> step -> stats."""
    _, results_dir = normalise_model_dir(model_dir_or_results_dir)
    if not os.path.isdir(results_dir):
        print(f"Warning: hygiene results directory not found: {results_dir}", file=sys.stderr)
        return {}

    out: Dict[str, Dict[str, Dict[int, dict]]] = {}
    for name in sorted(os.listdir(results_dir)):
        match = STATS_RE.match(name)
        if not match:
            continue
        step = int(match.group("step"))
        gen_num = int(match.group("gen_num"))
        gen_id = f"gen{gen_num}"
        suffix = match.group("suffix")
        path = os.path.join(results_dir, name)
        stats = read_stats_file(path)
        if not stats:
            continue
        out.setdefault(suffix, {}).setdefault(gen_id, {})[step] = stats
    return out


def load_reference_stats(model_dir_or_results_dir: str) -> Dict[str, List[Tuple[str, dict]]]:
    """Load reference hygiene stats as suffix -> [(path, stats), ...]."""
    _, results_dir = normalise_model_dir(model_dir_or_results_dir)
    if not os.path.isdir(results_dir):
        return {}

    out: Dict[str, List[Tuple[str, dict]]] = {}
    for name in sorted(os.listdir(results_dir)):
        match = REFERENCE_STATS_RE.match(name)
        if not match:
            continue
        path = os.path.join(results_dir, name)
        stats = read_stats_file(path)
        if not stats:
            continue
        out.setdefault(match.group("suffix"), []).append((path, stats))
    return out


def extract_metric_values(stats: dict) -> Dict[str, float]:
    """Extract passed_<metric>_pct fields from one filter-stats JSON."""
    metrics: Dict[str, float] = {}
    for key, value in stats.items():
        if not key.startswith("passed_") or not key.endswith("_pct"):
            continue
        metric_key = key[len("passed_"):-len("_pct")]
        if isinstance(value, (int, float)):
            metrics[metric_key] = float(value)
    return metrics


def reference_comparison_payload(stats: dict) -> dict:
    """Reference-stats payload used for cross-folder consistency checks.

    Ignore file-specific metadata such as input_file, but keep totals,
    thresholds, and passed_* counts/percentages.
    """
    payload = {
        "total": stats.get("total"),
        "thresholds": stats.get("thresholds"),
    }
    payload.update(
        {
            key: value
            for key, value in stats.items()
            if key.startswith("passed_")
        }
    )
    return payload


def extract_metrics(results: Dict[int, dict]) -> Dict[str, List[Tuple[int, float]]]:
    metrics: Dict[str, List[Tuple[int, float]]] = {}
    for step, stats in sorted(results.items()):
        for metric_key, value in extract_metric_values(stats).items():
            metrics.setdefault(metric_key, []).append((step, value))
    return metrics


def suffix_display_label(suffix: str) -> str:
    return suffix


def build_chart_data(model_dirs: List[str]) -> dict:
    models: Dict[str, dict] = {}
    all_metric_keys = set()
    all_suffixes = set()
    all_gens = set()
    all_steps = set()
    reference_stats_by_suffix: Dict[str, List[Tuple[str, dict]]] = {}

    for input_dir in model_dirs:
        model_dir, _ = normalise_model_dir(input_dir)
        model_name = model_display_name(model_dir)
        loaded = load_hygiene_stats(input_dir)
        suffixes_out: Dict[str, dict] = {}

        for suffix, gens in sorted(loaded.items()):
            gens_out: Dict[str, dict] = {}
            for gen_id, results in sorted(gens.items(), key=lambda item: int(item[0][3:])):
                extracted = extract_metrics(results)
                if not extracted:
                    continue
                entry = {"metrics": {}}
                for key, pts in extracted.items():
                    steps, vals = zip(*pts) if pts else ([], [])
                    entry["metrics"][key] = {"steps": list(steps), "values": list(vals)}
                    all_metric_keys.add(key)
                    all_steps.update(steps)
                gens_out[gen_id] = entry
                all_gens.add(gen_id)
            if gens_out:
                suffixes_out[suffix] = {"gens": gens_out}
                all_suffixes.add(suffix)

        if suffixes_out:
            models[model_name] = {"suffixes": suffixes_out}

        for suffix, refs in load_reference_stats(input_dir).items():
            reference_stats_by_suffix.setdefault(suffix, []).extend(refs)

    baseline_suffixes: Dict[str, dict] = {}
    for suffix, refs in sorted(reference_stats_by_suffix.items()):
        if not refs:
            continue
        reference_payloads = [reference_comparison_payload(stats) for _, stats in refs]
        first_payload = reference_payloads[0]
        mismatches = [
            path
            for (path, _), payload in zip(refs[1:], reference_payloads[1:])
            if payload != first_payload
        ]
        if mismatches:
            print(
                f"Warning: reference hygiene stats differ for suffix {suffix}; "
                f"using one random file among {len(refs)} reference files.",
                file=sys.stderr,
            )
            for path in mismatches[:5]:
                print(f"  differs: {path}", file=sys.stderr)
        selected_path, selected_stats = random.choice(refs)
        selected_metrics = extract_metric_values(selected_stats)
        if not selected_metrics:
            continue
        entry = {"metrics": {}}
        for key, value in selected_metrics.items():
            entry["metrics"][key] = {"steps": [], "values": [value], "baseline": True}
            all_metric_keys.add(key)
        baseline_suffixes[suffix] = {"gens": {"gen0": entry}}
        all_suffixes.add(suffix)
        all_gens.add("gen0")
        print(f"Reference baseline for {suffix}: {selected_path}", file=sys.stderr)

    if baseline_suffixes:
        models[BASELINE_MODEL_NAME] = {"suffixes": baseline_suffixes, "baseline": True}

    metric_catalog = {}
    for group_name, group_metrics in METRIC_GROUPS.items():
        for key, meta in group_metrics.items():
            if key in all_metric_keys:
                metric_catalog[key] = {**meta, "group": group_name}

    suffixes = [
        {"id": s, "label": suffix_display_label(s)}
        for s in sorted(all_suffixes, key=suffix_display_label)
    ]
    gens = [
        {"id": g, "label": g}
        for g in sorted(all_gens, key=lambda x: int(x[3:]))
    ]

    return {
        "models": models,
        "metric_catalog": metric_catalog,
        "suffixes": suffixes,
        "gens": gens,
        "baseline_model_name": BASELINE_MODEL_NAME,
        "x_range": [min(all_steps), max(all_steps)] if all_steps else [0, 1],
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Checkpoint Hygiene Viewer</title>
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
  .sidebar { width: 230px; min-width: 190px; padding: 12px 14px; background: #fff;
             border-left: 1px solid #ddd; overflow-y: auto; }
  .sidebar h2 { font-size: 13px; text-transform: uppercase; color: #888; margin: 10px 0 4px; }
  .sidebar h2:first-child { margin-top: 0; }
  .sidebar label { display: flex; align-items: center; gap: 5px; font-size: 13px;
                   cursor: pointer; padding: 2px 0; }
  .sidebar .metric-swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
  .toggle-row { padding: 4px 12px; background: #fff; border-bottom: 1px solid #ddd;
                display: flex; gap: 16px; align-items: center; font-size: 13px; }
  .toggle-row label { cursor: pointer; display: flex; align-items: center; gap: 4px; }
  .bulk-btn { font-size: 11px; cursor: pointer; background: #eee; border: 1px solid #ccc;
              border-radius: 3px; padding: 1px 6px; color: #555; }
  .bulk-btn:hover { background: #ddd; }
</style>
</head>
<body>

<header>
  <h1>Checkpoint Hygiene Viewer</h1>
  <div class="model-bar" id="model-bar"></div>
</header>

<div class="toggle-row" id="suffix-row">
  <strong>Example set:</strong>
  <div class="model-bar" id="suffix-bar" style="flex:1"></div>
</div>

<div class="toggle-row" id="gen-row">
  <strong>Generation:</strong>
  <div class="model-bar" id="gen-bar" style="flex:1"></div>
  <span style="margin-left:18px"><strong>Scale:</strong></span>
  <label><input type="checkbox" id="normalize"> Normalise per metric (0-1)</label>
</div>

<div class="main">
  <div id="chart"></div>
  <div class="sidebar" id="sidebar"></div>
</div>

<script>
const DATA = __DATA_PLACEHOLDER__;
const MODEL_PALETTE = __PALETTE_PLACEHOLDER__;

const baselineModelName = DATA.baseline_model_name || "Baseline (gpt-4o-mini)";
const modelNames = Object.keys(DATA.models).sort((a, b) => {
  if (a === baselineModelName) return 1;
  if (b === baselineModelName) return -1;
  return a.localeCompare(b);
});
const metricCatalog = DATA.metric_catalog;
const suffixes = DATA.suffixes || [];
const gens = DATA.gens || [];
const xRange = DATA.x_range || [0, 1];

const modelColors = {};
modelNames.forEach((m, i) => { modelColors[m] = MODEL_PALETTE[i % MODEL_PALETTE.length]; });

let selectedModels = new Set();
let selectedMetrics = new Set();
let selectedSuffixes = new Set();
let selectedGens = new Set();
if (suffixes.length) selectedSuffixes.add(suffixes[0].id);
if (gens.find(g => g.id === "gen0")) selectedGens.add("gen0");
else if (gens.length) selectedGens.add(gens[0].id);
let normalizeMetrics = false;

function toggle(set, val, on) { if (on) set.add(val); else set.delete(val); }

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
  lbl.appendChild(cb);
  lbl.appendChild(sw);
  const labelText = m === baselineModelName ? m : m.replace(/-apptainer-fsdp$/, "");
  lbl.appendChild(document.createTextNode(" " + labelText));
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

const sidebar = document.getElementById("sidebar");
const metricCheckboxes = [];
const groupOrder = ["Overall", "Core Hygiene", "Sentence Quality"];
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

document.getElementById("normalize").addEventListener("change", e => {
  normalizeMetrics = e.target.checked;
  render();
});

function getMetricRange(metricKey) {
  let min = Infinity, max = -Infinity;
  for (const model of modelNames) {
    if (!selectedModels.has(model)) continue;
    const modelEntry = DATA.models[model] || {};
    const isBaseline = !!modelEntry.baseline || model === baselineModelName;
    const suffixData = modelEntry.suffixes || {};
    for (const suffix of selectedSuffixes) {
      const suffixEntry = suffixData[suffix];
      if (!suffixEntry) continue;
      const genIdsForSuffix = isBaseline ? Object.keys(suffixEntry.gens || {}).slice(0, 1) : Array.from(selectedGens);
      for (const gen of genIdsForSuffix) {
        const genEntry = suffixEntry.gens[gen];
        if (!genEntry || !genEntry.metrics[metricKey]) continue;
        for (const v of genEntry.metrics[metricKey].values) {
          if (v < min) min = v;
          if (v > max) max = v;
        }
      }
    }
  }
  return { min, max };
}

function normalise(values, range) {
  const span = range.max - range.min;
  if (!Number.isFinite(span) || span === 0) return values.map(() => 0.5);
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
  const ranges = {};
  if (normalizeMetrics) {
    for (const metricKey of Object.keys(metricCatalog)) {
      if (selectedMetrics.has(metricKey)) ranges[metricKey] = getMetricRange(metricKey);
    }
  }

  for (const model of modelNames) {
    if (!selectedModels.has(model)) continue;
    const modelEntry = DATA.models[model] || {};
    const isBaseline = !!modelEntry.baseline || model === baselineModelName;
    const suffixData = modelEntry.suffixes || {};
    const shortName = isBaseline ? model : model.replace(/-apptainer-fsdp$/, "");
    const baseColor = modelColors[model];

    for (const suffix of selectedSuffixes) {
      const suffixEntry = suffixData[suffix];
      if (!suffixEntry) continue;
      const suffixLabel = (suffixes.find(x => x.id === suffix) || {}).label || suffix;

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
          const rawValues = isBaseline ? [metric.values[0], metric.values[0]] : metric.values;
          const xVals = isBaseline ? xRange : metric.steps;
          const yVals = normalizeMetrics ? normalise(rawValues, ranges[metricKey]) : rawValues;
          const hoverText = rawValues.map((v, i) =>
            `${meta.label}: ${v.toFixed(2)}%<br>${gen} / ${suffixLabel}` +
            (normalizeMetrics ? `<br>normalised: ${yVals[i].toFixed(3)}` : "")
          );
          traces.push({
            x: xVals,
            y: yVals,
            mode: isBaseline ? "lines" : "lines+markers",
            name: shortName + " / " + meta.label + " [" + gen + ", " + suffixLabel + "]",
            line: { color: isBaseline ? meta.color : baseColor, width: isBaseline ? 2.0 : 2.3, dash: meta.dash || "solid" },
            marker: { size: isBaseline ? 0 : 6 },
            opacity: isBaseline ? 0.65 : genOpacity,
            legendgroup: model + "_" + suffix + "_" + gen + "_" + metricKey,
            showlegend: true,
            text: hoverText,
            hoverinfo: "x+text+name",
          });
        }
      }
    }
  }

  const yTitle = normalizeMetrics ? "Normalised Score (0 = metric min, 1 = metric max)" : "Passed Examples (%)";
  const layout = {
    xaxis: { title: "Checkpoint Step" },
    yaxis: { title: yTitle, range: normalizeMetrics ? [-0.02, 1.02] : [0, 100] },
    template: "plotly_white",
    legend: { font: { size: 11 }, tracegroupgap: 2 },
    margin: { t: 30, b: 50, l: 60, r: 10 },
  };
  Plotly.react("chart", traces, layout, { responsive: true });
}

render();
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive HTML viewer for checkpoint hygiene filter results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model_dir", type=str, nargs="+", required=True,
        help="Model directory(ies), or direct all_eval_results directory(ies).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output HTML file path (default: view_hygiene_results.html).",
    )
    args = parser.parse_args()

    chart_data = build_chart_data(args.model_dir)
    if not chart_data["models"]:
        print("Error: No hygiene filter stats found.")
        sys.exit(1)

    output_path = args.output or "view_hygiene_results.html"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    html = generate_html(chart_data)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    n_models = len(chart_data["models"])
    n_metrics = len(chart_data["metric_catalog"])
    n_gens = len(chart_data.get("gens") or [])
    print(f"\nViewer written to: {output_path}")
    print(f"  {n_models} model(s), {n_metrics} metric(s), {n_gens} generation(s)")
    print("  Open in a browser or use Cursor's Simple Browser (Ctrl+Shift+P > Simple Browser)")


if __name__ == "__main__":
    main()

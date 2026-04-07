"""
Interactive HTML viewer for checkpoint evaluation results.

Generates a self-contained HTML file with:
- A large interactive Plotly chart in the centre
- Model checkboxes along the top
- Evaluation run checkboxes (``all_eval_results/`` vs ``eval_results_min*_max*_tokens/``)
- Metric checkboxes on the right

Reuses data loading from visualise_checkpoint_results.py.

Usage:
    # View a single model
    python view_checkpoint_results.py \
        --model_dir models/gemma-2-9b-apptainer-fsdp

    # View multiple models for comparison
    python view_checkpoint_results.py \
        --model_dir models/gemma-2-9b-apptainer-fsdp models/normistral-7b-apptainer-fsdp

    # Custom output path
    python view_checkpoint_results.py \
        --model_dir models/gemma-2-9b-apptainer-fsdp \
        --output viewer.html
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from visualise_checkpoint_results import (
    discover_all_eval_run_ids,
    eval_run_display_label,
    load_checkpoint_results,
    load_checkpoint_results_1000,
    extract_metrics,
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


def build_chart_data(model_dirs: List[str]) -> dict:
    """Load all models and evaluation runs; return JSON for the HTML viewer.

    Each model may have several folders: ``all_eval_results/`` and
    ``eval_results_min*_*max*_tokens/`` (same JSON layout). The viewer exposes
    one checkbox per distinct folder name found across the given model dirs.
    """
    eval_run_ids = discover_all_eval_run_ids(model_dirs)
    if not eval_run_ids:
        eval_run_ids = ["all_eval_results"]

    models: Dict[str, dict] = {}
    all_metric_keys = set()

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir.rstrip("/"))
        runs_out: Dict[str, dict] = {}

        for run_id in eval_run_ids:
            results_dir = os.path.join(model_dir, run_id)
            if not os.path.isdir(results_dir):
                continue

            r500 = load_checkpoint_results(model_dir, run_id)
            r1000 = load_checkpoint_results_1000(model_dir, run_id)
            if not r500 and not r1000:
                continue

            m500 = extract_metrics(r500) if r500 else {}
            m1000 = extract_metrics(r1000) if r1000 else {}

            entry = {"metrics_500": {}, "metrics_1000": {}}
            for key, pts in m500.items():
                steps, vals = zip(*pts) if pts else ([], [])
                entry["metrics_500"][key] = {"steps": list(steps), "values": list(vals)}
                all_metric_keys.add(key)
            for key, pts in m1000.items():
                steps, vals = zip(*pts) if pts else ([], [])
                entry["metrics_1000"][key] = {"steps": list(steps), "values": list(vals)}
                all_metric_keys.add(key)

            runs_out[run_id] = entry

        if runs_out:
            models[model_name] = {"runs": runs_out}

    present_run_ids = set()
    for mentry in models.values():
        present_run_ids.update(mentry["runs"].keys())

    eval_runs = [
        {"id": rid, "label": eval_run_display_label(rid)}
        for rid in sorted(present_run_ids, key=lambda n: (0 if n == "all_eval_results" else 1, n))
    ]

    metric_catalog = {}
    for group_name, group_metrics in METRIC_GROUPS.items():
        for key, meta in group_metrics.items():
            if key in all_metric_keys:
                metric_catalog[key] = {**meta, "group": group_name}

    return {"models": models, "metric_catalog": metric_catalog, "eval_runs": eval_runs}


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
  <strong>Data variant:</strong>
  <label><input type="checkbox" id="show500"> 500-example (faded)</label>
  <label><input type="checkbox" id="show1000" checked> 1000-example (solid)</label>
  <span style="margin-left:18px"><strong>Scale:</strong></span>
  <label><input type="checkbox" id="normalize"> Normalise per metric (0-1)</label>
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

const modelNames = Object.keys(DATA.models).sort();
const metricCatalog = DATA.metric_catalog;
const evalRuns = DATA.eval_runs || [];
const evalRunIds = evalRuns.map(r => r.id);

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
let show500 = false;
let show1000 = true;
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
  const shortName = m.replace(/-apptainer-fsdp$/, "");
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

// Variant toggles
document.getElementById("show500").addEventListener("change", e => { show500 = e.target.checked; render(); });
document.getElementById("show1000").addEventListener("change", e => { show1000 = e.target.checked; render(); });
document.getElementById("normalize").addEventListener("change", e => { normalizeMetrics = e.target.checked; render(); });

function toggle(set, val, on) { if (on) set.add(val); else set.delete(val); }

function getMetricRange(metricKey) {
  let min = Infinity, max = -Infinity;
  for (const model of modelNames) {
    if (!selectedModels.has(model)) continue;
    const runs = DATA.models[model].runs || {};
    for (const runId of selectedEvalRuns) {
      const runData = runs[runId];
      if (!runData) continue;
      for (const variant of ["metrics_500", "metrics_1000"]) {
        const d = runData[variant][metricKey];
        if (!d) continue;
        for (const v of d.values) {
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
  if (span === 0) return values.map(() => 0.5);
  return values.map(v => (v - range.min) / span);
}

function render() {
  const traces = [];

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
    const runs = DATA.models[model].runs || {};
    const baseColor = modelColors[model];
    const shortName = model.replace(/-apptainer-fsdp$/, "");

    for (const runId of selectedEvalRuns) {
      const runData = runs[runId];
      if (!runData) continue;
      const runTag = evalRunLabel(runId);

      for (const metricKey of Object.keys(metricCatalog)) {
        if (!selectedMetrics.has(metricKey)) continue;
        const meta = metricCatalog[metricKey];
        const metaLabel = meta.label;
        const metricDash = meta.dash || "solid";
        const range = ranges[metricKey];

        // 500-example trace (faded) -- always dotted regardless of run
        if (show500 && runData.metrics_500[metricKey]) {
          const d = runData.metrics_500[metricKey];
          const yVals = normalizeMetrics ? normalise(d.values, range) : d.values;
          const hoverText = normalizeMetrics
            ? d.values.map((v, i) => `${metaLabel}: ${v.toFixed(4)}<br>normalised: ${yVals[i].toFixed(3)}`)
            : undefined;
          traces.push({
            x: d.steps, y: yVals,
            mode: "lines+markers",
            name: shortName + " / " + metaLabel + " (500) [" + runTag + "]",
            line: { color: baseColor, width: 1.2, dash: "dot" },
            marker: { size: 4 },
            opacity: 0.30,
            legendgroup: model + "_" + runId + "_" + metricKey,
            showlegend: true,
            ...(hoverText ? { text: hoverText, hoverinfo: "x+text+name" } : {}),
          });
        }

        // 1000-example trace (solid) — dash style follows metric type (catalog)
        if (show1000 && runData.metrics_1000[metricKey]) {
          const d = runData.metrics_1000[metricKey];
          const yVals = normalizeMetrics ? normalise(d.values, range) : d.values;
          const hoverText = normalizeMetrics
            ? d.values.map((v, i) => `${metaLabel}: ${v.toFixed(4)}<br>normalised: ${yVals[i].toFixed(3)}`)
            : undefined;
          traces.push({
            x: d.steps, y: yVals,
            mode: "lines+markers",
            name: shortName + " / " + metaLabel + " (1000) [" + runTag + "]",
            line: { color: baseColor, width: 2.5, dash: metricDash },
            marker: { size: 6 },
            opacity: 1.0,
            legendgroup: model + "_" + runId + "_" + metricKey,
            showlegend: true,
            ...(hoverText ? { text: hoverText, hoverinfo: "x+text+name" } : {}),
          });
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
        help="Model directory(ies); each may contain all_eval_results/ and eval_results_min*_max*_tokens/",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output HTML file path (default: view_checkpoint_results.html)",
    )
    args = parser.parse_args()

    chart_data = build_chart_data(args.model_dir)

    if not chart_data["models"]:
        print("Error: No evaluation data found.")
        sys.exit(1)

    output_path = args.output or "view_checkpoint_results.html"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    html = generate_html(chart_data)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    n_models = len(chart_data["models"])
    n_metrics = len(chart_data["metric_catalog"])
    n_runs = len(chart_data.get("eval_runs") or [])
    print(f"\nViewer written to: {output_path}")
    print(f"  {n_models} model(s), {n_metrics} metric(s), {n_runs} evaluation run folder(s)")
    print(f"  Open in a browser or use Cursor's Simple Browser (Ctrl+Shift+P > Simple Browser)")


if __name__ == "__main__":
    main()

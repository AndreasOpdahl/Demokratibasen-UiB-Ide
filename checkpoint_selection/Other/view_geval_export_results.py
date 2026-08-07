#!/usr/bin/env python3
"""
Interactive HTML viewer for **this repo's** G-Eval **export** results.

**Metrics (pick one, like the notebook):**

- **Bradley–Terry θ** — from ``tables/bradley_terry_theta__<dimension>.csv``
- **Win rate** — per-judge columns from ``reports/results_summary.md`` (§1 Pairwise win rates)

**Models:** By default, discovers every subfolder of ``DATA_ROOT/eval/`` that contains ``*.jsonl`` and has a
matching export under ``.deepeval/geval_exports/<same_name>/``. The folder named ``25`` is skipped.
If you only have one such eval folder (e.g. only ``llama-2-13b``) but **many** export trees under
``.deepeval/geval_exports/``, use ``--all_export_leaves`` to list **every** export leaf there (not
tied to ``DATA_ROOT/eval``).

**Mean (selected judges):** When the checkbox is on, adds a curve that averages per-judge values
at each checkpoint **only over judges that are checked** in the sidebar (it updates as you change
the selection). The curve is drawn only when at least two checked judges have data for that panel
(so it is not redundant with a single judge line).

Usage (from repo root; auto-pick all eval models with exports):

    python Other/view_geval_export_results.py -o .deepeval/geval_exports/images/geval_viewer.html

All export subfolders under ``.deepeval/geval_exports/`` (ignores e.g. ``images``):

    python Other/view_geval_export_results.py --all_export_leaves -o geval_export_viewer.html

Manual leaves still supported:

    python Other/view_geval_export_results.py --export_leaf llama-2-13b -o viewer.html
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parent
_DEFAULT_EXPORT_ROOT = REPO_ROOT / ".deepeval" / "geval_exports"
# Data moved out of the repo (2026-06) into the shared OneDrive folder. Override with
# CHECKPOINT_SELECTION_DATA_DIR if your OneDrive root or the dataset snapshot name differs.
DATA_ROOT = Path(
    os.environ.get("CHECKPOINT_SELECTION_DATA_DIR")
    or (
        Path(os.environ.get("ONEDRIVE", str(Path.home() / "OneDrive")))
        / "Shared"
        / "Demokratibasen-UiB-Ide"
        / "EvaluationDatasets"
        / "CheckpointSelection"
        / "Data_202606"
    )
)
_DATA_EVAL_ROOT = DATA_ROOT / "eval"
_IGNORE_EVAL_DIR_NAMES = frozenset({"25"})
# Subdirs of export_root that are not summarization-model export trees (plots, scratch, etc.).
_IGNORE_EXPORT_LEAF_NAMES = frozenset({"images"})

_MEAN_JUDGE_KEY = "__MEAN_OVER_JUDGES__"
_MEAN_JUDGE_LABEL = "Mean (selected judges)"

_DEFAULT_DIMENSIONS: Tuple[str, ...] = (
    "relevance",
    "consistency",
    "newsworthiness",
    "hygiene",
)

_LEAF_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
]

# Plot line colors: one color per G-Eval dimension (tab10-like).
_DIMENSION_LINE_COLORS: Tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
)


def _dimension_color_map(dimensions_sorted: List[str]) -> Dict[str, str]:
    return {
        d: _DIMENSION_LINE_COLORS[i % len(_DIMENSION_LINE_COLORS)]
        for i, d in enumerate(dimensions_sorted)
    }


def _checkpoint_step(model_id: str) -> Optional[int]:
    mm = re.search(r"checkpoint-(\d+)-", str(model_id))
    return int(mm.group(1)) if mm else None


def _checkpoint_generation(model_id: str) -> str:
    mid = str(model_id).lower()
    return "gen1" if "-gen1-" in mid else "legacy"


def _model_display_name(model_id: str) -> Optional[str]:
    mid = str(model_id).strip()
    if "__checkpoint-" in mid:
        base = mid.split("__checkpoint-", 1)[0].strip()
        return base or None
    if mid:
        return mid
    return None


def _model_sort_key(model_id: str, row_index: int) -> int:
    step = _checkpoint_step(model_id)
    if step is not None:
        return step
    # Keep non-checkpoint candidates (reference summaries, prompt variants,
    # etc.) visible after numeric checkpoints, preserving table/model order.
    return 10**9 + row_index


def _resolve_leaf_path(arg: str, export_root: Path) -> Path:
    p = Path(arg)
    if p.is_dir():
        return p.resolve()
    cand = export_root / arg
    if cand.is_dir():
        return cand.resolve()
    raise FileNotFoundError(f"Not a directory and not under export root: {arg!r} (tried {cand})")


def _read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            return [], []
        rows = [dict(row) for row in r]
        return list(r.fieldnames), rows


def _load_theta_curves(leaf_path: Path, dimensions: Tuple[str, ...]) -> Dict[str, Dict[str, Dict[str, List[Any]]]]:
    tables = leaf_path / "tables"
    out: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for dim in dimensions:
        csvp = tables / f"bradley_terry_theta__{dim}.csv"
        if not csvp.is_file():
            continue
        headers, rows = _read_csv_rows(csvp)
        if "model" not in headers:
            continue
        judge_cols = [c for c in headers if str(c).endswith("_theta")]
        dim_j: Dict[str, Dict[str, List[Any]]] = {}
        for jc in judge_cols:
            judge_id = str(jc)[: -len("_theta")]
            pts: List[Tuple[int, float, str, Optional[str]]] = []
            for row_idx, row in enumerate(rows):
                mid = str(row.get("model", "")).strip()
                if not mid:
                    continue
                st = _model_sort_key(mid, row_idx)
                gen = _checkpoint_generation(mid)
                model_label = _model_display_name(mid)
                raw = (row.get(jc) or "").strip()
                if raw == "":
                    continue
                try:
                    v = float(raw)
                except ValueError:
                    continue
                if v != v:
                    continue
                pts.append((st, v, gen, model_label))
            pts.sort(key=lambda t: t[0])
            if pts:
                dim_j[judge_id] = {
                    "steps": [a for a, _, _, _ in pts],
                    "values": [b for _, b, _, _ in pts],
                    "generations": [g for _, _, g, _ in pts],
                    "labels": [lb for _, _, _, lb in pts],
                }
        if dim_j:
            out[dim] = dim_j
    return out


def _parse_win_rates_from_summary(
    md_path: Path, dimensions: Tuple[str, ...]
) -> Dict[str, Dict[str, Dict[str, List[Any]]]]:
    """dimension -> judge_id -> {steps, values} for checkpoint-* rows (from markdown tables)."""
    try:
        text = md_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    m = re.search(r"## 1\. Pairwise win rates\s*\n(.*?)(?=\n## 2\.)", text, re.S)
    if not m:
        return {}
    sec = m.group(1)
    out: Dict[str, Dict[str, Dict[str, List[Any]]]] = {}
    for dim in dimensions:
        dim_title = dim.capitalize()
        block_m = re.search(rf"### {re.escape(dim_title)}\s*\n+(.*?)(?=\n### |\Z)", sec, re.S)
        if not block_m:
            continue
        block = block_m.group(1).strip()
        lines = [ln for ln in block.splitlines() if ln.strip().startswith("|")]
        if len(lines) < 2:
            continue
        header = [c.strip() for c in lines[0].strip().strip("|").split("|")]
        judge_cols = [h for h in header if h.endswith("_win_rate") and h != "model"]
        if "model" not in header or not judge_cols:
            continue
        per_judge: Dict[str, List[Tuple[int, float, str, Optional[str]]]] = {
            jc[: -len("_win_rate")]: [] for jc in judge_cols
        }
        row_idx = 0
        for ln in lines[1:]:
            if re.match(r"\|\s*---", ln):
                continue
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if len(cells) != len(header):
                continue
            row = dict(zip(header, cells))
            mid = str(row.get("model", "")).strip()
            if not mid:
                continue
            st = _model_sort_key(mid, row_idx)
            row_idx += 1
            gen = _checkpoint_generation(mid)
            model_label = _model_display_name(mid)
            for jc in judge_cols:
                judge_id = jc[: -len("_win_rate")]
                raw = (row.get(jc) or "").strip()
                if raw == "":
                    continue
                try:
                    v = float(raw)
                except ValueError:
                    continue
                if v != v:
                    continue
                per_judge[judge_id].append((st, v, gen, model_label))
        dim_j: Dict[str, Dict[str, List[Any]]] = {}
        for jid, pts in per_judge.items():
            if not pts:
                continue
            pts.sort(key=lambda t: t[0])
            dim_j[jid] = {
                "steps": [a for a, _, _, _ in pts],
                "values": [b for _, b, _, _ in pts],
                "generations": [g for _, _, g, _ in pts],
                "labels": [lb for _, _, _, lb in pts],
            }
        if dim_j:
            out[dim] = dim_j
    return out


def discover_export_leaves_from_data_eval(
    eval_root: Path = _DATA_EVAL_ROOT,
    export_root: Path = _DEFAULT_EXPORT_ROOT,
    ignore_names: frozenset[str] = _IGNORE_EVAL_DIR_NAMES,
) -> List[Path]:
    """One export path per eval model subfolder (skips ``ignore_names``), if export exists."""
    if not eval_root.is_dir():
        return []
    out: List[Path] = []
    for child in sorted(eval_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in ignore_names:
            continue
        if not any(child.glob("*.jsonl")):
            continue
        exp = export_root / child.name
        if exp.is_dir() and (exp / "tables").is_dir():
            out.append(exp.resolve())
    return out


def _looks_like_geval_export_leaf(path: Path) -> bool:
    """True if ``path`` has tables (and optional BT CSV or results summary) like a pipeline export."""
    tables = path / "tables"
    if not tables.is_dir():
        return False
    if (path / "reports" / "results_summary.md").is_file():
        return True
    return any(tables.glob("bradley_terry_theta__*.csv"))


def _document_count_for_export_leaf(lp: Path) -> Optional[int]:
    """Distinct ``doc_id`` count for this export (subset used in pairwise G-Eval).

    Prefer ``reports/results_summary.md`` (same line as :func:`pairwise_eval.io_export` preamble);
    fall back to ``json/pairs_table.json`` distinct ``doc_id`` values.
    """
    summary = lp / "reports" / "results_summary.md"
    if summary.is_file():
        text = summary.read_text(encoding="utf-8")
        m = re.search(r"\*\*Documents in subset:\*\*\s*(\d+)\s+distinct", text)
        if m:
            return int(m.group(1))
    pairs_json = lp / "json" / "pairs_table.json"
    if pairs_json.is_file():
        try:
            rows = json.loads(pairs_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(rows, list):
            return None
        doc_ids = {r.get("doc_id") for r in rows if isinstance(r, dict)}
        doc_ids.discard(None)
        return len(doc_ids) if doc_ids else None
    return None


def discover_export_leaves_under_export_root(
    export_root: Path,
    ignore_names: frozenset[str] = _IGNORE_EXPORT_LEAF_NAMES,
) -> List[Path]:
    """Every immediate subdirectory of ``export_root`` that looks like a G-Eval export (has ``tables/``).

    Use this when exports exist for many models under ``.deepeval/geval_exports/`` but ``DATA_ROOT/eval``
    only mirrors one folder (so :func:`discover_export_leaves_from_data_eval` would return a single leaf).
    """
    if not export_root.is_dir():
        return []
    out: List[Path] = []
    for child in sorted(export_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            continue
        if child.name in ignore_names:
            continue
        if not _looks_like_geval_export_leaf(child):
            continue
        out.append(child.resolve())
    return out


def build_chart_data(
    leaf_paths: List[Path],
    dimensions: Tuple[str, ...] = _DEFAULT_DIMENSIONS,
) -> Dict[str, Any]:
    leaves: Dict[str, Any] = {}
    all_dims: set[str] = set()
    all_judges: set[str] = set()

    for lp in leaf_paths:
        name = lp.name
        theta = _load_theta_curves(lp, dimensions)
        summary = lp / "reports" / "results_summary.md"
        win_rate = _parse_win_rates_from_summary(summary, dimensions) if summary.is_file() else {}

        for d, jd in theta.items():
            all_dims.add(d)
            for k in jd:
                if k != _MEAN_JUDGE_KEY:
                    all_judges.add(k)
        for d, jd in win_rate.items():
            all_dims.add(d)
            for k in jd:
                if k != _MEAN_JUDGE_KEY:
                    all_judges.add(k)

        n_docs = _document_count_for_export_leaf(lp)
        leaves[name] = {
            "theta": theta,
            "win_rate": win_rate,
            "path": str(lp),
            "document_count": n_docs,
        }

    judge_list = sorted(all_judges)
    dims_sorted = sorted(
        all_dims, key=lambda x: (_DEFAULT_DIMENSIONS.index(x) if x in _DEFAULT_DIMENSIONS else 99, x)
    )
    dim_colors = _dimension_color_map(dims_sorted)
    leaf_colors = {
        p.name: _LEAF_PALETTE[i % len(_LEAF_PALETTE)] for i, p in enumerate(leaf_paths)
    }
    return {
        "leaves": leaves,
        "leaf_order": [p.name for p in leaf_paths],
        "dimensions": dims_sorted,
        "judges": judge_list,
        "mean_judge_key": _MEAN_JUDGE_KEY,
        "mean_judge_label": _MEAN_JUDGE_LABEL,
        "dimension_colors": dim_colors,
        "leaf_colors": leaf_colors,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>G-Eval export viewer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         display: flex; flex-direction: column; height: 100vh; background: #f6f7fb; }
  header { padding: 10px 18px; background: #fff; border-bottom: 1px solid #ddd;
           display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  header h1 { font-size: 16px; }
  .bar { display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center; }
  .bar label { font-size: 13px; cursor: pointer; display: flex; align-items: center; gap: 4px; }
  .sw { width: 11px; height: 11px; border-radius: 2px; display: inline-block; }
  .row2 { padding: 8px 18px; background: #fff; border-bottom: 1px solid #e5e7eb;
          display: flex; flex-wrap: wrap; gap: 16px; align-items: center; font-size: 13px; }
  .main { display: flex; flex: 1; overflow: hidden; }
  #chart { flex: 1; min-width: 0; min-height: 360px; }
  .sidebar { width: 230px; min-width: 200px; padding: 12px 14px; background: #fff;
             border-left: 1px solid #ddd; overflow-y: auto; }
  .sidebar h2 { font-size: 12px; text-transform: uppercase; color: #64748b; margin: 12px 0 6px; }
  .sidebar h2:first-child { margin-top: 0; }
  .sidebar label { display: flex; align-items: center; gap: 5px; font-size: 12px;
                    cursor: pointer; padding: 2px 0; }
  .bulk-btn { font-size: 11px; cursor: pointer; background: #eef2ff; border: 1px solid #c7d2fe;
              border-radius: 4px; padding: 2px 7px; color: #3730a3; margin-left: 6px; }
  .bulk-btn:hover { background: #e0e7ff; }
</style>
</head>
<body>

<header>
  <h1 id="hdr-title">G-Eval exports</h1>
  <div class="bar" id="leaf-bar"></div>
</header>

<div class="row2">
  <strong>Metric:</strong>
  <label><input type="radio" name="metric" value="bt"> Bradley–Terry θ</label>
  <label><input type="radio" name="metric" value="wr" checked> Win rate (per judge)</label>
  <strong style="margin-left:10px;">Checkpoints:</strong>
  <label><input type="checkbox" id="legacy-mode" checked> Legacy</label>
  <label><input type="checkbox" id="gen1-mode" checked> Gen1</label>
</div>

<div class="main">
  <div id="chart"></div>
  <div class="sidebar" id="sidebar"></div>
</div>

<script>
const DATA = __DATA_PLACEHOLDER__;
const MEAN_KEY = DATA.mean_judge_key;
const MEAN_LAB = DATA.mean_judge_label;

const leafNames = DATA.leaf_order.slice();
const leafColors = DATA.leaf_colors || {};
const dimColors = DATA.dimension_colors || {};

let selectedLeaves = new Set();  // none selected by default; user picks models to plot
let selectedDims = new Set(DATA.dimensions);
let selectedJudges = new Set(DATA.judges);
let showMeanJudges = false;
let showLegend = false;
let metricMode = "wr";
let showLegacy = true;
let showGen1 = true;

function colorForDim(dim) {
  return dimColors[dim] || "#334155";
}

/** Stable dash per judge (same judge = same style across dimensions / leaves). */
const JUDGE_DASH_CYCLE = ["solid", "dot", "dash", "dashdot"];
function dashForJudge(judgeId) {
  const order = DATA.judges || [];
  const i = order.indexOf(judgeId);
  if (i < 0) return "solid";
  return JUDGE_DASH_CYCLE[i % JUDGE_DASH_CYCLE.length];
}

/** Marker shape per judge so curves differ even when dimension color matches. */
const JUDGE_MARKER_CYCLE = ["circle", "square", "diamond", "cross"];
function markerForJudge(judgeId) {
  const order = DATA.judges || [];
  const i = order.indexOf(judgeId);
  if (i < 0) return "circle";
  return JUDGE_MARKER_CYCLE[i % JUDGE_MARKER_CYCLE.length];
}

function makeBulk(setAll, clearAll) {
  const w = document.createElement("span");
  w.style.cssText = "display:inline-flex;gap:4px;margin-left:4px;";
  const a = document.createElement("button");
  a.className = "bulk-btn"; a.textContent = "All"; a.onclick = setAll;
  const z = document.createElement("button");
  z.className = "bulk-btn"; z.textContent = "None"; z.onclick = clearAll;
  w.appendChild(a); w.appendChild(z);
  return w;
}

const leafBar = document.getElementById("leaf-bar");
const leafCbs = [];
leafNames.forEach(name => {
  const lbl = document.createElement("label");
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.checked = selectedLeaves.has(name);
  cb.onchange = () => { if (cb.checked) selectedLeaves.add(name); else selectedLeaves.delete(name); render(); };
  const sw = document.createElement("span"); sw.className = "sw"; sw.style.background = leafColors[name];
  lbl.appendChild(cb); lbl.appendChild(sw);
  const leafRec = DATA.leaves[name];
  const nd = leafRec && leafRec.document_count != null ? leafRec.document_count : null;
  const suffix = (typeof nd === "number" && Number.isFinite(nd)) ? ` (${nd} docs)` : "";
  lbl.appendChild(document.createTextNode(" " + name + suffix));
  leafBar.appendChild(lbl);
  leafCbs.push({cb, name});
});
leafBar.appendChild(makeBulk(
  () => { leafCbs.forEach(({cb, name}) => { cb.checked = true; selectedLeaves.add(name); }); render(); },
  () => { leafCbs.forEach(({cb, name}) => { cb.checked = false; selectedLeaves.delete(name); }); render(); }
));

const sidebar = document.getElementById("sidebar");

sidebar.appendChild(document.createElement("h2")).textContent = "Dimensions";
const dimCbs = [];
DATA.dimensions.forEach(dim => {
  const lbl = document.createElement("label");
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.checked = selectedDims.has(dim);
  cb.onchange = () => { if (cb.checked) selectedDims.add(dim); else selectedDims.delete(dim); render(); };
  const sw = document.createElement("span");
  sw.className = "sw";
  sw.style.background = colorForDim(dim);
  lbl.appendChild(cb);
  lbl.appendChild(sw);
  lbl.appendChild(document.createTextNode(" " + dim));
  sidebar.appendChild(lbl);
  dimCbs.push({cb, dim});
});
sidebar.appendChild(makeBulk(
  () => { dimCbs.forEach(({cb, dim}) => { cb.checked = true; selectedDims.add(dim); }); render(); },
  () => { dimCbs.forEach(({cb, dim}) => { cb.checked = false; selectedDims.delete(dim); }); render(); }
));

sidebar.appendChild(document.createElement("h2")).textContent = "Judges";
const jCbs = [];
DATA.judges.forEach(j => {
  const lbl = document.createElement("label");
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.checked = selectedJudges.has(j);
  cb.onchange = () => { if (cb.checked) selectedJudges.add(j); else selectedJudges.delete(j); render(); };
  lbl.appendChild(cb);
  lbl.appendChild(document.createTextNode(" " + (j.length > 30 ? j.slice(0, 27) + "…" : j)));
  sidebar.appendChild(lbl);
  jCbs.push({cb, j});
});
sidebar.appendChild(makeBulk(
  () => { jCbs.forEach(({cb, j}) => { cb.checked = true; selectedJudges.add(j); }); render(); },
  () => { jCbs.forEach(({cb, j}) => { cb.checked = false; selectedJudges.delete(j); }); render(); }
));

const meanLbl = document.createElement("label");
const meanCb = document.createElement("input");
meanCb.type = "checkbox";
meanCb.checked = showMeanJudges;
meanCb.onchange = () => { showMeanJudges = meanCb.checked; render(); };
meanLbl.appendChild(meanCb);
meanLbl.appendChild(document.createTextNode(" " + MEAN_LAB));
sidebar.appendChild(document.createElement("h2")).textContent = "Aggregate";
sidebar.appendChild(meanLbl);

sidebar.appendChild(document.createElement("h2")).textContent = "Display";
const legLbl = document.createElement("label");
const legCb = document.createElement("input");
legCb.type = "checkbox";
legCb.checked = showLegend;
legCb.onchange = () => { showLegend = legCb.checked; render(); };
legLbl.appendChild(legCb);
legLbl.appendChild(document.createTextNode(" Show legend"));
sidebar.appendChild(legLbl);

document.querySelectorAll('input[name="metric"]').forEach(r => {
  r.addEventListener("change", () => {
    if (r.checked) { metricMode = r.value; render(); }
  });
});

const legacyCb = document.getElementById("legacy-mode");
const gen1Cb = document.getElementById("gen1-mode");
legacyCb.addEventListener("change", () => { showLegacy = legacyCb.checked; render(); });
gen1Cb.addEventListener("change", () => { showGen1 = gen1Cb.checked; render(); });

/** How many checked judges have a non-empty series in ``byJ`` (ignores synthetic mean key). */
function countJudgesWithData(byJ, selectedJudges) {
  let n = 0;
  for (const judge of Object.keys(byJ)) {
    if (judge === MEAN_KEY) continue;
    if (!selectedJudges.has(judge)) continue;
    const d = byJ[judge];
    if (!d || !d.steps || !d.steps.length) continue;
    const gens = Array.isArray(d.generations) ? d.generations : [];
    let hasVisiblePoint = false;
    for (let i = 0; i < d.steps.length; i++) {
      const g = String(gens[i] || "legacy");
      if (g === "gen1" && showGen1) { hasVisiblePoint = true; break; }
      if (g !== "gen1" && showLegacy) { hasVisiblePoint = true; break; }
    }
    if (hasVisiblePoint) n++;
  }
  return n;
}

function selectedGenerations() {
  const out = [];
  if (showLegacy) out.push("legacy");
  if (showGen1) out.push("gen1");
  return out;
}

/** Mean of y over selected judges, keyed by model label (if present) else checkpoint step. */
function meanSeriesOverSelectedJudges(byJ, selectedJudges, targetGen, useModelLabels) {
  const keyToVals = new Map();
  const keyOrder = new Map();
  for (const judge of Object.keys(byJ)) {
    if (judge === MEAN_KEY) continue;
    if (!selectedJudges.has(judge)) continue;
    const d = byJ[judge];
    if (!d || !d.steps || !d.values) continue;
    const gens = Array.isArray(d.generations) ? d.generations : [];
    const labels = Array.isArray(d.labels) ? d.labels : [];
    for (let i = 0; i < d.steps.length; i++) {
      const g = String(gens[i] || "legacy");
      if (g !== targetGen) continue;
      const s = Number(d.steps[i]);
      const v = Number(d.values[i]);
      if (!Number.isFinite(v)) continue;
      const rawLabel = labels[i];
      const label = typeof rawLabel === "string" ? rawLabel.trim() : "";
      const key = useModelLabels && label ? label : s;
      if (!Number.isFinite(s) && !(useModelLabels && label)) continue;
      if (!keyToVals.has(key)) keyToVals.set(key, []);
      keyToVals.get(key).push(v);
      if (!keyOrder.has(key)) keyOrder.set(key, i);
    }
  }
  if (keyToVals.size === 0) return null;
  const keys = Array.from(keyToVals.keys());
  if (useModelLabels) {
    keys.sort((a, b) => (keyOrder.get(a) || 0) - (keyOrder.get(b) || 0));
  } else {
    keys.sort((a, b) => Number(a) - Number(b));
  }
  const values = keys.map(k => {
    const arr = keyToVals.get(k);
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  });
  return { keys, values };
}

function render() {
  const traces = [];
  const yTitle = metricMode === "bt" ? "Bradley–Terry θ" : "Win rate";
  const layout = {
    template: "plotly_white",
    margin: { t: showLegend ? 56 : 40, b: 72, l: 58, r: 16 },
    xaxis: { title: "Checkpoint" },
    yaxis: { title: yTitle, domain: [0, 1] },
    showlegend: showLegend,
    legend: {
      font: { size: 9 },
      orientation: "h",
      y: 1.02,
      x: 0,
      bgcolor: "rgba(255,255,255,0.85)",
    },
  };

  document.getElementById("hdr-title").textContent =
    metricMode === "bt" ? "G-Eval exports — Bradley–Terry θ" : "G-Eval exports — Win rate";

  for (const leaf of leafNames) {
    if (!selectedLeaves.has(leaf)) continue;
    const L = DATA.leaves[leaf];
    const block = metricMode === "bt" ? (L.theta || {}) : (L.win_rate || {});

    for (const dim of Object.keys(block)) {
      if (!selectedDims.has(dim)) continue;
      const byJ = block[dim];
      if (!byJ) continue;
      const dcol = colorForDim(dim);

      const chLab = (s) => "ch-" + String(Math.round(Number(s)));
      const firstJudge = Object.keys(byJ).find(j => j !== MEAN_KEY);
      const firstD = firstJudge ? byJ[firstJudge] : null;
      const firstLabels = firstD && Array.isArray(firstD.labels) ? firstD.labels : [];
      const useModelLabels = firstLabels.some(lbl => typeof lbl === "string" && lbl.trim().length > 0);

      if (showMeanJudges && countJudgesWithData(byJ, selectedJudges) >= 2) {
        for (const gen of selectedGenerations()) {
          const d = meanSeriesOverSelectedJudges(byJ, selectedJudges, gen, useModelLabels);
          if (!(d && d.keys.length)) continue;
          const glab = gen === "gen1" ? "Gen1" : "Legacy";
          traces.push({
            x: d.keys,
            y: d.values,
            text: d.keys.map(k => (typeof k === "number" ? chLab(k) : String(k))),
            hovertemplate: "%{x}<br>%{y:.4f}<extra></extra>",
            mode: "lines+markers",
            name: leaf + " / " + dim + " / " + MEAN_LAB + " / " + glab,
            line: {
              color: dcol,
              width: 5,
              dash: gen === "gen1" ? "longdashdot" : "dashdot",
            },
            marker: {
              size: 10,
              symbol: gen === "gen1" ? "star" : "star-triangle-up",
              line: { color: "#0f172a", width: 1.2 },
              color: dcol,
            },
            opacity: gen === "gen1" ? 1.0 : 0.45,
          });
        }
      }

      for (const judge of Object.keys(byJ)) {
        if (judge === MEAN_KEY) continue;
        if (!selectedJudges.has(judge)) continue;
        const d = byJ[judge];
        if (!d.steps.length) continue;
        const gens = Array.isArray(d.generations) ? d.generations : [];
        const labels = Array.isArray(d.labels) ? d.labels : [];
        const jd = dashForJudge(judge);
        const msym = markerForJudge(judge);
        for (const gen of selectedGenerations()) {
          const fX = [];
          const fVals = [];
          const fText = [];
          for (let i = 0; i < d.steps.length; i++) {
            const g = String(gens[i] || "legacy");
            if (g !== gen) continue;
            const step = d.steps[i];
            const rawLabel = labels[i];
            const label = typeof rawLabel === "string" ? rawLabel.trim() : "";
            const xVal = useModelLabels && label ? label : step;
            fX.push(xVal);
            fVals.push(d.values[i]);
            fText.push(label || (Number.isFinite(Number(step)) ? chLab(step) : String(step)));
          }
          if (!fX.length) continue;
          const glab = gen === "gen1" ? "Gen1" : "Legacy";
          traces.push({
            x: fX,
            y: fVals,
            text: fText,
            hovertemplate: "%{x}<br>%{text}<br>%{y:.4f}<extra></extra>",
            mode: "lines+markers",
            name: leaf + " / " + dim + " / " + judge.split("/").pop() + " / " + glab,
            line: { color: dcol, width: 2.4, dash: gen === "gen1" ? jd : "dot" },
            marker: {
              size: 5.5,
              symbol: gen === "gen1" ? msym : "triangle-up",
              line: { color: "rgba(15,23,42,0.35)", width: 0.8 },
              color: dcol,
            },
            opacity: gen === "gen1" ? 0.98 : 0.4,
          });
        }
      }
    }
  }

  if (metricMode === "wr") {
    layout.yaxis.range = [-0.02, 1.02];
  }

  const xSet = new Set();
  let hasCategoricalX = false;
  traces.forEach(tr => {
    (tr.x || []).forEach(v => {
      if (typeof v === "string") {
        hasCategoricalX = true;
        return;
      }
      const n = Number(v);
      if (Number.isFinite(n)) xSet.add(n);
    });
  });
  if (hasCategoricalX) {
    layout.xaxis = Object.assign({}, layout.xaxis, {
      title: "Model",
      type: "category",
      tickangle: -28,
      categoryorder: "trace",
    });
  } else {
    const tickvals = Array.from(xSet).sort((a, b) => a - b);
    if (tickvals.length) {
      layout.xaxis = Object.assign({}, layout.xaxis, {
        title: "Checkpoint",
        tickmode: "array",
        tickvals: tickvals,
        ticktext: tickvals.map(v => "ch-" + String(Math.round(v))),
        tickangle: -38,
      });
    }
  }

  Plotly.react("chart", traces, layout, { responsive: true });
}

render();
window.addEventListener("resize", () => Plotly.Plots.resize(document.getElementById("chart")));
</script>
</body>
</html>
"""


def generate_html(data: dict) -> str:
    s = json.dumps(data, ensure_ascii=False)
    return HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--export_root",
        type=str,
        default=str(_DEFAULT_EXPORT_ROOT),
        help=f"Export root (default: {_DEFAULT_EXPORT_ROOT})",
    )
    ap.add_argument(
        "--eval_root",
        type=str,
        default=str(_DATA_EVAL_ROOT),
        help=f"DATA_ROOT/eval root for auto-discovery (default: {_DATA_EVAL_ROOT})",
    )
    ap.add_argument(
        "--export_leaf",
        type=str,
        nargs="*",
        default=[],
        help="Export leaf name(s) under export_root (optional if using auto-discover)",
    )
    ap.add_argument(
        "--export_dir",
        type=str,
        nargs="*",
        default=[],
        help="Full path(s) to export leaf directory(ies)",
    )
    ap.add_argument(
        "--no_auto_eval",
        action="store_true",
        help="Do not fall back to discovering all models under DATA_ROOT/eval (requires --export_leaf or --export_dir)",
    )
    ap.add_argument(
        "--all_export_leaves",
        action="store_true",
        help=(
            "Discover every export leaf under --export_root (subdirs with tables/ + summary or BT CSV). "
            "Ignores names like 'images'. Use when many models exist under .deepeval/geval_exports but "
            "DATA_ROOT/eval does not list them all."
        ),
    )
    ap.add_argument(
        "--output",
        "-o",
        type=str,
        default="geval_export_viewer.html",
        help="Output HTML path",
    )
    args = ap.parse_args()

    export_root = Path(args.export_root).resolve()
    paths: List[Path] = []
    if args.all_export_leaves:
        if args.export_dir or args.export_leaf:
            print(
                "Warning: --all_export_leaves ignores --export_dir / --export_leaf; "
                "using every export leaf under export_root.",
                file=sys.stderr,
            )
        paths = discover_export_leaves_under_export_root(export_root)
    else:
        for d in args.export_dir:
            p = Path(d).resolve()
            if not p.is_dir():
                print(f"Error: not a directory: {p}", file=sys.stderr)
                sys.exit(1)
            paths.append(p)
        for name in args.export_leaf:
            paths.append(_resolve_leaf_path(name, export_root))

        if not paths and not args.no_auto_eval:
            eval_root = Path(args.eval_root).resolve()
            paths = discover_export_leaves_from_data_eval(eval_root, export_root)

    if not paths:
        print(
            "Error: no export folders found. Use --all_export_leaves, pass --export_leaf / --export_dir, "
            f"or ensure DATA_ROOT/eval/<model>/*.jsonl exists with matching {export_root}/<model>/ "
            f"(skipped eval dirs: {_IGNORE_EVAL_DIR_NAMES}).",
            file=sys.stderr,
        )
        sys.exit(1)

    data = build_chart_data(paths)
    has_bt = any(data["leaves"][k].get("theta") for k in data["leaves"])
    has_wr = any(data["leaves"][k].get("win_rate") for k in data["leaves"])
    if not has_bt and not has_wr:
        print("Error: no bradley_terry_theta__*.csv and no win-rate tables in results_summary.md.", file=sys.stderr)
        sys.exit(1)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(generate_html(data), encoding="utf-8")

    print(f"Wrote: {out.resolve()}")
    print(f"  Leaves: {', '.join(data['leaf_order'])}")
    doc_line = []
    for ln in data["leaf_order"]:
        n = data["leaves"][ln].get("document_count")
        doc_line.append(f"{ln}={n}" if isinstance(n, int) else ln)
    print(f"  Documents per leaf: {', '.join(doc_line)}")
    print(f"  Dimensions: {', '.join(data['dimensions'])}")
    print(f"  Judges: {len(data['judges'])} (+ optional mean curve)")
    print("  Open in a browser (or Cursor Simple Browser).")


if __name__ == "__main__":
    main()

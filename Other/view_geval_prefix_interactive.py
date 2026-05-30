#!/usr/bin/env python3
"""
Interactive HTML: cumulative-document win-rate curves M_k(n) from raw G-Eval JSON.

Same statistic as ``Test.ipynb`` (cumulative-prefix section): fix document order
(first-seen ``doc_id``), take the first *n* documents, pool rows from the **selected
judges**, then for each training checkpoint *k* compute pairwise win rate per dimension
and **average across dimensions** (default **equal** weights; the page has **sliders** for
relative weights w<sub>d</sub> and Σ w<sub>d</sub>·WR / Σ w<sub>d</sub>). In the HTML UI
you can **shuffle** the document order (whole ``doc_id`` blocks stay together) and redraw,
or reset to export order.

**Controls in the page (like ``view_checkpoint_results.py``):**

- Checkboxes for **judges** (subset or all).
- Checkboxes for **checkpoints** (subset or all).
- Checkboxes for **dimensions**: M_k(n) includes only the dimensions you tick; **sliders (0–100)**
  set relative weights (equal values ⇒ same as the old unweighted mean over ticked dimensions).
- Optional **mean** trace over the selected checkpoints.
- **Export weighted table:** sidebar button downloads a Markdown table only (same columns as
  ``reports/win_rates_by_model.md``): ``win_rate_*`` = weighted mean of per-dimension rates for the
  current sliders, judges, doc order, and full *n*; human vs LLM split from ``HUMAN_JUDGES`` at HTML build.

Usage (from repo root):

**Default — one HTML, every export leaf in the model dropdown** (under ``--export_root``;
skips e.g. ``images``):

    python Other/view_geval_prefix_interactive.py \\
        -o .deepeval/geval_exports/images/geval_prefix_interactive.html

**Subset — same single HTML, dropdown lists only the named leaves** (order matches your list):

    python Other/view_geval_prefix_interactive.py \\
        --export-leaves viking-13b norwai-mistral-7b \\
        -o .deepeval/geval_exports/images/geval_prefix_subset.html

**Single leaf** (one entry; dropdown is hidden):

    python Other/view_geval_prefix_interactive.py \\
        --export_leaf gemma-2b \\
        -o .deepeval/geval_exports/images/geval_prefix_interactive.html

**Explicit JSON directory** (bypasses ``--export_root`` / leaf names):

    python Other/view_geval_prefix_interactive.py \\
        --export_json_dir .deepeval/geval_exports/gemma-2b/json \\
        -o /tmp/prefix.html
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pairwise_eval.config import HUMAN_JUDGES, REFERENCE_SUMMARY_MODEL_ID
_DEFAULT_EXPORT_ROOT = REPO_ROOT / ".deepeval" / "geval_exports"
# Subdirs of the export root that are not summarization-model export trees (plots, scratch, etc.).
_IGNORE_EXPORT_LEAF_NAMES = frozenset({"images"})


def _resolve_leaf_path(arg: str, export_root: Path) -> Path:
    p = Path(arg)
    if p.is_dir():
        return p.resolve()
    cand = export_root / arg
    if cand.is_dir():
        return cand.resolve()
    raise FileNotFoundError(f"Not a directory and not under export root: {arg!r} (tried {cand})")


def _discover_export_leaves(export_root: Path) -> List[Path]:
    """All subdirectories under ``export_root`` that contain a ``json/`` folder, sorted by name.

    Skips entries in :data:`_IGNORE_EXPORT_LEAF_NAMES` (e.g. ``images``). A leaf without a
    ``json/`` subfolder is silently ignored — those have no G-Eval JSON for this viewer.
    """
    if not export_root.is_dir():
        return []
    leaves: List[Path] = []
    for child in sorted(export_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if child.name in _IGNORE_EXPORT_LEAF_NAMES:
            continue
        if (child / "json").is_dir():
            leaves.append(child.resolve())
    return leaves


_DEFAULT_DIMENSIONS: Tuple[str, ...] = (
    "relevance",
    "consistency",
    "newsworthiness",
    "hygiene",
)

_CK_PALETTE = [
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
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
]


def judge_fragment_from_id(judge_id: str) -> str:
    return judge_id.replace("\\", "__").replace("/", "__")


def parse_geval_filename(path: Path, dimensions: Sequence[str]) -> Optional[Tuple[str, str]]:
    stem = path.stem
    if not stem.startswith("geval__"):
        return None
    body = stem[len("geval__") :]
    for dim in dimensions:
        suf = "__" + dim
        if body.endswith(suf):
            jfrag = body[: -len(suf)]
            judge_id = jfrag.replace("__", "/")
            return judge_id, dim
    return None


def load_geval_rows(
    export_json_dir: Path, judge_fragment: Optional[str], dimensions: Sequence[str]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(export_json_dir.glob("geval__*.json")):
        parsed = parse_geval_filename(path, dimensions)
        if parsed is None:
            continue
        judge_id, dim = parsed
        if judge_fragment is not None and judge_fragment_from_id(judge_id) != judge_fragment:
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"load_geval_rows: skip (unreadable) {path}: {exc}", file=sys.stderr)
            continue
        if not raw.strip():
            print(f"load_geval_rows: skip (empty) {path}", file=sys.stderr)
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"load_geval_rows: skip (invalid JSON) {path}: {exc}", file=sys.stderr)
            continue
        if not isinstance(data, list):
            print(
                f"load_geval_rows: skip (expected JSON array) {path}: root is {type(data).__name__}",
                file=sys.stderr,
            )
            continue
        for r in data:
            if not isinstance(r, dict):
                continue
            rows.append(
                {
                    "judge_id": judge_id,
                    "dimension": dim,
                    "doc_id": r.get("doc_id"),
                    "left": r.get("left"),
                    "right": r.get("right"),
                    "chosen": r.get("chosen"),
                    "choice_side": r.get("choice_side"),
                }
            )
    return rows


def checkpoint_step(model_id: str) -> Optional[int]:
    m = re.search(r"checkpoint-(\d+)-", str(model_id))
    return int(m.group(1)) if m else None


def checkpoint_generation(model_id: str) -> str:
    return "gen1" if "-gen1-" in str(model_id).lower() else "legacy"


def checkpoint_display_label(model_id: str, step: Optional[int]) -> str:
    """Legend label: model prefix for winner ids, otherwise ``ch-<step>``."""
    s = str(model_id)
    marker = "__checkpoint-"
    if marker in s:
        prefix = s.split(marker, 1)[0].strip()
        if prefix:
            return prefix
    return f"ch-{step}" if step is not None else s


def build_n_grid(n_docs: int, start_n: int, step_n: int) -> List[int]:
    if n_docs <= 0:
        return []
    start = max(1, min(start_n, n_docs))
    step = max(1, step_n)
    out: List[int] = []
    n = start
    while n < n_docs:
        out.append(int(n))
        n += step
    if not out or out[-1] != n_docs:
        out.append(int(n_docs))
    return sorted(set(out))


def rows_to_compact_payload(
    rows: List[Dict[str, Any]],
    dimensions: Sequence[str],
    start_n: int,
    step_n: int,
    title: str,
) -> Dict[str, Any]:
    kept: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("doc_id") is None or r.get("left") is None or r.get("right") is None:
            continue
        kept.append(r)

    doc_order: List[Any] = []
    seen: set = set()
    for r in kept:
        did = r["doc_id"]
        if did not in seen:
            seen.add(did)
            doc_order.append(did)
    doc_rank = {doc: i for i, doc in enumerate(doc_order)}

    model_set: set[str] = set()
    for r in kept:
        model_set.add(str(r["left"]))
        model_set.add(str(r["right"]))
        c = r.get("chosen")
        if c is not None:
            model_set.add(str(c))
    model_list = sorted(model_set)
    mi = {m: i for i, m in enumerate(model_list)}

    checkpoints = sorted(
        # Accept both plain ids ("checkpoint-3500-...") and prefixed ids
        # ("model__checkpoint-3500-..."), used by Data/winners flat exports.
        (m for m in model_list if checkpoint_step(m) is not None),
        key=lambda m: checkpoint_step(m) or 0,
    )
    # Also expose the reference (gold) model as a normal selectable trace.
    if REFERENCE_SUMMARY_MODEL_ID in mi and REFERENCE_SUMMARY_MODEL_ID not in checkpoints:
        checkpoints.append(REFERENCE_SUMMARY_MODEL_ID)

    judges = sorted({str(r["judge_id"]) for r in kept})
    human_set = frozenset(str(h) for h in HUMAN_JUDGES)
    judge_human = [j in human_set for j in judges]
    judge_index = {j: i for i, j in enumerate(judges)}
    dim_index = {d: i for i, d in enumerate(dimensions)}

    doci: List[int] = []
    judi: List[int] = []
    dimi: List[int] = []
    Ls: List[int] = []
    Rs: List[int] = []
    chs: List[int] = []

    for r in kept:
        left = str(r["left"])
        right = str(r["right"])
        if left not in mi or right not in mi:
            continue
        dim = str(r["dimension"])
        if dim not in dim_index:
            continue
        tie = r.get("choice_side") == "tie" or r.get("chosen") is None
        if tie:
            ci = -1
        else:
            ci = mi.get(str(r["chosen"]), -1)
        doci.append(int(doc_rank[r["doc_id"]]))
        judi.append(int(judge_index[str(r["judge_id"])]))
        dimi.append(int(dim_index[dim]))
        Ls.append(int(mi[left]))
        Rs.append(int(mi[right]))
        chs.append(int(ci))

    n_grid = build_n_grid(len(doc_order), start_n, step_n)

    ck_meta = []
    for k in checkpoints:
        step = checkpoint_step(k)
        gen = checkpoint_generation(k)
        glab = "gen1" if gen == "gen1" else "legacy"
        short = checkpoint_display_label(k, step)
        ck_meta.append(
            {
                "id": k,
                "step": step,
                "mi": mi[k],
                "gen": gen,
                "short": short,
                "short_gen": f"{short} ({glab})",
            }
        )

    return {
        "title": title,
        "n_grid": n_grid,
        "n_docs": len(doc_order),
        "dimensions": list(dimensions),
        "judges": judges,
        "judge_human": judge_human,
        "checkpoints": ck_meta,
        "models": model_list,
        "rows": {"doc": doci, "judge": judi, "dim": dimi, "L": Ls, "R": Rs, "ch": chs},
        "palette": _CK_PALETTE,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>G-Eval prefix viewer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         display: flex; flex-direction: column; height: 100vh; background: #f6f7fb; }
  header { padding: 10px 18px; background: #fff; border-bottom: 1px solid #ddd;
           display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  header h1 { font-size: 16px; }
  .sub { font-size: 12px; color: #64748b; max-width: 720px; line-height: 1.35; }
  .row2 { padding: 8px 18px; background: #fff; border-bottom: 1px solid #e5e7eb;
          display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: flex-start; font-size: 13px; }
  .block { display: flex; flex-direction: column; gap: 6px; }
  .block strong { font-size: 12px; color: #334155; }
  .checks { display: flex; flex-wrap: wrap; gap: 4px 12px; align-items: center; max-width: 100%; }
  .checks label { font-size: 12px; cursor: pointer; display: inline-flex; align-items: center; gap: 4px; }
  .sw { width: 10px; height: 10px; border-radius: 2px; display: inline-block; border: 1px solid rgba(15,23,42,0.25); }
  .bulk-btn { font-size: 11px; cursor: pointer; background: #eef2ff; border: 1px solid #c7d2fe;
              border-radius: 4px; padding: 2px 7px; color: #3730a3; margin-left: 4px; }
  .bulk-btn:hover { background: #e0e7ff; }
  .main { display: flex; flex: 1; overflow: hidden; min-height: 0; }
  #chart { flex: 1; min-width: 0; min-height: 320px; }
  .sidebar { width: 240px; min-width: 210px; padding: 12px 14px; background: #fff;
             border-left: 1px solid #ddd; overflow-y: auto; font-size: 13px; }
  .sidebar label { display: flex; align-items: center; gap: 6px; cursor: pointer; margin: 6px 0; }
  .dim-weight-row { display: flex; align-items: center; gap: 6px; font-size: 11px; margin: 4px 0; }
  .dim-weight-row span:first-child { min-width: 0; flex: 0 0 88px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .dim-weight-row input[type="range"] { flex: 1; min-width: 50px; }
  .dim-weight-row .wval { width: 26px; text-align: right; color: #64748b; font-variant-numeric: tabular-nums; }
</style>
</head>
<body>

<header>
  <h1 id="hdr">G-Eval — cumulative documents</h1>
  <div id="leaf-picker-wrap" style="display:flex;align-items:center;gap:6px;font-size:12px;color:#334155">
    <label for="leaf-picker"><strong>Model:</strong></label>
    <select id="leaf-picker" style="font-size:12px;padding:2px 6px"></select>
  </div>
  <p class="sub" id="sub"></p>
</header>

<div class="row2">
  <div class="block">
    <strong>Judges <button type="button" class="bulk-btn" id="j-all">All</button><button type="button" class="bulk-btn" id="j-none">None</button></strong>
    <div class="checks" id="judge-bar"></div>
  </div>
  <div class="block">
    <strong>Checkpoints <button type="button" class="bulk-btn" id="c-all">All</button><button type="button" class="bulk-btn" id="c-none">None</button></strong>
    <div class="checks">
      <label><input type="checkbox" id="legacy-mode" checked> Legacy</label>
      <label><input type="checkbox" id="gen1-mode" checked> Gen1</label>
    </div>
    <div class="checks" id="ck-bar"></div>
  </div>
</div>

<div class="main">
  <div id="chart"></div>
  <div class="sidebar">
    <label><input type="checkbox" id="show-mean" checked> Mean over selected checkpoints</label>
    <div style="margin-top:12px;display:flex;flex-direction:column;gap:6px;align-items:flex-start">
      <strong style="font-size:12px;color:#334155">Document order</strong>
      <button type="button" class="bulk-btn" id="shuffle-docs">New random order</button>
      <button type="button" class="bulk-btn" id="reset-docs">Export (first-seen) order</button>
      <p id="doc-order-hint" style="font-size:11px;color:#64748b;line-height:1.35;margin:0"></p>
    </div>
    <div style="margin-top:10px" id="dim-weight-wrap">
      <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:4px">
        <strong style="font-size:12px;color:#334155">Dimension weights</strong>
        <button type="button" class="bulk-btn" id="w-equal" title="Set all weights to 100">Equal (100)</button>
      </div>
      <p style="font-size:10px;color:#64748b;line-height:1.35;margin:0 0 6px 0">
        Relative weights w<sub>d</sub> (0–100). Curve uses
        Σ (w<sub>d</sub> · WR<sub>k,d</sub>) / Σ w<sub>d</sub> over dimensions
        with data; w=0 omits a dimension from the aggregate.
      </p>
      <div id="dim-weight-panel"></div>
      <button type="button" class="bulk-btn" id="export-weighted-md" style="margin-top:8px"
        title="Markdown table only: same columns as reports/win_rates_by_model.md (weighted rates from current UI).">
        Export win_rates_by_model (weighted).md
      </button>
    </div>
    <p style="margin-top:10px;font-size:11px;color:#64748b;line-height:1.4">
      M<sub>k</sub>(n): weighted combination of per-dimension win rates, pooling
      selected judges, first <em>n</em> documents in current order.
    </p>
  </div>
</div>

<script>
const DATA = __DATA_PLACEHOLDER__;

const leafOrder = Array.isArray(DATA.leaf_order) ? DATA.leaf_order : [];
const leafPayloads = (DATA.leaves && typeof DATA.leaves === "object") ? DATA.leaves : {};
const pal = DATA.palette || [];

// Per-leaf state (re-bound by activateLeaf).
let nGrid = [], dims = [], nd = 0, judges = [], cks = [];
let R = { doc: [], judge: [], dim: [], L: [], R: [], ch: [] };
let nR = 0, nDocs = 0;
let currentLeafKey = null;
let docInvPos = [];
let docOrderShuffled = false;
let judgeOn = [];
let ckOn = [];
let ckControls = [];
let dimWeight = [];
let weightSliders = [];
let models = [];
let judgeHuman = [];
// Global UI toggles persist across leaf changes (user preference).
let showLegacy = true;
let showGen1 = true;

function identityDocInv() {
  const inv = new Array(nDocs);
  for (let i = 0; i < nDocs; i++) inv[i] = i;
  return inv;
}
function randomDocInv() {
  const perm = Array.from({ length: nDocs }, (_, i) => i);
  for (let i = nDocs - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const t = perm[i];
    perm[i] = perm[j];
    perm[j] = t;
  }
  const inv = new Array(nDocs);
  for (let p = 0; p < nDocs; p++) inv[perm[p]] = p;
  return inv;
}

function rowInFirstNDocs(docOldRank, n) {
  const pos = docInvPos[docOldRank];
  return pos < n;
}

function outcome(L, R, ch, kMi) {
  if (L !== kMi && R !== kMi) return null;
  if (ch < 0) return 0.5;
  return ch === kMi ? 1 : 0;
}

/** Same weighted M_k(n) as the Plotly traces: per-dim win rate from ``outcome``, then Σ w_d·WR/Σ w_d. */
function weightedWinRateForModelAtN(mMi, judgeMask, dimW, n) {
  const dimSum = new Array(nd).fill(0);
  const dimCnt = new Array(nd).fill(0);
  for (let i = 0; i < nR; i++) {
    if (!rowInFirstNDocs(R.doc[i], n)) continue;
    if (!judgeMask[R.judge[i]]) continue;
    const o = outcome(R.L[i], R.R[i], R.ch[i], mMi);
    if (o === null) continue;
    const d = R.dim[i];
    dimSum[d] += o;
    dimCnt[d] += 1;
  }
  let s = 0, wtot = 0;
  for (let d = 0; d < nd; d++) {
    if (dimCnt[d] <= 0) continue;
    const wd = dimW[d] != null ? dimW[d] : 0;
    if (wd <= 0) continue;
    s += (dimSum[d] / dimCnt[d]) * wd;
    wtot += wd;
  }
  return wtot > 0 ? s / wtot : NaN;
}

function seriesForCk(ckObj, judgeOn, dimW) {
  const kMi = ckObj.mi;
  const y = [];
  for (let gi = 0; gi < nGrid.length; gi++) {
    y.push(weightedWinRateForModelAtN(kMi, judgeOn, dimW, nGrid[gi]));
  }
  return y;
}

function nPairwiseForModel(mMi, judgeMask, n) {
  let cnt = 0;
  for (let i = 0; i < nR; i++) {
    if (!judgeMask[R.judge[i]]) continue;
    if (!rowInFirstNDocs(R.doc[i], n)) continue;
    if (R.L[i] === mMi || R.R[i] === mMi) cnt++;
  }
  return cnt;
}

function fmtWinRate(v) {
  if (!Number.isFinite(v)) return "—";
  return v.toFixed(3);
}

/** Escape pipes / newlines so Markdown tables stay aligned (model ids are usually clean). */
function mdTableCell(s) {
  return String(s).replace(/\r\n|\r|\n/g, " ").replace(/\|/g, "\\|");
}

function buildWeightedWinRatesMd() {
  const n = (nGrid && nGrid.length) ? nGrid[nGrid.length - 1] : nDocs;
  const maskH = judges.map((_, j) => judgeOn[j] && judgeHuman[j]);
  const maskL = judges.map((_, j) => judgeOn[j] && !judgeHuman[j]);
  const lines = [];
  const cols = ["model", "n_pairwise_human", "win_rate_human", "n_pairwise_llm", "win_rate_llm_pooled"];
  lines.push("| " + cols.map(mdTableCell).join(" | ") + " |");
  lines.push("| " + cols.map(() => "---").join(" | ") + " |");
  for (let mMi = 0; mMi < models.length; mMi++) {
    const mname = models[mMi];
    const nh = nPairwiseForModel(mMi, maskH, n);
    const nl = nPairwiseForModel(mMi, maskL, n);
    const wh = nh > 0 ? weightedWinRateForModelAtN(mMi, maskH, dimWeight, n) : NaN;
    const wl = nl > 0 ? weightedWinRateForModelAtN(mMi, maskL, dimWeight, n) : NaN;
    const row = [
      mname,
      String(nh),
      fmtWinRate(wh),
      String(nl),
      fmtWinRate(wl),
    ];
    lines.push("| " + row.map(mdTableCell).join(" | ") + " |");
  }
  lines.push("");
  return lines.join(String.fromCharCode(10));
}

function downloadTextBlob(filename, text) {
  const blob = new Blob([text], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

const judgeBar = document.getElementById("judge-bar");
const ckBar = document.getElementById("ck-bar");
const weightPanel = document.getElementById("dim-weight-panel");
const legacyCb = document.getElementById("legacy-mode");
const gen1Cb = document.getElementById("gen1-mode");

function refreshCheckpointVisibility() {
  ckControls.forEach(({ label, gen }) => {
    const visible = (gen === "gen1" && showGen1) || (gen !== "gen1" && showLegacy);
    label.style.display = visible ? "inline-flex" : "none";
  });
}

function refreshDocOrderHint() {
  const el = document.getElementById("doc-order-hint");
  if (!el) return;
  el.textContent = docOrderShuffled
    ? "Using a random permutation of documents (pairs stay within each doc). Click again for another draw."
    : "Using first-seen document order from the stacked export (same as default pipeline).";
}

function refreshSubheader() {
  document.getElementById("sub").textContent =
    (nDocs ? nDocs + " documents · " : "") +
    nGrid.length + " sample sizes · " + judges.length + " judges · " + cks.length + " checkpoints · " + nd + " dimensions" +
    (docOrderShuffled ? " · doc order: shuffled" : " · doc order: export");
}

/** Tear down per-leaf bindings and rebuild the judges / checkpoints / dim-weights UI for ``key``. */
function activateLeaf(key) {
  const leaf = leafPayloads[key];
  if (!leaf) return;
  currentLeafKey = key;

  // Per-leaf data
  nGrid = leaf.n_grid || [];
  dims = leaf.dimensions || [];
  nd = dims.length;
  judges = leaf.judges || [];
  models = leaf.models || [];
  judgeHuman = (Array.isArray(leaf.judge_human) && leaf.judge_human.length === judges.length)
    ? leaf.judge_human.map((x) => !!x)
    : judges.map(() => false);
  cks = leaf.checkpoints || [];
  R = leaf.rows || { doc: [], judge: [], dim: [], L: [], R: [], ch: [] };
  nR = (R.doc || []).length;
  nDocs = leaf.n_docs || 0;

  // Per-leaf UI state (reset on switch).
  docInvPos = identityDocInv();
  docOrderShuffled = false;
  judgeOn = judges.map(() => true);
  ckOn = cks.map(() => true);
  ckControls = [];
  dimWeight = dims.map(() => 100);
  weightSliders = [];

  // Header / picker reflect the active leaf.
  document.getElementById("hdr").textContent = leaf.title || key || "G-Eval — cumulative documents";

  // Rebuild dimension-weight panel.
  weightPanel.innerHTML = "";
  dims.forEach((dn, di) => {
    const row = document.createElement("div");
    row.className = "dim-weight-row";
    const name = document.createElement("span");
    name.textContent = dn;
    name.title = dn;
    const rng = document.createElement("input");
    rng.type = "range";
    rng.min = "0";
    rng.max = "100";
    rng.value = "100";
    const val = document.createElement("span");
    val.className = "wval";
    val.textContent = "100";
    rng.addEventListener("input", () => {
      const v = parseInt(rng.value, 10);
      dimWeight[di] = Number.isNaN(v) ? 0 : v;
      val.textContent = String(dimWeight[di]);
      render();
    });
    weightSliders.push(rng);
    row.appendChild(name);
    row.appendChild(rng);
    row.appendChild(val);
    weightPanel.appendChild(row);
  });

  // Rebuild judges bar.
  judgeBar.innerHTML = "";
  judges.forEach((j, ji) => {
    const lbl = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = true;
    cb.addEventListener("change", () => { judgeOn[ji] = cb.checked; render(); });
    lbl.appendChild(cb);
    const short = j.includes("/") ? j.split("/").pop() : j;
    lbl.appendChild(document.createTextNode(" " + short));
    judgeBar.appendChild(lbl);
  });

  // Rebuild checkpoints bar.
  ckBar.innerHTML = "";
  cks.forEach((ck, ci) => {
    const lbl = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = true;
    cb.addEventListener("change", () => { ckOn[ci] = cb.checked; render(); });
    const sw = document.createElement("span");
    sw.className = "sw";
    sw.style.background = pal[ci % pal.length] || "#334155";
    lbl.appendChild(cb);
    lbl.appendChild(sw);
    lbl.appendChild(document.createTextNode(" " + (ck.short_gen || ck.short || ck.step)));
    ckBar.appendChild(lbl);
    ckControls.push({ label: lbl, cb, ci, gen: String(ck.gen || "legacy") });
  });

  refreshCheckpointVisibility();
  refreshSubheader();
  refreshDocOrderHint();
  render();
}

// One-time event wiring for static controls (they reference per-leaf state via closure on let bindings).
legacyCb.addEventListener("change", () => {
  showLegacy = legacyCb.checked;
  refreshCheckpointVisibility();
  render();
});
gen1Cb.addEventListener("change", () => {
  showGen1 = gen1Cb.checked;
  refreshCheckpointVisibility();
  render();
});

document.getElementById("w-equal").addEventListener("click", () => {
  weightSliders.forEach((r, i) => {
    r.value = "100";
    dimWeight[i] = 100;
    r.parentElement.querySelector(".wval").textContent = "100";
  });
  render();
});

document.getElementById("export-weighted-md").addEventListener("click", () => {
  if (!currentLeafKey) return;
  if (!judgeOn.some(Boolean)) {
    alert("Select at least one judge.");
    return;
  }
  let anyW = false;
  for (let d = 0; d < nd; d++) {
    if ((dimWeight[d] != null ? dimWeight[d] : 0) > 0) { anyW = true; break; }
  }
  if (!anyW) {
    alert("Set at least one dimension weight above 0.");
    return;
  }
  const text = buildWeightedWinRatesMd();
  const safeLeaf = String(currentLeafKey).replace(/[/\\\\]/g, "_");
  downloadTextBlob("win_rates_by_model_weighted__" + safeLeaf + ".md", text);
});

document.getElementById("j-all").onclick = () => {
  judgeBar.querySelectorAll("input").forEach((cb, i) => { cb.checked = true; judgeOn[i] = true; });
  render();
};
document.getElementById("j-none").onclick = () => {
  judgeBar.querySelectorAll("input").forEach((cb, i) => { cb.checked = false; judgeOn[i] = false; });
  render();
};
document.getElementById("c-all").onclick = () => {
  ckControls.forEach(({ label, cb, ci }) => {
    if (label.style.display === "none") return;
    cb.checked = true;
    ckOn[ci] = true;
  });
  render();
};
document.getElementById("c-none").onclick = () => {
  ckControls.forEach(({ label, cb, ci }) => {
    if (label.style.display === "none") return;
    cb.checked = false;
    ckOn[ci] = false;
  });
  render();
};
document.getElementById("show-mean").addEventListener("change", render);

document.getElementById("shuffle-docs").addEventListener("click", () => {
  docInvPos = randomDocInv();
  docOrderShuffled = true;
  refreshDocOrderHint();
  refreshSubheader();
  render();
});
document.getElementById("reset-docs").addEventListener("click", () => {
  docInvPos = identityDocInv();
  docOrderShuffled = false;
  refreshDocOrderHint();
  refreshSubheader();
  render();
});

// Leaf dropdown setup (hidden when there's only one leaf).
const picker = document.getElementById("leaf-picker");
const pickerWrap = document.getElementById("leaf-picker-wrap");
if (leafOrder.length > 1) {
  leafOrder.forEach((key) => {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = key;
    picker.appendChild(opt);
  });
  picker.addEventListener("change", () => activateLeaf(picker.value));
} else {
  pickerWrap.style.display = "none";
}

const initialLeaf = leafOrder.length ? leafOrder[0] : null;
if (initialLeaf) {
  if (picker) picker.value = initialLeaf;
  activateLeaf(initialLeaf);
}

function render() {
  const traces = [];
  const anyJ = judgeOn.some(Boolean);
  if (!anyJ) {
    Plotly.react("chart", [], {
      template: "plotly_white",
      annotations: [{
        text: "Select at least one judge",
        xref: "paper", yref: "paper", x: 0.5, y: 0.5, showarrow: false,
        font: { size: 15, color: "#64748b" }
      }],
      margin: { t: 30, b: 50, l: 58, r: 16 },
    }, { responsive: true });
    return;
  }

  const meanYs = nGrid.map(() => ({ a: 0, b: 0 }));
  let anyCk = false;

  cks.forEach((ck, ci) => {
    if (!ckOn[ci]) return;
    const gen = String(ck.gen || "legacy");
    if (gen === "gen1" && !showGen1) return;
    if (gen !== "gen1" && !showLegacy) return;
    anyCk = true;
    const y = seriesForCk(ck, judgeOn, dimWeight);
    const col = pal[ci % pal.length];
    const glab = gen === "gen1" ? "Gen1" : "Legacy";
    traces.push({
      x: nGrid,
      y,
      mode: "lines+markers",
      name: (ck.short || String(ck.step)) + " / " + glab,
      line: { color: col, width: 2, dash: gen === "gen1" ? "solid" : "dash" },
      marker: { size: 5, color: col },
      opacity: gen === "gen1" ? 1.0 : 0.4,
      hovertemplate: "%{x} docs<br>M=%{y:.4f}<extra></extra>",
    });
    y.forEach((v, gi) => {
      if (Number.isFinite(v)) { meanYs[gi].a += v; meanYs[gi].b += 1; }
    });
  });

  if (!anyCk) {
    Plotly.react("chart", [], {
      template: "plotly_white",
      annotations: [{
        text: "Select at least one checkpoint",
        xref: "paper", yref: "paper", x: 0.5, y: 0.5, showarrow: false,
        font: { size: 15, color: "#64748b" }
      }],
      margin: { t: 30, b: 50, l: 58, r: 16 },
    }, { responsive: true });
    return;
  }

  if (document.getElementById("show-mean").checked && traces.length >= 2) {
    const my = meanYs.map(({ a, b }) => (b > 0 ? a / b : NaN));
    traces.push({
      x: nGrid,
      y: my,
      mode: "lines+markers",
      name: "Mean (selected ckpts)",
      line: { color: "#111", width: 3, dash: "dashdot" },
      marker: { size: 6, symbol: "diamond", color: "#111" },
      hovertemplate: "%{x} docs<br>mean=%{y:.4f}<extra></extra>",
    });
  }

  const nDimOn = nd;
  let uniformW = true;
  let wRef = null;
  let anyPosW = false;
  for (let d = 0; d < nd; d++) {
    if (dimWeight[d] <= 0) continue;
    anyPosW = true;
    if (wRef === null) wRef = dimWeight[d];
    else if (dimWeight[d] !== wRef) { uniformW = false; break; }
  }
  const yDimLabel = !anyPosW
    ? "M_k(n) — set positive weights for selected dimensions"
    : (uniformW
        ? (nDimOn === nd
            ? "M_k(n) — mean over dimensions (equal weights)"
            : "M_k(n) — mean over " + nDimOn + " dimension(s) (equal weights)")
        : "M_k(n) — weighted mean over selected dimensions");

  const layout = {
    template: "plotly_white",
    xaxis: {
      title: docOrderShuffled
        ? "First n documents (cumulative) [shuffled]"
        : "First n documents (cumulative)",
    },
    yaxis: { title: yDimLabel, range: [-0.02, 1.02] },
    margin: { t: 36, b: 52, l: 58, r: 16 },
    showlegend: traces.length <= 14,
    legend: { orientation: "h", y: 1.08, x: 0, font: { size: 10 } },
    annotations: [],
  };

  Plotly.react("chart", traces, layout, { responsive: true });
}

window.addEventListener("resize", () => Plotly.Plots.resize(document.getElementById("chart")));
</script>
</body>
</html>
"""


def generate_html(payload: dict) -> str:
    return HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", json.dumps(payload, ensure_ascii=False))


def _wrap_leaves(
    leaf_payloads: Dict[str, dict], leaf_order: List[str]
) -> dict:
    """Wrap one or more single-leaf payloads in the multi-leaf envelope the HTML expects."""
    first = leaf_payloads[leaf_order[0]] if leaf_order else None
    palette = (first or {}).get("palette", _CK_PALETTE)
    return {
        "leaf_order": list(leaf_order),
        "leaves": leaf_payloads,
        "palette": palette,
    }


def _build_leaf_payload(
    jdir: Path,
    title: str,
    *,
    dims: Tuple[str, ...],
    start_n: int,
    step_n: int,
) -> Optional[dict]:
    """Load one leaf's G-Eval JSON and return the single-leaf payload (or ``None`` if empty)."""
    print(f"Loading G-Eval JSON from {jdir} …", flush=True)
    rows = load_geval_rows(jdir, None, dims)
    if not rows:
        print(f"  (skip) no rows loaded from {jdir}", file=sys.stderr)
        return None
    print(f"  Loaded {len(rows)} raw row(s)", flush=True)
    payload = rows_to_compact_payload(rows, dims, start_n, step_n, title=title)
    print(
        f"  Built payload: docs={payload['n_docs']}, "
        f"checkpoints={len(payload['checkpoints'])}, "
        f"judges={len(payload['judges'])}",
        flush=True,
    )
    return payload


def _render_one(
    jdir: Path,
    title: str,
    out: Path,
    *,
    dims: Tuple[str, ...],
    start_n: int,
    step_n: int,
) -> bool:
    """Build the interactive HTML for one ``<leaf>/json`` directory and write it to ``out``.

    Returns ``True`` on success, ``False`` if the directory had no usable G-Eval rows (the
    caller can then skip it without failing the whole batch).
    """
    leaf_payload = _build_leaf_payload(
        jdir, title, dims=dims, start_n=start_n, step_n=step_n
    )
    if leaf_payload is None:
        return False
    wrapped = _wrap_leaves({title: leaf_payload}, [title])
    out.parent.mkdir(parents=True, exist_ok=True)
    print("Writing HTML …", flush=True)
    out.write_text(generate_html(wrapped), encoding="utf-8")
    print(f"Wrote: {out.resolve()}")
    print(f"  Export: {jdir}")
    return True


def _render_combined(
    leaves: List[Path],
    out: Path,
    *,
    dims: Tuple[str, ...],
    start_n: int,
    step_n: int,
) -> bool:
    """Build one HTML containing payloads for every leaf, switchable via a dropdown."""
    leaf_payloads: Dict[str, dict] = {}
    leaf_order: List[str] = []
    for leaf in leaves:
        jdir = leaf / "json"
        print(f"\n=== {leaf.name} ===", flush=True)
        payload = _build_leaf_payload(
            jdir, leaf.name, dims=dims, start_n=start_n, step_n=step_n
        )
        if payload is None:
            continue
        leaf_payloads[leaf.name] = payload
        leaf_order.append(leaf.name)
    if not leaf_payloads:
        print("Error: no leaves produced any usable rows.", file=sys.stderr)
        return False
    wrapped = _wrap_leaves(leaf_payloads, leaf_order)
    out.parent.mkdir(parents=True, exist_ok=True)
    print("\nWriting combined HTML …", flush=True)
    out.write_text(generate_html(wrapped), encoding="utf-8")
    print(f"Wrote: {out.resolve()}")
    print(f"  Leaves embedded ({len(leaf_order)}): {leaf_order}")
    return True


def _resolve_multi_leaves(
    export_root: Path, export_leaves: Optional[Sequence[str]]
) -> List[Path]:
    """Discover leaves under ``export_root``; optionally keep only ``export_leaves`` (by folder name).

    ``None`` or an empty sequence means every discoverable leaf (sorted as in
    :func:`_discover_export_leaves`). A non-empty list preserves the user's order.
    """
    discovered = _discover_export_leaves(export_root)
    if not export_leaves:
        return discovered
    by_name = {p.name: p for p in discovered}
    missing = [n for n in export_leaves if n not in by_name]
    if missing:
        raise ValueError(
            f"--export-leaves not found under {export_root}: {missing!r}. "
            f"Known: {sorted(by_name)}"
        )
    return [by_name[n] for n in export_leaves]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--export_json_dir",
        type=str,
        default=None,
        help="Path to …/<leaf>/json (G-Eval export JSON files). Mutually exclusive with leaf modes.",
    )
    ap.add_argument(
        "--export_leaf",
        type=str,
        default=None,
        help="Single leaf under --export_root (one model; dropdown hidden). "
        "Mutually exclusive with --export-leaves and the default multi-leaf build.",
    )
    ap.add_argument(
        "--export-leaves",
        dest="export_leaves",
        nargs="*",
        default=None,
        metavar="LEAF",
        help=(
            "Default multi-leaf build only: restrict the dropdown to these export folder names "
            "(under --export_root), in list order. Omit the flag entirely to include every "
            "discoverable leaf. Passing --export-leaves with no names is treated as “all leaves”."
        ),
    )
    ap.add_argument(
        "--export_root",
        type=str,
        default=str(_DEFAULT_EXPORT_ROOT),
        help=f"Export root for multi-leaf and --export_leaf (default: {_DEFAULT_EXPORT_ROOT})",
    )
    ap.add_argument("--start_n", type=int, default=10, help="Smallest n in the grid (default: 10)")
    ap.add_argument("--step_n", type=int, default=5, help="Step between n values (default: 5)")
    ap.add_argument(
        "--dimensions",
        type=str,
        nargs="*",
        default=list(_DEFAULT_DIMENSIONS),
        help="Dimensions (default: four summarization criteria)",
    )
    ap.add_argument("-o", "--output", type=str, default="geval_prefix_interactive.html", help="Output HTML path")
    args = ap.parse_args()

    dims = tuple(args.dimensions)

    if args.export_json_dir:
        if args.export_leaf:
            print("Error: use only one of --export_json_dir or --export_leaf", file=sys.stderr)
            sys.exit(1)
        if args.export_leaves is not None and len(args.export_leaves) > 0:
            print("Error: --export_json_dir cannot be combined with --export-leaves", file=sys.stderr)
            sys.exit(1)
        jdir = Path(args.export_json_dir).resolve()
        title = jdir.parent.name if jdir.name == "json" else jdir.name
        if not jdir.is_dir():
            print(f"Error: not a directory: {jdir}", file=sys.stderr)
            sys.exit(1)
        out = Path(args.output)
        if not _render_one(jdir, title, out, dims=dims, start_n=args.start_n, step_n=args.step_n):
            sys.exit(1)
        print("  Open in a browser (or Cursor Simple Browser).")
        return

    if args.export_leaf:
        if args.export_leaves is not None and len(args.export_leaves) > 0:
            print(
                "Error: use --export_leaf alone, or omit it and use --export-leaves (without --export_leaf)",
                file=sys.stderr,
            )
            sys.exit(1)
        export_root = Path(args.export_root).resolve()
        leaf = _resolve_leaf_path(args.export_leaf, export_root)
        jdir = leaf / "json"
        if not jdir.is_dir():
            print(f"Error: not a directory: {jdir}", file=sys.stderr)
            sys.exit(1)
        out = Path(args.output)
        if not _render_one(jdir, leaf.name, out, dims=dims, start_n=args.start_n, step_n=args.step_n):
            sys.exit(1)
        print("  Open in a browser (or Cursor Simple Browser).")
        return

    # Default: one combined HTML; optional --export-leaves narrows the dropdown.
    export_root = Path(args.export_root).resolve()
    leaf_names: Optional[List[str]] = None
    if args.export_leaves is not None and len(args.export_leaves) > 0:
        leaf_names = list(args.export_leaves)
    try:
        leaves = _resolve_multi_leaves(export_root, leaf_names)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    if not leaves:
        print(
            f"Error: no export leaves with a json/ subfolder under {export_root}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(
        f"Building combined HTML from {len(leaves)} leaf(es) under {export_root}: "
        f"{[p.name for p in leaves]}",
        flush=True,
    )
    out = Path(args.output)
    ok = _render_combined(leaves, out, dims=dims, start_n=args.start_n, step_n=args.step_n)
    if not ok:
        sys.exit(1)
    print("  Open in a browser (or Cursor Simple Browser); pick a model from the dropdown.")


if __name__ == "__main__":
    main()

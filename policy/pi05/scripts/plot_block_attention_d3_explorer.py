#!/usr/bin/env python3
"""Build a D3 block-level denoise attention explorer.

The generated HTML lets you select one image block and inspect how its
attention score changes across denoise steps or across layers, split by head
and by suffix/action query token.

It expects traces saved with:
  PI05_TRACE_SAVE_ATTN=1 PI05_TRACE_SAVE_ATTN_FULL=1
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from plot_denoise_head_query_attention import find_image_meta, infer_dir_from_trace, parse_layer
from plot_patch_kv_overlay import image_hwc


def parse_steps(value: str | None, infer_dir: Path) -> list[int]:
    if value is None or value.strip().lower() in {"", "all"}:
        steps = []
        for path in sorted(infer_dir.glob("attn_denoise_step_*.pt")):
            steps.append(int(path.stem.rsplit("_", 1)[1]))
        return steps
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(part))
    return sorted(dict.fromkeys(out))


def load_payload(infer_dir: Path, step: int) -> dict[str, Any]:
    path = infer_dir / f"attn_denoise_step_{step:03d}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def image_data_uri(path: Path) -> str:
    data = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def round_nested(values: np.ndarray, digits: int = 8) -> Any:
    return np.round(values.astype(np.float32), digits).tolist()


def available_layers(first_payload: dict[str, Any]) -> list[str]:
    full_layers = first_payload.get("full_layers")
    if not full_layers:
        raise ValueError("Trace does not contain full_layers. Re-run with PI05_TRACE_SAVE_ATTN_FULL=1.")
    return sorted(full_layers.keys())


def parse_layers(value: str, first_payload: dict[str, Any]) -> list[str]:
    if value.strip().lower() in {"", "all"}:
        return available_layers(first_payload)
    return [parse_layer(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Trace root, episode dir, or infer dir.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--infer", type=int, default=0)
    parser.add_argument("--steps", default="all")
    parser.add_argument("--layers", default="all", help="all, or comma-separated layer ids like 0,3,12.")
    parser.add_argument("--image-key", default="base_0_rgb")
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--round-digits", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.path.name.startswith("infer_"):
        infer_dir = args.path
    else:
        infer_dir = infer_dir_from_trace(args.path, args.episode, args.infer)

    steps = parse_steps(args.steps, infer_dir)
    if not steps:
        raise ValueError("No denoise attention step files found.")

    first_payload = load_payload(infer_dir, steps[0])
    layer_names = parse_layers(args.layers, first_payload)
    missing = [name for name in layer_names if name not in first_payload.get("full_layers", {})]
    if missing:
        raise KeyError(f"Requested layers missing in first step: {missing}. Available: {available_layers(first_payload)}")

    image_meta = find_image_meta(first_payload, args.image_key)
    obs_key = image_meta.get("obs_key", args.image_key)
    obs = np.load(infer_dir / "obs.npz")
    base_image = image_hwc(obs[obs_key])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.out_dir / f"{args.image_key}_reference.png"
    Image.fromarray(base_image).save(image_path)

    query_axis = first_payload.get("query_axis", {})
    state_query_count = int(query_axis.get("state_query_count", 0))
    action_query_start = int(query_axis.get("action_query_start", state_query_count))
    action_query_count = int(query_axis.get("action_query_count", 0))

    layer_data: dict[str, Any] = {}
    shape_info: tuple[int, int, int, int] | None = None
    for layer_name in layer_names:
        step_arrays = []
        for step in steps:
            payload = load_payload(infer_dir, step)
            full = payload.get("full_layers", {})
            if layer_name not in full:
                raise KeyError(f"{layer_name} missing in step {step}. Available: {sorted(full)}")
            if args.image_key not in full[layer_name]:
                raise KeyError(f"{args.image_key} missing in {layer_name} step {step}.")
            arr = full[layer_name][args.image_key].detach().to(torch.float32).numpy()
            if shape_info is None:
                shape_info = arr.shape
            elif arr.shape != shape_info:
                raise ValueError(f"Shape mismatch at {layer_name} step {step}: {arr.shape} vs {shape_info}")
            step_arrays.append(arr)
        stacked = np.stack(step_arrays, axis=0)  # [steps, heads, queries, grid_h, grid_w]
        if args.block_size > 1:
            steps_n, heads_n, queries_n, grid_h, grid_w = stacked.shape
            block_h = (grid_h + args.block_size - 1) // args.block_size
            block_w = (grid_w + args.block_size - 1) // args.block_size
            blocked = np.zeros((steps_n, heads_n, queries_n, block_h, block_w), dtype=np.float32)
            for br in range(block_h):
                for bc in range(block_w):
                    r0 = br * args.block_size
                    r1 = min(grid_h, r0 + args.block_size)
                    c0 = bc * args.block_size
                    c1 = min(grid_w, c0 + args.block_size)
                    blocked[:, :, :, br, bc] = stacked[:, :, :, r0:r1, c0:c1].mean(axis=(-2, -1))
            stacked = blocked
        layer_data[layer_name] = round_nested(stacked, digits=args.round_digits)

    if shape_info is None:
        raise ValueError("No attention tensors loaded.")
    num_heads, suffix_query_len, source_grid_h, source_grid_w = shape_info
    if args.block_size > 1:
        grid_h = (source_grid_h + args.block_size - 1) // args.block_size
        grid_w = (source_grid_w + args.block_size - 1) // args.block_size
    else:
        grid_h, grid_w = source_grid_h, source_grid_w

    data_payload = {
        "inferDir": str(infer_dir),
        "steps": steps,
        "layers": layer_names,
        "imageKey": args.image_key,
        "obsKey": obs_key,
        "imageUri": image_data_uri(image_path),
        "numHeads": num_heads,
        "suffixQueryLen": suffix_query_len,
        "stateQueryCount": state_query_count,
        "actionQueryStart": action_query_start,
        "actionQueryCount": action_query_count,
        "gridH": grid_h,
        "gridW": grid_w,
        "sourceGridH": source_grid_h,
        "sourceGridW": source_grid_w,
        "blockSize": args.block_size,
        "data": layer_data,
    }

    page = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(data_payload, ensure_ascii=False))
    (args.out_dir / "index.html").write_text(page, encoding="utf-8")
    summary = {key: value for key, value in data_payload.items() if key not in {"data", "imageUri"}}
    (args.out_dir / "data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_dir / 'index.html'}")


HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>D3 Block Attention Explorer</title>
  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <style>
    :root {
      --blue: #1463ff;
      --grid: rgba(255,255,255,0.85);
      --ink: #20242b;
      --muted: #667085;
      --line: #d8dde6;
      --panel: #fff;
      --bg: #f6f7f9;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, "Helvetica Neue", sans-serif;
    }
    header {
      padding: 16px 20px 10px;
      background: #fff;
      border-bottom: 1px solid var(--line);
    }
    h1 { margin: 0 0 8px; font-size: 22px; }
    code { background: #eef1f5; padding: 2px 5px; border-radius: 4px; }
    .subtle { color: var(--muted); font-size: 13px; line-height: 1.45; }
    .warning {
      margin-top: 10px;
      padding: 8px 10px;
      background: #fff4d6;
      border: 1px solid #f0c15a;
      border-radius: 6px;
      color: #6d4b00;
      display: none;
    }
    main {
      display: grid;
      grid-template-columns: minmax(360px, 520px) minmax(760px, 1fr);
      gap: 16px;
      padding: 16px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      box-shadow: 0 1px 2px rgba(16,24,40,0.04);
    }
    h2 { margin: 0 0 10px; font-size: 17px; }
    h3 { margin: 18px 0 8px; font-size: 15px; }
    .imageStage {
      position: relative;
      width: 100%;
      max-width: 500px;
      border: 1px solid var(--line);
      background: #fff;
    }
    #refImage {
      width: 100%;
      display: block;
    }
    #gridSvg {
      position: absolute;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
      cursor: crosshair;
    }
    .blockRect {
      fill: transparent;
      stroke: var(--grid);
      stroke-width: 1;
      vector-effect: non-scaling-stroke;
    }
    .blockRect:hover {
      fill: rgba(20, 99, 255, 0.12);
      stroke: var(--blue);
      stroke-width: 2;
    }
    .blockRect.selected {
      fill: rgba(20, 99, 255, 0.18);
      stroke: var(--blue);
      stroke-width: 3;
    }
    .qSelector {
      margin: 12px 0 4px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfe;
    }
    #qBrushSvg {
      width: 100%;
      height: 64px;
      display: block;
      background: #fff;
      border: 1px solid #e5e9f0;
      border-radius: 6px;
      touch-action: none;
    }
    .qCell {
      stroke: #fff;
      stroke-width: 1.2px;
      rx: 3;
      cursor: pointer;
    }
    .qCell.selected {
      stroke: #003bce;
      stroke-width: 2.2px;
    }
    .qLabel {
      pointer-events: none;
      font-size: 10px;
      fill: #475467;
      text-anchor: middle;
    }
    #qBrushSvg .selection {
      fill: rgba(20, 99, 255, 0.18);
      stroke: var(--blue);
    }
    #qBrushSvg .overlay {
      fill: transparent;
      pointer-events: all;
      cursor: crosshair;
    }
    #qBrushSvg .handle {
      fill: rgba(20, 99, 255, 0.22);
      cursor: ew-resize;
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(4, minmax(150px, 1fr));
      gap: 8px 12px;
      align-items: end;
    }
    label { display: grid; gap: 3px; font-size: 12px; color: var(--muted); }
    select, input {
      height: 30px;
      border: 1px solid #cfd5df;
      border-radius: 6px;
      padding: 4px 7px;
      background: #fff;
      color: var(--ink);
    }
    .headChecks {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      margin-top: 10px;
    }
    .headChecks label {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      color: var(--ink);
    }
    .chart {
      width: 100%;
      height: 330px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
    }
    .heatmapGrid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 12px;
      margin-top: 8px;
    }
    .heatmapCard {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #fff;
    }
    .heatmapCard h4 {
      margin: 0 0 6px;
      font-size: 13px;
      color: var(--ink);
    }
    .heatmapSvg {
      width: 100%;
      height: 230px;
      display: block;
    }
    .heatCell {
      stroke: #fff;
      stroke-width: 1;
    }
    .overlayStrip {
      display: flex;
      gap: 12px;
      overflow-x: auto;
      padding: 8px 2px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfe;
    }
    .overlayCard {
      flex: 0 0 260px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #fff;
    }
    .overlayCard h4 {
      margin: 0 0 6px;
      font-size: 13px;
    }
    .overlayStage {
      position: relative;
      width: 100%;
      border: 1px solid #e5e9f0;
      background: #fff;
    }
    .overlayStage img {
      width: 100%;
      display: block;
    }
    .overlayStage svg {
      position: absolute;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
    }
    .overlayCell {
      stroke: rgba(255,255,255,0.7);
      stroke-width: 0.8;
    }
    .axis text { fill: #667085; font-size: 11px; }
    .axis path, .axis line { stroke: #b7c0cf; }
    .gridLine line { stroke: #eceff4; }
    .linePath { fill: none; stroke-width: 2.2px; }
    .dot { stroke: #fff; stroke-width: 1.2px; }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      margin: 8px 0 0;
      font-size: 12px;
    }
    .legendItem { display: inline-flex; align-items: center; gap: 5px; }
    .swatch { width: 14px; height: 3px; border-radius: 99px; display: inline-block; }
    .chips span {
      display: inline-block;
      margin: 3px;
      padding: 3px 7px;
      background: #eef1f5;
      border-radius: 999px;
      font-size: 12px;
    }
    #tooltip {
      position: fixed;
      pointer-events: none;
      z-index: 10;
      display: none;
      max-width: 320px;
      padding: 8px 9px;
      border-radius: 7px;
      background: rgba(20, 24, 31, 0.94);
      color: #fff;
      font-size: 12px;
      line-height: 1.45;
      box-shadow: 0 8px 20px rgba(0,0,0,0.18);
    }
    .metricTable {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      margin-top: 10px;
    }
    .metricTable th, .metricTable td {
      border-bottom: 1px solid #edf0f4;
      text-align: left;
      padding: 6px;
    }
    .metricTable th { color: var(--muted); font-weight: 600; }
  </style>
</head>
<body>
  <header>
    <h1>D3 Block Attention Explorer</h1>
    <div class="subtle">Trace: <code id="tracePath"></code></div>
    <div class="subtle">Click a block on the image, then inspect attention score curves across denoise step or across layer.</div>
    <div id="layerWarning" class="warning"></div>
    <div id="d3Warning" class="warning">D3 没有加载成功。这个页面默认从 jsDelivr 加载 D3，如果当前机器不能访问外网，需要换成内联/本地 d3.js。</div>
  </header>

  <main>
    <section class="panel">
      <h2>Image Block</h2>
      <div class="imageStage">
        <img id="refImage" alt="reference observation">
        <svg id="gridSvg"></svg>
      </div>
      <p id="blockText" class="subtle"></p>
      <p class="subtle">
        Tooltip 中的 token id 是 image patch token 在该相机 16x16 网格内的局部编号：
        <code>token = row * gridW + col</code>。如果 block-size 大于 1，则 score 是 block 内 patch 的平均。
      </p>
    </section>

    <section class="panel">
      <h2>Curves</h2>
      <div class="controls">
        <label>View mode
          <select id="mode">
            <option value="step">x-axis: denoise step</option>
            <option value="layer">x-axis: layer</option>
          </select>
        </label>
        <label>Fixed layer
          <select id="layerSelect"></select>
        </label>
        <label>Fixed step
          <select id="stepSelect"></select>
        </label>
        <label>Aggregation
          <select id="aggSelect">
            <option value="mean">mean over q</option>
            <option value="max">max over q</option>
            <option value="single">single q only</option>
          </select>
        </label>
        <label>Q start
          <input id="qStart" type="number" min="0">
        </label>
        <label>Q end
          <input id="qEnd" type="number" min="0">
        </label>
        <label>Q stride
          <input id="qStride" type="number" min="1" value="4">
        </label>
        <label>Selected head for q curves
          <select id="queryHeadSelect"></select>
        </label>
        <label>Overlay scale
          <select id="overlayScaleSelect">
            <option value="frame_p95">per image p95</option>
            <option value="frame_max">per image max</option>
            <option value="sequence_p95">shared sequence p95</option>
            <option value="sequence_max">shared sequence max</option>
          </select>
        </label>
        <label>Overlay sequence axis
          <select id="overlayAxisSelect">
            <option value="view">follow view mode</option>
            <option value="q">q token</option>
          </select>
        </label>
      </div>
      <div id="headChecks" class="headChecks"></div>
      <div class="qSelector">
        <div class="subtle">
          Q token selector: click a square to choose one q, or drag/brush over squares to select a q range.
        </div>
        <svg id="qBrushSvg"></svg>
        <div id="qSelectionText" class="subtle"></div>
      </div>
      <p id="metaText" class="subtle"></p>

      <h3>Per-head curve for selected block</h3>
      <svg id="headChart" class="chart"></svg>
      <div id="headLegend" class="legend"></div>

      <h3>Per-q curve for selected head and block</h3>
      <svg id="queryChart" class="chart"></svg>
      <div id="queryLegend" class="legend"></div>

      <h3>Step × Layer heatmap for selected block</h3>
      <p class="subtle">
        Each mini heatmap is one head. X-axis is layer, Y-axis is denoise step.
        Color is attention score on the selected image block, aggregated over the selected q tokens.
      </p>
      <div id="headHeatmaps" class="heatmapGrid"></div>

      <h3>Image overlay heatmap sequence</h3>
      <p class="subtle">
        These heatmaps are overlaid on the original image. In layer mode they are ordered by layer at the fixed step;
        in step mode they are ordered by denoise step at the fixed layer. The selected head and selected q tokens are used.
      </p>
      <div id="overlaySequence" class="overlayStrip"></div>

      <h3>Hover / selection summary</h3>
      <div id="summary" class="chips"></div>
      <table class="metricTable">
        <thead><tr><th>Series</th><th>min</th><th>mean</th><th>max</th><th>argmax</th></tr></thead>
        <tbody id="metricRows"></tbody>
      </table>
    </section>
  </main>

  <div id="tooltip"></div>

<script>
const DATA = __DATA_JSON__;
const COLORS = ["#e6194b", "#3cb44b", "#0082c8", "#f58231", "#911eb4", "#00a7a7", "#f032e6", "#b08a00", "#111111", "#777777"];

if (!window.d3) {
  document.getElementById("d3Warning").style.display = "block";
} else {
  init();
}

function init() {
  document.getElementById("tracePath").textContent = DATA.inferDir;
  if (DATA.layers.length <= 1) {
    const warning = document.getElementById("layerWarning");
    warning.style.display = "block";
    warning.innerHTML = `当前 trace 只有 <b>${DATA.layers.length}</b> 个 layer：<code>${DATA.layers.join(", ")}</code>。跨 layer 曲线只能显示一个点；需要重新 trace 时设置 <code>PI05_TRACE_ATTN_LAYERS=0,3,6,9,12,15,17</code> 或 <code>all</code>。`;
  }

  const state = {
    row: Math.floor(DATA.gridH / 2),
    col: Math.floor(DATA.gridW / 2),
  };

  const img = d3.select("#refImage").attr("src", DATA.imageUri);
  const gridSvg = d3.select("#gridSvg");
  const tooltip = d3.select("#tooltip");
  const layerSelect = d3.select("#layerSelect");
  const stepSelect = d3.select("#stepSelect");
  const modeSelect = d3.select("#mode");
  const aggSelect = d3.select("#aggSelect");
  const qStartInput = d3.select("#qStart");
  const qEndInput = d3.select("#qEnd");
  const qStrideInput = d3.select("#qStride");
  const queryHeadSelect = d3.select("#queryHeadSelect");
  const overlayScaleSelect = d3.select("#overlayScaleSelect");
  const overlayAxisSelect = d3.select("#overlayAxisSelect");
  const headChecks = d3.select("#headChecks");
  const qBrushSvg = d3.select("#qBrushSvg");
  let suppressBrushEvent = false;

  layerSelect.selectAll("option")
    .data(DATA.layers)
    .join("option")
    .attr("value", d => d)
    .text(d => d);
  stepSelect.selectAll("option")
    .data(DATA.steps.map((s, i) => ({step: s, index: i})))
    .join("option")
    .attr("value", d => d.index)
    .text(d => d.step);
  queryHeadSelect.selectAll("option")
    .data(d3.range(DATA.numHeads))
    .join("option")
    .attr("value", d => d)
    .text(d => `head ${d}`);

  const qDefaultStart = DATA.actionQueryStart ?? 0;
  const qDefaultEnd = Math.min(DATA.suffixQueryLen - 1, qDefaultStart + (DATA.actionQueryCount || DATA.suffixQueryLen) - 1);
  qStartInput.attr("max", DATA.suffixQueryLen - 1).property("value", qDefaultStart);
  qEndInput.attr("max", DATA.suffixQueryLen - 1).property("value", qDefaultEnd);

  const headLabels = headChecks.selectAll("label")
    .data(d3.range(DATA.numHeads))
    .join("label");
  headLabels.append("input")
    .attr("type", "checkbox")
    .attr("value", d => d)
    .property("checked", true)
    .on("change", update);
  headLabels.append("span").text(d => `H${d}`);

  function showTooltip(event, html) {
    tooltip
      .style("display", "block")
      .style("left", `${event.clientX + 12}px`)
      .style("top", `${event.clientY + 12}px`)
      .html(html);
  }

  function hideTooltip() {
    tooltip.style("display", "none");
  }

  function qList() {
    const start = Math.max(0, Number(qStartInput.property("value")));
    const end = Math.min(DATA.suffixQueryLen - 1, Number(qEndInput.property("value")));
    const stride = Math.max(1, Number(qStrideInput.property("value")));
    const out = [];
    for (let q = start; q <= end; q += stride) out.push(q);
    return out;
  }

  function qRangeFromInputs() {
    const start = Math.max(0, Math.min(DATA.suffixQueryLen - 1, Number(qStartInput.property("value"))));
    const end = Math.max(0, Math.min(DATA.suffixQueryLen - 1, Number(qEndInput.property("value"))));
    return start <= end ? [start, end] : [end, start];
  }

  function setQRange(start, end, stride = 1) {
    const a = Math.max(0, Math.min(DATA.suffixQueryLen - 1, Math.round(start)));
    const b = Math.max(0, Math.min(DATA.suffixQueryLen - 1, Math.round(end)));
    qStartInput.property("value", Math.min(a, b));
    qEndInput.property("value", Math.max(a, b));
    qStrideInput.property("value", Math.max(1, Math.round(stride)));
  }

  function checkedHeads() {
    return headChecks.selectAll("input:checked").nodes().map(node => Number(node.value));
  }

  function stepIndex() {
    return Number(stepSelect.property("value"));
  }

  function fixedLayer() {
    return layerSelect.property("value");
  }

  function valueAt(layer, stepIdx, head, q, row, col) {
    return DATA.data[layer][stepIdx][head][q][row][col];
  }

  function aggregateQ(layer, stepIdx, head, qs, row, col) {
    if (qs.length === 0) return NaN;
    const mode = aggSelect.property("value");
    const values = qs.map(q => valueAt(layer, stepIdx, head, q, row, col));
    if (mode === "max") return d3.max(values);
    if (mode === "single") return values[0];
    return d3.mean(values);
  }

  function selectedToken() {
    return state.row * DATA.gridW + state.col;
  }

  function drawGrid() {
    const imgNode = img.node();
    const width = imgNode.clientWidth || 500;
    const height = imgNode.clientHeight || Math.round(width * 0.75);
    gridSvg.attr("viewBox", `0 0 ${width} ${height}`);
    const cw = width / DATA.gridW;
    const ch = height / DATA.gridH;
    const cells = [];
    for (let r = 0; r < DATA.gridH; r++) {
      for (let c = 0; c < DATA.gridW; c++) {
        cells.push({row: r, col: c, token: r * DATA.gridW + c});
      }
    }
    gridSvg.selectAll("rect.blockRect")
      .data(cells, d => d.token)
      .join("rect")
      .attr("class", d => `blockRect ${d.row === state.row && d.col === state.col ? "selected" : ""}`)
      .attr("x", d => d.col * cw)
      .attr("y", d => d.row * ch)
      .attr("width", cw)
      .attr("height", ch)
      .on("mouseenter", (event, d) => {
        const score = aggregateQ(fixedLayer(), stepIndex(), Number(queryHeadSelect.property("value")), qList(), d.row, d.col);
        showTooltip(event, `block r=${d.row}, c=${d.col}<br>local token=${d.token}<br>score=${formatScore(score)}<br><span style="color:#c8d1e0">click to select</span>`);
      })
      .on("mousemove", (event, d) => {
        const score = aggregateQ(fixedLayer(), stepIndex(), Number(queryHeadSelect.property("value")), qList(), d.row, d.col);
        showTooltip(event, `block r=${d.row}, c=${d.col}<br>local token=${d.token}<br>score=${formatScore(score)}<br><span style="color:#c8d1e0">click to select</span>`);
      })
      .on("mouseleave", hideTooltip)
      .on("click", (event, d) => {
        state.row = d.row;
        state.col = d.col;
        update();
      });
  }

  function drawQSelector() {
    const width = qBrushSvg.node().clientWidth || 900;
    const height = 64;
    const margin = {left: 12, right: 12, top: 10, bottom: 18};
    const n = DATA.suffixQueryLen;
    const innerW = width - margin.left - margin.right;
    const cellGap = 2;
    const cellW = Math.max(8, (innerW - cellGap * (n - 1)) / n);
    const cellH = 25;
    const [qStart, qEnd] = qRangeFromInputs();
    const selected = new Set(qList());
    qBrushSvg.attr("viewBox", `0 0 ${width} ${height}`);

    const color = d3.scaleSequential(d3.interpolateTurbo).domain([0, Math.max(1, n - 1)]);
    const cells = d3.range(n).map(q => ({
      q,
      x: margin.left + q * (cellW + cellGap),
      y: margin.top,
      selected: selected.has(q),
    }));

    function qFromX(x) {
      const clamped = Math.max(margin.left, Math.min(margin.left + innerW, x));
      const raw = Math.round((clamped - margin.left) / (cellW + cellGap));
      return Math.max(0, Math.min(n - 1, raw));
    }

    qBrushSvg.selectAll("rect.qCell")
      .data(cells, d => d.q)
      .join("rect")
      .attr("class", d => `qCell ${d.selected ? "selected" : ""}`)
      .attr("x", d => d.x)
      .attr("y", d => d.y)
      .attr("width", cellW)
      .attr("height", cellH)
      .attr("fill", d => d.selected ? color(d.q) : "#eef1f5")
      .attr("opacity", d => d.selected ? 0.95 : 0.55)
      .on("mouseenter", (event, d) => {
        showTooltip(event, `q${d.q}<br>${d.selected ? "selected" : "not selected"}<br><span style="color:#c8d1e0">click to select only this q; drag to brush a range</span>`);
      })
      .on("mousemove", (event, d) => {
        showTooltip(event, `q${d.q}<br>${d.selected ? "selected" : "not selected"}<br><span style="color:#c8d1e0">click to select only this q; drag to brush a range</span>`);
      })
      .on("mouseleave", hideTooltip)
      .on("click", (event, d) => {
        setQRange(d.q, d.q, 1);
        update();
      });

    qBrushSvg.selectAll("text.qLabel")
      .data(cells, d => d.q)
      .join("text")
      .attr("class", "qLabel")
      .attr("x", d => d.x + cellW / 2)
      .attr("y", margin.top + cellH + 12)
      .text(d => d.q);

    const brushExtent = [[margin.left, margin.top], [margin.left + innerW, margin.top + cellH]];
    const brush = d3.brushX()
      .extent(brushExtent)
      .on("end", event => {
        if (suppressBrushEvent) return;
        if (!event.selection) {
          if (!event.sourceEvent) return;
          const [x] = d3.pointer(event.sourceEvent, qBrushSvg.node());
          const q = qFromX(x);
          setQRange(q, q, 1);
          update();
          return;
        }
        const [x0, x1] = event.selection;
        const selectedQs = cells
          .filter(d => d.x + cellW >= x0 && d.x <= x1)
          .map(d => d.q);
        if (!selectedQs.length) return;
        setQRange(d3.min(selectedQs), d3.max(selectedQs), 1);
        update();
      });

    let brushG = qBrushSvg.select("g.qBrush");
    if (brushG.empty()) {
      brushG = qBrushSvg.append("g").attr("class", "qBrush");
    }
    brushG.call(brush);
    brushG.raise();
    const selectedCells = cells.filter(d => d.q >= qStart && d.q <= qEnd);
    if (selectedCells.length) {
      const x0 = d3.min(selectedCells, d => d.x);
      const x1 = d3.max(selectedCells, d => d.x + cellW);
      suppressBrushEvent = true;
      brushG.call(brush.move, [x0, x1]);
      suppressBrushEvent = false;
    }

    document.getElementById("qSelectionText").textContent =
      `selected q tokens: ${qList().join(", ")} (${qList().length} token${qList().length === 1 ? "" : "s"})`;
  }

  function formatScore(value) {
    if (!Number.isFinite(value)) return "nan";
    if (Math.abs(value) < 0.001) return value.toExponential(4);
    return value.toFixed(6);
  }

  function seriesStats(series, labels) {
    return series.map(s => {
      const finite = s.values.filter(Number.isFinite);
      const maxValue = d3.max(finite);
      const argmax = s.values.indexOf(maxValue);
      return {
        name: s.name,
        min: d3.min(finite),
        mean: d3.mean(finite),
        max: maxValue,
        argmax: labels[argmax],
      };
    });
  }

  function buildSeries() {
    const mode = modeSelect.property("value");
    const layer = fixedLayer();
    const step = stepIndex();
    const heads = checkedHeads();
    const qs = qList();
    const qHead = Number(queryHeadSelect.property("value"));
    const row = state.row;
    const col = state.col;

    let labels;
    let headSeries;
    let querySeries;
    if (mode === "step") {
      labels = DATA.steps.map(String);
      headSeries = heads.map(h => ({
        name: `H${h}`,
        color: COLORS[h % COLORS.length],
        values: DATA.steps.map((_, si) => aggregateQ(layer, si, h, qs, row, col)),
      }));
      querySeries = qs.map((q, i) => ({
        name: `q${q}`,
        color: d3.interpolateTurbo(qs.length <= 1 ? 0.5 : i / (qs.length - 1)),
        values: DATA.steps.map((_, si) => valueAt(layer, si, qHead, q, row, col)),
      }));
    } else {
      labels = DATA.layers;
      headSeries = heads.map(h => ({
        name: `H${h}`,
        color: COLORS[h % COLORS.length],
        values: DATA.layers.map(l => aggregateQ(l, step, h, qs, row, col)),
      }));
      querySeries = qs.map((q, i) => ({
        name: `q${q}`,
        color: d3.interpolateTurbo(qs.length <= 1 ? 0.5 : i / (qs.length - 1)),
        values: DATA.layers.map(l => valueAt(l, step, qHead, q, row, col)),
      }));
    }
    return {labels, headSeries, querySeries, qs};
  }

  function drawChart(svgSelector, series, labels, legendSelector, titlePrefix) {
    const svg = d3.select(svgSelector);
    const node = svg.node();
    const width = node.clientWidth || 900;
    const height = node.clientHeight || 330;
    const margin = {top: 22, right: 22, bottom: 44, left: 62};
    svg.attr("viewBox", `0 0 ${width} ${height}`);
    svg.selectAll("*").remove();

    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    const allValues = series.flatMap(s => s.values).filter(Number.isFinite);
    const ymax = Math.max(d3.max(allValues) || 0, 1e-12);
    const x = d3.scalePoint().domain(labels).range([margin.left, margin.left + innerW]).padding(0.35);
    const y = d3.scaleLinear().domain([0, ymax * 1.08]).nice().range([margin.top + innerH, margin.top]);

    svg.append("g")
      .attr("class", "gridLine")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).tickSize(-innerW).tickFormat(""));
    svg.append("g")
      .attr("class", "axis")
      .attr("transform", `translate(0,${margin.top + innerH})`)
      .call(d3.axisBottom(x));
    svg.append("g")
      .attr("class", "axis")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".2e")));

    svg.append("text")
      .attr("x", margin.left)
      .attr("y", 15)
      .attr("font-size", 12)
      .attr("fill", "#667085")
      .text(`${titlePrefix}; selected block r=${state.row}, c=${state.col}, local token=${selectedToken()}`);

    const line = d3.line()
      .defined((d) => Number.isFinite(d.value))
      .x(d => x(d.label))
      .y(d => y(d.value));

    const groups = svg.selectAll("g.series")
      .data(series)
      .join("g")
      .attr("class", "series");

    groups.append("path")
      .attr("class", "linePath")
      .attr("stroke", d => d.color)
      .attr("d", d => line(labels.map((label, i) => ({label, value: d.values[i]}))));

    groups.selectAll("circle.dot")
      .data(d => labels.map((label, i) => ({series: d.name, color: d.color, label, value: d.values[i]})))
      .join("circle")
      .attr("class", "dot")
      .attr("r", 3.4)
      .attr("cx", d => x(d.label))
      .attr("cy", d => y(Number.isFinite(d.value) ? d.value : 0))
      .attr("fill", d => d.color)
      .on("mouseenter", (event, d) => {
        showTooltip(event, `${d.series}<br>${modeSelect.property("value")}=${d.label}<br>attn_score=${formatScore(d.value)}<br>block r=${state.row}, c=${state.col}`);
      })
      .on("mousemove", (event, d) => {
        showTooltip(event, `${d.series}<br>${modeSelect.property("value")}=${d.label}<br>attn_score=${formatScore(d.value)}<br>block r=${state.row}, c=${state.col}`);
      })
      .on("mouseleave", hideTooltip);

    const legend = d3.select(legendSelector);
    legend.selectAll("*").remove();
    legend.selectAll("span.legendItem")
      .data(series)
      .join("span")
      .attr("class", "legendItem")
      .html(d => `<span class="swatch" style="background:${d.color}"></span>${htmlEscape(d.name)}`);
  }

  function drawHeadHeatmaps(heads, qs) {
    const container = d3.select("#headHeatmaps");
    const cards = container.selectAll("div.heatmapCard")
      .data(heads, d => d)
      .join(
        enter => {
          const card = enter.append("div").attr("class", "heatmapCard");
          card.append("h4");
          card.append("svg").attr("class", "heatmapSvg");
          return card;
        },
        update => update,
        exit => exit.remove()
      );

    const allValues = [];
    for (const h of heads) {
      for (let si = 0; si < DATA.steps.length; si++) {
        for (const layer of DATA.layers) {
          allValues.push(aggregateQ(layer, si, h, qs, state.row, state.col));
        }
      }
    }
    const vmax = Math.max(d3.max(allValues.filter(Number.isFinite)) || 0, 1e-12);
    const color = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, vmax]);

    cards.each(function(head) {
      const card = d3.select(this);
      card.select("h4").text(`Head ${head} · q=[${qs.join(",")}]`);
      const svg = card.select("svg");
      const node = svg.node();
      const width = node.clientWidth || 320;
      const height = 230;
      const margin = {top: 12, right: 18, bottom: 42, left: 42};
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;
      svg.attr("viewBox", `0 0 ${width} ${height}`);
      svg.selectAll("*").remove();

      const x = d3.scaleBand().domain(DATA.layers).range([margin.left, margin.left + innerW]).padding(0.04);
      const y = d3.scaleBand().domain(DATA.steps.map(String)).range([margin.top, margin.top + innerH]).padding(0.04);
      const cells = [];
      for (let si = 0; si < DATA.steps.length; si++) {
        for (const layer of DATA.layers) {
          cells.push({
            head,
            stepIndex: si,
            step: DATA.steps[si],
            layer,
            value: aggregateQ(layer, si, head, qs, state.row, state.col),
          });
        }
      }

      svg.selectAll("rect.heatCell")
        .data(cells)
        .join("rect")
        .attr("class", "heatCell")
        .attr("x", d => x(d.layer))
        .attr("y", d => y(String(d.step)))
        .attr("width", x.bandwidth())
        .attr("height", y.bandwidth())
        .attr("fill", d => color(d.value))
        .on("mouseenter", (event, d) => {
          showTooltip(event, `Head ${d.head}<br>step=${d.step}<br>layer=${d.layer}<br>q=[${qs.join(",")}]<br>attn_score=${formatScore(d.value)}<br>block r=${state.row}, c=${state.col}`);
        })
        .on("mousemove", (event, d) => {
          showTooltip(event, `Head ${d.head}<br>step=${d.step}<br>layer=${d.layer}<br>q=[${qs.join(",")}]<br>attn_score=${formatScore(d.value)}<br>block r=${state.row}, c=${state.col}`);
        })
        .on("mouseleave", hideTooltip);

      svg.append("g")
        .attr("class", "axis")
        .attr("transform", `translate(0,${margin.top + innerH})`)
        .call(d3.axisBottom(x).tickValues(DATA.layers.filter((_, i) => i === 0 || i === DATA.layers.length - 1 || i % Math.ceil(DATA.layers.length / 6) === 0)))
        .selectAll("text")
        .attr("transform", "rotate(-35)")
        .attr("text-anchor", "end");
      svg.append("g")
        .attr("class", "axis")
        .attr("transform", `translate(${margin.left},0)`)
        .call(d3.axisLeft(y));
      svg.append("text")
        .attr("x", margin.left)
        .attr("y", height - 5)
        .attr("font-size", 11)
        .attr("fill", "#667085")
        .text(`max scale=${formatScore(vmax)}`);
    });
  }

  function drawOverlaySequence(qs) {
    const mode = modeSelect.property("value");
    const layer = fixedLayer();
    const step = stepIndex();
    const head = Number(queryHeadSelect.property("value"));
    const axis = overlayAxisSelect.property("value");
    const items = axis === "q"
      ? qs.map(q => ({label: `q_${String(q).padStart(2, "0")}`, layer, stepIndex: step, step: DATA.steps[step], qOverride: q}))
      : mode === "layer"
        ? DATA.layers.map(l => ({label: l, layer: l, stepIndex: step, step: DATA.steps[step], qOverride: null}))
        : DATA.steps.map((s, si) => ({label: `step_${String(s).padStart(2, "0")}`, layer, stepIndex: si, step: s, qOverride: null}));

    function finitePositive(values) {
      return values.filter(v => Number.isFinite(v) && v > 0).sort((a, b) => a - b);
    }
    function percentile(values, pct) {
      const finite = finitePositive(values);
      if (!finite.length) return 1e-12;
      const idx = Math.max(0, Math.min(finite.length - 1, Math.ceil((pct / 100) * finite.length) - 1));
      return Math.max(finite[idx], 1e-12);
    }

    const allValues = [];
    const itemScale = new Map();
    for (const item of items) {
      const values = [];
      for (let r = 0; r < DATA.gridH; r++) {
        for (let c = 0; c < DATA.gridW; c++) {
          const itemQs = item.qOverride === null ? qs : [item.qOverride];
          const value = aggregateQ(item.layer, item.stepIndex, head, itemQs, r, c);
          values.push(value);
          allValues.push(value);
        }
      }
      itemScale.set(item.label, {
        max: Math.max(d3.max(values.filter(Number.isFinite)) || 0, 1e-12),
        p95: percentile(values, 95),
      });
    }
    const sequenceScale = {
      max: Math.max(d3.max(allValues.filter(Number.isFinite)) || 0, 1e-12),
      p95: percentile(allValues, 95),
    };

    const cards = d3.select("#overlaySequence")
      .selectAll("div.overlayCard")
      .data(items, d => d.label)
      .join(
        enter => {
          const card = enter.append("div").attr("class", "overlayCard");
          card.append("h4");
          const stage = card.append("div").attr("class", "overlayStage");
          stage.append("img").attr("src", DATA.imageUri);
          stage.append("svg");
          card.append("div").attr("class", "subtle");
          return card;
        },
        update => update,
        exit => exit.remove()
      );

    cards.each(function(item) {
      const card = d3.select(this);
      card.select("h4").text(`${item.label} · H${head}`);
      const scaleMode = overlayScaleSelect.property("value");
      const frameScale = itemScale.get(item.label);
      const vmax = scaleMode === "frame_p95" ? frameScale.p95
        : scaleMode === "frame_max" ? frameScale.max
        : scaleMode === "sequence_p95" ? sequenceScale.p95
        : sequenceScale.max;
      const itemQs = item.qOverride === null ? qs : [item.qOverride];
      card.select("div.subtle").text(`step=${item.step}, layer=${item.layer}, q=[${itemQs.join(",")}], ${scaleMode} scale=${formatScore(vmax)}`);
      const svg = card.select("svg");
      const stage = card.select(".overlayStage").node();
      const width = stage.clientWidth || 260;
      const imgRatio = 0.75;
      const height = Math.round(width * imgRatio);
      svg.attr("viewBox", `0 0 ${width} ${height}`);

      const cw = width / DATA.gridW;
      const ch = height / DATA.gridH;
      const cells = [];
      for (let r = 0; r < DATA.gridH; r++) {
        for (let c = 0; c < DATA.gridW; c++) {
          const value = aggregateQ(item.layer, item.stepIndex, head, itemQs, r, c);
          cells.push({row: r, col: c, token: r * DATA.gridW + c, value});
        }
      }

      svg.selectAll("rect.overlayCell")
        .data(cells, d => d.token)
        .join("rect")
        .attr("class", "overlayCell")
        .attr("x", d => d.col * cw)
        .attr("y", d => d.row * ch)
        .attr("width", cw)
        .attr("height", ch)
        .attr("fill", "#ff1800")
        .attr("opacity", d => {
          const norm = Math.min(1, Math.max(0, d.value / vmax));
          return Math.min(0.65, Math.max(0, norm * 0.65));
        })
        .on("mouseenter", (event, d) => {
          showTooltip(event, `${item.label}<br>Head ${head}<br>step=${item.step}<br>layer=${item.layer}<br>q=[${itemQs.join(",")}]<br>block r=${d.row}, c=${d.col}<br>token=${d.token}<br>attn_score=${formatScore(d.value)}`);
        })
        .on("mousemove", (event, d) => {
          showTooltip(event, `${item.label}<br>Head ${head}<br>step=${item.step}<br>layer=${item.layer}<br>q=[${itemQs.join(",")}]<br>block r=${d.row}, c=${d.col}<br>token=${d.token}<br>attn_score=${formatScore(d.value)}`);
        })
        .on("mouseleave", hideTooltip);
    });
  }

  function htmlEscape(text) {
    return String(text).replace(/[&<>"']/g, ch => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[ch]));
  }

  function update() {
    drawGrid();
    drawQSelector();
    const {labels, headSeries, querySeries, qs} = buildSeries();
    const agg = aggSelect.property("value");
    const mode = modeSelect.property("value");
    const head = Number(queryHeadSelect.property("value"));
    drawChart("#headChart", headSeries, labels, "#headLegend", `Per-head ${agg} score over q=[${qs.join(",")}]`);
    drawChart("#queryChart", querySeries, labels, "#queryLegend", `Per-q score for head ${head}`);
    drawHeadHeatmaps(checkedHeads(), qs);
    drawOverlaySequence(qs);

    document.getElementById("blockText").textContent =
      `selected block row=${state.row}, col=${state.col}, local token=${selectedToken()}, block-size=${DATA.blockSize}`;
    document.getElementById("metaText").textContent =
      `image=${DATA.imageKey}, grid=${DATA.gridH}x${DATA.gridW}, heads=${DATA.numHeads}, q=${DATA.suffixQueryLen}, state_q=${DATA.stateQueryCount}, action_q_start=${DATA.actionQueryStart}, mode=${mode}`;

    const top = seriesStats(headSeries, labels).sort((a, b) => b.max - a.max).slice(0, 6);
    d3.select("#summary").html(top.map(d => `<span>${htmlEscape(d.name)} max=${formatScore(d.max)} @ ${htmlEscape(d.argmax)}</span>`).join(""));
    const rows = seriesStats(headSeries.concat(querySeries.slice(0, 12)), labels);
    d3.select("#metricRows")
      .selectAll("tr")
      .data(rows)
      .join("tr")
      .html(d => `<td>${htmlEscape(d.name)}</td><td>${formatScore(d.min)}</td><td>${formatScore(d.mean)}</td><td>${formatScore(d.max)}</td><td>${htmlEscape(d.argmax)}</td>`);
  }

  img.on("load", () => {
    drawGrid();
    update();
  });
  window.addEventListener("resize", update);
  [layerSelect, stepSelect, modeSelect, aggSelect, qStartInput, qEndInput, qStrideInput, queryHeadSelect, overlayScaleSelect, overlayAxisSelect].forEach(sel => sel.on("input", update));
  update();
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()

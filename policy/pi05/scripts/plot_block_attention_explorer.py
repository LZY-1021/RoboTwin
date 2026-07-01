#!/usr/bin/env python3
"""Build an interactive block-level denoise attention explorer.

The generated HTML lets you click a 16x16 image patch/block and inspect how
its attention score changes over denoise steps or layers for each head/query.
It expects traces saved with PI05_TRACE_SAVE_ATTN_FULL=1.
"""

from __future__ import annotations

import argparse
import base64
import html
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Trace root, episode dir, or infer dir.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--infer", type=int, default=0)
    parser.add_argument("--steps", default="all")
    parser.add_argument("--layers", default="all", help="all, or comma-separated layer ids like 0,3,12.")
    parser.add_argument("--image-key", default="base_0_rgb")
    parser.add_argument("--block-size", type=int, default=1)
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
    full_layers = first_payload.get("full_layers")
    if not full_layers:
        raise ValueError("Trace does not contain full_layers. Re-run with PI05_TRACE_SAVE_ATTN_FULL=1.")
    if args.layers.strip().lower() in {"", "all"}:
        layer_names = sorted(full_layers.keys())
    else:
        layer_names = [parse_layer(item) for item in args.layers.split(",") if item.strip()]

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

    layer_data: dict[str, list[Any]] = {}
    shape_info = None
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
            # [heads, queries, grid_h, grid_w]
            if shape_info is None:
                shape_info = arr.shape
            elif arr.shape != shape_info:
                raise ValueError(f"Shape mismatch at {layer_name} step {step}: {arr.shape} vs {shape_info}")
            step_arrays.append(arr)
        # [steps, heads, queries, grid_h, grid_w]
        stacked = np.stack(step_arrays, axis=0)
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
        layer_data[layer_name] = round_nested(stacked)

    if shape_info is None:
        raise ValueError("No attention tensors loaded.")
    num_heads, suffix_query_len, grid_h, grid_w = shape_info
    if args.block_size > 1:
        view_grid_h = (grid_h + args.block_size - 1) // args.block_size
        view_grid_w = (grid_w + args.block_size - 1) // args.block_size
    else:
        view_grid_h, view_grid_w = grid_h, grid_w

    payload_json = {
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
        "gridH": view_grid_h,
        "gridW": view_grid_w,
        "sourceGridH": grid_h,
        "sourceGridW": grid_w,
        "blockSize": args.block_size,
        "data": layer_data,
    }

    script_json = json.dumps(payload_json, ensure_ascii=False)
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Block Attention Explorer</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 18px; color: #222; }}
    .layout {{ display: grid; grid-template-columns: 420px minmax(760px, 1fr); gap: 18px; align-items: start; }}
    .panel {{ border: 1px solid #ddd; padding: 12px; background: #fff; }}
    .image-wrap {{ position: relative; width: 384px; }}
    #refImage {{ width: 384px; display: block; border: 1px solid #ccc; }}
    #gridCanvas {{ position: absolute; left: 0; top: 0; width: 384px; height: 384px; cursor: crosshair; }}
    label {{ margin-right: 12px; white-space: nowrap; }}
    select, input {{ margin: 4px; }}
    svg {{ border: 1px solid #ddd; background: #fafafa; }}
    .small {{ color: #666; font-size: 13px; }}
    .chips span {{ display: inline-block; padding: 2px 6px; margin: 2px; border-radius: 4px; background: #eee; }}
    .headChecks label {{ display: inline-block; width: 58px; }}
  </style>
</head>
<body>
  <h1>Block Attention Explorer</h1>
  <p><code>{html.escape(str(infer_dir))}</code></p>
  <div class="layout">
    <div class="panel">
      <h2>Pick Block</h2>
      <div class="image-wrap">
        <img id="refImage" src="">
        <canvas id="gridCanvas" width="384" height="384"></canvas>
      </div>
      <p id="blockText" class="small"></p>
      <p class="small">Click any image block. For block-size &gt; 1, the score is averaged over source patches inside that block.</p>
    </div>
    <div class="panel">
      <h2>Controls</h2>
      <div>
        <label>Mode
          <select id="mode">
            <option value="step">x-axis: denoise step</option>
            <option value="layer">x-axis: layer</option>
          </select>
        </label>
        <label>Layer <select id="layerSelect"></select></label>
        <label>Step <select id="stepSelect"></select></label>
        <label>Head for per-q curves <select id="queryHeadSelect"></select></label>
      </div>
      <div>
        <label>Q start <input id="qStart" type="number" value="0" min="0" style="width:64px"></label>
        <label>Q end <input id="qEnd" type="number" value="31" min="0" style="width:64px"></label>
        <label>Q stride <input id="qStride" type="number" value="4" min="1" style="width:64px"></label>
      </div>
      <div class="headChecks" id="headChecks"></div>
      <p class="small" id="metaText"></p>
      <h3>Per-head mean over selected q</h3>
      <svg id="headChart" width="900" height="300"></svg>
      <h3>Per-q curves for selected head</h3>
      <svg id="queryChart" width="900" height="300"></svg>
      <h3>Selected q summary</h3>
      <div id="summary" class="chips"></div>
    </div>
  </div>
<script>
const DATA = {script_json};
const img = document.getElementById('refImage');
const canvas = document.getElementById('gridCanvas');
const ctx = canvas.getContext('2d');
const layerSelect = document.getElementById('layerSelect');
const stepSelect = document.getElementById('stepSelect');
const modeSelect = document.getElementById('mode');
const queryHeadSelect = document.getElementById('queryHeadSelect');
const qStartInput = document.getElementById('qStart');
const qEndInput = document.getElementById('qEnd');
const qStrideInput = document.getElementById('qStride');
const headChecks = document.getElementById('headChecks');
let selected = {{row: Math.floor(DATA.gridH / 2), col: Math.floor(DATA.gridW / 2)}};

img.src = DATA.imageUri;
DATA.layers.forEach(l => layerSelect.add(new Option(l, l)));
DATA.steps.forEach((s, i) => stepSelect.add(new Option(String(s), String(i))));
for (let h = 0; h < DATA.numHeads; h++) {{
  queryHeadSelect.add(new Option('H' + h, String(h)));
  const label = document.createElement('label');
  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.value = h;
  cb.checked = true;
  cb.addEventListener('change', update);
  label.appendChild(cb);
  label.appendChild(document.createTextNode(' H' + h));
  headChecks.appendChild(label);
}}
qStartInput.value = DATA.actionQueryStart;
qEndInput.value = Math.min(DATA.suffixQueryLen - 1, DATA.actionQueryStart + DATA.actionQueryCount - 1);

function getLayer() {{ return layerSelect.value; }}
function getStepIndex() {{ return Number(stepSelect.value); }}
function checkedHeads() {{
  return Array.from(headChecks.querySelectorAll('input:checked')).map(x => Number(x.value));
}}
function qList() {{
  const start = Math.max(0, Number(qStartInput.value));
  const end = Math.min(DATA.suffixQueryLen - 1, Number(qEndInput.value));
  const stride = Math.max(1, Number(qStrideInput.value));
  const out = [];
  for (let q = start; q <= end; q += stride) out.push(q);
  return out;
}}
function valueAt(layer, stepIdx, head, q, row, col) {{
  return DATA.data[layer][stepIdx][head][q][row][col];
}}
function drawGrid() {{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const cw = canvas.width / DATA.gridW;
  const ch = canvas.height / DATA.gridH;
  ctx.strokeStyle = 'rgba(255,255,255,0.75)';
  ctx.lineWidth = 1;
  for (let r=1;r<DATA.gridH;r++) {{ ctx.beginPath(); ctx.moveTo(0,r*ch); ctx.lineTo(canvas.width,r*ch); ctx.stroke(); }}
  for (let c=1;c<DATA.gridW;c++) {{ ctx.beginPath(); ctx.moveTo(c*cw,0); ctx.lineTo(c*cw,canvas.height); ctx.stroke(); }}
  ctx.strokeStyle = '#0066ff';
  ctx.lineWidth = 3;
  ctx.strokeRect(selected.col*cw, selected.row*ch, cw, ch);
  document.getElementById('blockText').textContent = `selected block row=${{selected.row}}, col=${{selected.col}}, source block-size=${{DATA.blockSize}}`;
}}
canvas.addEventListener('click', e => {{
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  selected.col = Math.max(0, Math.min(DATA.gridW - 1, Math.floor(x / rect.width * DATA.gridW)));
  selected.row = Math.max(0, Math.min(DATA.gridH - 1, Math.floor(y / rect.height * DATA.gridH)));
  update();
}});

function lineChart(svgId, series, xLabels, yLabel) {{
  const svg = document.getElementById(svgId);
  const W = Number(svg.getAttribute('width')), H = Number(svg.getAttribute('height'));
  const m = {{l:55,r:18,t:18,b:40}};
  const innerW = W - m.l - m.r, innerH = H - m.t - m.b;
  svg.innerHTML = '';
  const vals = series.flatMap(s => s.y).filter(v => Number.isFinite(v));
  const yMax = Math.max(...vals, 1e-12);
  const yMin = 0;
  const xN = Math.max(1, xLabels.length - 1);
  function sx(i) {{ return m.l + innerW * (i / xN); }}
  function sy(v) {{ return m.t + innerH * (1 - (v - yMin) / (yMax - yMin + 1e-12)); }}
  function el(name, attrs) {{
    const node = document.createElementNS('http://www.w3.org/2000/svg', name);
    for (const [k,v] of Object.entries(attrs)) node.setAttribute(k, v);
    svg.appendChild(node);
    return node;
  }}
  el('line', {{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,stroke:'#888'}});
  el('line', {{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,stroke:'#888'}});
  for (let i=0;i<xLabels.length;i++) {{
    const x = sx(i);
    if (i === 0 || i === xLabels.length-1 || i % Math.ceil(xLabels.length/8) === 0) {{
      el('text', {{x:x, y:H-14, 'text-anchor':'middle', 'font-size':'11', fill:'#555'}}).textContent = xLabels[i];
    }}
  }}
  [0,0.25,0.5,0.75,1].forEach(t => {{
    const y = m.t + innerH * (1-t);
    el('line', {{x1:m.l,y1:y,x2:W-m.r,y2:y,stroke:'#eee'}});
    el('text', {{x:m.l-8,y:y+4,'text-anchor':'end','font-size':'11',fill:'#555'}}).textContent = (yMax*t).toExponential(1);
  }});
  const palette = ['#e6194b','#3cb44b','#0082c8','#f58231','#911eb4','#46f0f0','#f032e6','#d2b414','#000','#888'];
  series.forEach((s, si) => {{
    const color = s.color || palette[si % palette.length];
    let d = '';
    s.y.forEach((v,i) => {{
      const cmd = i === 0 ? 'M' : 'L';
      d += `${{cmd}} ${{sx(i).toFixed(1)}} ${{sy(v).toFixed(1)}} `;
    }});
    el('path', {{d:d, fill:'none', stroke:color, 'stroke-width':2, opacity:0.9}});
    const lx = W - m.r - 95;
    const ly = m.t + 14 + si * 16;
    if (si < 14) {{
      el('line', {{x1:lx,y1:ly-4,x2:lx+18,y2:ly-4,stroke:color,'stroke-width':3}});
      el('text', {{x:lx+22,y:ly,'font-size':'12',fill:'#333'}}).textContent = s.name;
    }}
  }});
}}

function update() {{
  drawGrid();
  const mode = modeSelect.value;
  const layer = getLayer();
  const stepIdx = getStepIndex();
  const heads = checkedHeads();
  const qs = qList();
  const row = selected.row, col = selected.col;
  const qHead = Number(queryHeadSelect.value);
  let xLabels = [];
  let headSeries = [];
  let querySeries = [];
  if (mode === 'step') {{
    xLabels = DATA.steps.map(String);
    headSeries = heads.map(h => ({{
      name:'H'+h,
      y: DATA.steps.map((_, si) => qs.reduce((a,q)=>a+valueAt(layer, si, h, q, row, col), 0) / Math.max(1, qs.length))
    }}));
    querySeries = qs.map(q => ({{
      name:'q'+q,
      y: DATA.steps.map((_, si) => valueAt(layer, si, qHead, q, row, col))
    }}));
  }} else {{
    xLabels = DATA.layers;
    headSeries = heads.map(h => ({{
      name:'H'+h,
      y: DATA.layers.map(l => qs.reduce((a,q)=>a+valueAt(l, stepIdx, h, q, row, col), 0) / Math.max(1, qs.length))
    }}));
    querySeries = qs.map(q => ({{
      name:'q'+q,
      y: DATA.layers.map(l => valueAt(l, stepIdx, qHead, q, row, col))
    }}));
  }}
  lineChart('headChart', headSeries, xLabels, 'score');
  lineChart('queryChart', querySeries, xLabels, 'score');
  const maxHead = headSeries.map(s => [s.name, Math.max(...s.y)]).sort((a,b)=>b[1]-a[1]).slice(0,5);
  document.getElementById('summary').innerHTML = maxHead.map(x => `<span>${{x[0]}} max=${{x[1].toExponential(3)}}</span>`).join('');
  document.getElementById('metaText').textContent =
    `image=${{DATA.imageKey}}, grid=${{DATA.gridH}}x${{DATA.gridW}}, heads=${{DATA.numHeads}}, q=${{DATA.suffixQueryLen}}, state_q=${{DATA.stateQueryCount}}, action_q_start=${{DATA.actionQueryStart}}`;
}}
[layerSelect, stepSelect, modeSelect, queryHeadSelect, qStartInput, qEndInput, qStrideInput].forEach(el => el.addEventListener('input', update));
img.onload = update;
update();
</script>
</body>
</html>
"""
    (args.out_dir / "index.html").write_text(page, encoding="utf-8")
    (args.out_dir / "data_summary.json").write_text(
        json.dumps({k: v for k, v in payload_json.items() if k != "data"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()

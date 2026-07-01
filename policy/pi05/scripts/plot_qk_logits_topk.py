#!/usr/bin/env python3
"""Build a top-k block QK/logits analysis page.

The page compares two traced denoise forwards:
  full_delta_logits = Q_b K_b^T - Q_a K_a^T
  delta_q_term      = (Q_b - Q_a) K_a^T

It then lets the user choose top-k image blocks and compares the selected-block
approximation with the full image-logit delta.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from plot_denoise_head_query_attention import find_image_meta, infer_dir_from_trace, parse_layer
from plot_patch_kv_overlay import image_hwc


def image_data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def load_qk(infer_dir: Path, step: int) -> dict[str, Any]:
    path = infer_dir / f"qk_denoise_step_{step:03d}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Re-run trace with PI05_TRACE_SAVE_QK_LOGITS=1.")
    return torch.load(path, map_location="cpu")


def load_attn(infer_dir: Path, step: int) -> dict[str, Any] | None:
    path = infer_dir / f"attn_denoise_step_{step:03d}.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def flatten_blocks(values: np.ndarray, block_size: int) -> tuple[np.ndarray, list[list[int]], int, int]:
    grid_h, grid_w = values.shape
    block_h = math.ceil(grid_h / block_size)
    block_w = math.ceil(grid_w / block_size)
    block_values = np.zeros((block_h, block_w), dtype=np.float32)
    block_indices: list[list[int]] = []
    for br in range(block_h):
        for bc in range(block_w):
            r0 = br * block_size
            r1 = min(grid_h, r0 + block_size)
            c0 = bc * block_size
            c1 = min(grid_w, c0 + block_size)
            patch = values[r0:r1, c0:c1]
            block_values[br, bc] = float(np.mean(patch))
            indices = [r * grid_w + c for r in range(r0, r1) for c in range(c0, c1)]
            block_indices.append(indices)
    return block_values, block_indices, block_h, block_w


def round_array(values: np.ndarray, digits: int = 7) -> list[Any]:
    return np.round(values.astype(np.float32), digits).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_a", type=Path, help="Trace root, episode dir, or infer dir for source A.")
    parser.add_argument("trace_b", type=Path, help="Trace root, episode dir, or infer dir for source B.")
    parser.add_argument("--episode-a", type=int, default=0)
    parser.add_argument("--infer-a", type=int, default=0)
    parser.add_argument("--episode-b", type=int, default=1)
    parser.add_argument("--infer-b", type=int, default=0)
    parser.add_argument("--step", type=int, default=9)
    parser.add_argument("--layer", default="12")
    parser.add_argument("--image-key", default="base_0_rgb")
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--q", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--topk-default", type=int, default=8)
    parser.add_argument(
        "--topk-source",
        choices=["attn_b", "attn_a", "abs_full_delta", "abs_delta_q"],
        default="attn_b",
        help="Score used to rank blocks for top-k selection.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    infer_a = args.trace_a if args.trace_a.name.startswith("infer_") else infer_dir_from_trace(args.trace_a, args.episode_a, args.infer_a)
    infer_b = args.trace_b if args.trace_b.name.startswith("infer_") else infer_dir_from_trace(args.trace_b, args.episode_b, args.infer_b)
    layer_name = parse_layer(args.layer)

    qk_a = load_qk(infer_a, args.step)
    qk_b = load_qk(infer_b, args.step)
    layer_a = qk_a["layers"][layer_name]
    layer_b = qk_b["layers"][layer_name]
    if args.image_key not in layer_a["image_logits"]:
        raise KeyError(f"{args.image_key} missing in A qk image_logits. Available: {sorted(layer_a['image_logits'])}")
    if args.image_key not in layer_b["image_logits"]:
        raise KeyError(f"{args.image_key} missing in B qk image_logits. Available: {sorted(layer_b['image_logits'])}")

    logits_a = layer_a["image_logits"][args.image_key][args.head, args.q].detach().to(torch.float32).numpy()
    logits_b = layer_b["image_logits"][args.image_key][args.head, args.q].detach().to(torch.float32).numpy()
    q_a = layer_a["query_states"][args.head, args.q].detach().to(torch.float32)
    q_b = layer_b["query_states"][args.head, args.q].detach().to(torch.float32)
    k_a = layer_a["image_key_states"][args.image_key][args.head].detach().to(torch.float32)
    k_b = layer_b["image_key_states"][args.image_key][args.head].detach().to(torch.float32)
    scaling = float(q_a.shape[-1] ** -0.5)
    grid_h, grid_w = logits_a.shape

    full_delta = logits_b - logits_a
    delta_q = ((q_b - q_a)[None, :] @ k_a.T).squeeze(0).numpy().reshape(grid_h, grid_w) * scaling
    delta_k = (q_a[None, :] @ (k_b - k_a).T).squeeze(0).numpy().reshape(grid_h, grid_w) * scaling
    cross = ((q_b - q_a)[None, :] @ (k_b - k_a).T).squeeze(0).numpy().reshape(grid_h, grid_w) * scaling
    recomposed = delta_q + delta_k + cross

    attn_a = load_attn(infer_a, args.step)
    attn_b = load_attn(infer_b, args.step)
    if args.topk_source == "attn_a" and attn_a is not None:
        source = attn_a["full_layers"][layer_name][args.image_key][args.head, args.q].detach().to(torch.float32).numpy()
    elif args.topk_source == "attn_b" and attn_b is not None:
        source = attn_b["full_layers"][layer_name][args.image_key][args.head, args.q].detach().to(torch.float32).numpy()
    elif args.topk_source == "abs_delta_q":
        source = np.abs(delta_q)
    else:
        source = np.abs(full_delta)

    source_blocks, block_indices, block_h, block_w = flatten_blocks(source, args.block_size)
    ranked_blocks = np.argsort(-source_blocks.reshape(-1)).astype(int).tolist()

    obs_key = find_image_meta(qk_b, args.image_key).get("obs_key", args.image_key)
    obs = np.load(infer_b / "obs.npz")
    base_image = image_hwc(obs[obs_key])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.out_dir / f"{args.image_key}_reference.png"
    Image.fromarray(base_image).save(image_path)

    payload = {
        "inferA": str(infer_a),
        "inferB": str(infer_b),
        "step": args.step,
        "layer": layer_name,
        "imageKey": args.image_key,
        "head": args.head,
        "q": args.q,
        "gridH": grid_h,
        "gridW": grid_w,
        "blockH": block_h,
        "blockW": block_w,
        "blockSize": args.block_size,
        "topkDefault": args.topk_default,
        "topkSource": args.topk_source,
        "rankedBlocks": ranked_blocks,
        "blockIndices": block_indices,
        "imageUri": image_data_uri(image_path),
        "arrays": {
            "logitsA": round_array(logits_a),
            "logitsB": round_array(logits_b),
            "fullDelta": round_array(full_delta),
            "deltaQ": round_array(delta_q),
            "deltaK": round_array(delta_k),
            "cross": round_array(cross),
            "recomposed": round_array(recomposed),
            "topkSource": round_array(source),
        },
    }
    (args.out_dir / "data.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    html_text = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(payload, ensure_ascii=False))
    (args.out_dir / "index.html").write_text(html_text, encoding="utf-8")
    print(f"Wrote {args.out_dir / 'index.html'}")


HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>QK Logits Top-k Explorer</title>
  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <style>
    body { margin: 0; background: #f6f7f9; color: #20242b; font-family: Arial, sans-serif; }
    header { padding: 16px 20px; background: #fff; border-bottom: 1px solid #d8dde6; }
    h1 { margin: 0 0 6px; font-size: 24px; }
    .subtle { color: #667085; font-size: 14px; line-height: 1.45; }
    .controls { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; padding: 12px 20px; background: #eef2f8; border-bottom: 1px solid #d8dde6; }
    .card { background: #fff; border: 1px solid #d8dde6; border-radius: 8px; padding: 12px; }
    main { display: grid; grid-template-columns: 420px 1fr; gap: 14px; padding: 14px; }
    .image-wrap { position: relative; display: inline-block; line-height: 0; }
    .image-wrap img { width: 384px; height: auto; display: block; }
    .block-grid { position: absolute; inset: 0; display: grid; pointer-events: none; }
    .block-cell { border: 1px solid rgba(255,255,255,0.85); }
    .block-cell.selected { outline: 2px solid #1463ff; outline-offset: -2px; background: rgba(20, 99, 255, 0.12); }
    .heatmaps { display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 12px; }
    .heatmap-title { font-weight: 700; margin-bottom: 8px; }
    .heatmap { display: grid; width: 100%; aspect-ratio: 1 / 1; border: 1px solid #d8dde6; background: #fff; }
    .heat-cell { border: 1px solid rgba(255,255,255,0.55); }
    table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 12px; }
    th, td { border-bottom: 1px solid #e5e8ef; padding: 7px 8px; text-align: right; }
    th:first-child, td:first-child { text-align: left; }
    code { background: #eef2f8; padding: 2px 4px; border-radius: 4px; }
  </style>
</head>
<body>
  <header>
    <h1>QK / Logits Top-k Explorer</h1>
    <div class="subtle" id="meta"></div>
  </header>
  <div class="controls">
    <label>top-k blocks <input id="topk" type="range" min="1" max="64" value="8"></label>
    <strong id="topkLabel"></strong>
    <span class="subtle">Top-k source: <code id="sourceName"></code></span>
  </div>
  <main>
    <section class="card">
      <h2>Selected Blocks</h2>
      <div class="image-wrap">
        <img id="refImage" alt="reference image">
        <div id="blockGrid" class="block-grid"></div>
      </div>
      <p class="subtle">Blue blocks are selected by top-k score. They are the K columns used by the sparse approximation.</p>
      <div id="blockList" class="subtle"></div>
    </section>
    <section class="card">
      <h2>Logits / ΔQK Maps</h2>
      <div class="heatmaps" id="heatmaps"></div>
      <table id="metrics"></table>
    </section>
  </main>
  <script>
    const DATA = __DATA_JSON__;
    const arrays = DATA.arrays;
    const topk = document.getElementById("topk");
    topk.max = DATA.rankedBlocks.length;
    topk.value = Math.min(DATA.topkDefault, DATA.rankedBlocks.length);
    document.getElementById("sourceName").textContent = DATA.topkSource;
    document.getElementById("refImage").src = DATA.imageUri;
    document.getElementById("meta").textContent =
      `A=${DATA.inferA} | B=${DATA.inferB} | step=${DATA.step} | ${DATA.layer} | image=${DATA.imageKey} | head=${DATA.head} | q=${DATA.q}`;

    function flat(arr) { return arr.flat(); }
    function selectedMask(k) {
      const mask = new Array(DATA.gridH * DATA.gridW).fill(false);
      for (const block of DATA.rankedBlocks.slice(0, k)) {
        for (const idx of DATA.blockIndices[block]) mask[idx] = true;
      }
      return mask;
    }
    function selectedArray(arr, mask) {
      const vals = flat(arr);
      return vals.map((v, i) => mask[i] ? v : 0);
    }
    function residual(full, approx) {
      const a = flat(full);
      return a.map((v, i) => v - approx[i]);
    }
    function l2(vals) { return Math.sqrt(vals.reduce((s, v) => s + v * v, 0)); }
    function dot(a, b) { return a.reduce((s, v, i) => s + v * b[i], 0); }
    function cosine(a, b) {
      const denom = l2(a) * l2(b);
      return denom > 0 ? dot(a, b) / denom : 0;
    }
    function rmse(a, b) {
      return Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / Math.max(1, a.length));
    }
    function signedColor(v, maxAbs) {
      const t = Math.min(1, Math.abs(v) / (maxAbs || 1));
      if (v >= 0) return `rgba(220, 38, 38, ${0.08 + 0.82 * t})`;
      return `rgba(20, 99, 255, ${0.08 + 0.82 * t})`;
    }
    function drawHeatmap(title, values, sharedMax) {
      const root = document.createElement("div");
      const label = document.createElement("div");
      label.className = "heatmap-title";
      label.textContent = title;
      const grid = document.createElement("div");
      grid.className = "heatmap";
      grid.style.gridTemplateColumns = `repeat(${DATA.gridW}, 1fr)`;
      const vals = flat(values);
      const maxAbs = sharedMax ?? Math.max(...vals.map(Math.abs), 1e-12);
      vals.forEach((v, idx) => {
        const cell = document.createElement("div");
        cell.className = "heat-cell";
        cell.style.background = signedColor(v, maxAbs);
        cell.title = `token=${idx} row=${Math.floor(idx / DATA.gridW)} col=${idx % DATA.gridW} value=${v}`;
        grid.appendChild(cell);
      });
      root.appendChild(label);
      root.appendChild(grid);
      return root;
    }
    function renderBlocks(k) {
      const blockGrid = document.getElementById("blockGrid");
      blockGrid.innerHTML = "";
      blockGrid.style.gridTemplateColumns = `repeat(${DATA.blockW}, 1fr)`;
      blockGrid.style.gridTemplateRows = `repeat(${DATA.blockH}, 1fr)`;
      const selected = new Set(DATA.rankedBlocks.slice(0, k));
      for (let i = 0; i < DATA.blockH * DATA.blockW; i++) {
        const cell = document.createElement("div");
        cell.className = "block-cell" + (selected.has(i) ? " selected" : "");
        blockGrid.appendChild(cell);
      }
      document.getElementById("blockList").textContent =
        DATA.rankedBlocks.slice(0, k).map((b) => `#${b}(r=${Math.floor(b / DATA.blockW)},c=${b % DATA.blockW})`).join(" ");
    }
    function render() {
      const k = Number(topk.value);
      document.getElementById("topkLabel").textContent = `${k} / ${DATA.rankedBlocks.length}`;
      renderBlocks(k);
      const mask = selectedMask(k);
      const full = flat(arrays.fullDelta);
      const dqAll = flat(arrays.deltaQ);
      const selectedDq = selectedArray(arrays.deltaQ, mask);
      const selectedFull = selectedArray(arrays.fullDelta, mask);
      const resDq = residual(arrays.fullDelta, selectedDq);
      const heatRoot = document.getElementById("heatmaps");
      heatRoot.innerHTML = "";
      const shared = Math.max(...full.map(Math.abs), ...dqAll.map(Math.abs), ...selectedDq.map(Math.abs), ...resDq.map(Math.abs), 1e-12);
      heatRoot.appendChild(drawHeatmap("full Δlogits", arrays.fullDelta, shared));
      heatRoot.appendChild(drawHeatmap("ΔQ @ K_old", arrays.deltaQ, shared));
      heatRoot.appendChild(drawHeatmap("top-k ΔQ @ K_old", reshape(selectedDq), shared));
      heatRoot.appendChild(drawHeatmap("residual: full - top-k", reshape(resDq), shared));
      heatRoot.appendChild(drawHeatmap("ΔK term", arrays.deltaK, shared));
      heatRoot.appendChild(drawHeatmap("cross term", arrays.cross, shared));
      const metrics = [
        ["cos(full, all ΔQ)", cosine(full, dqAll)],
        ["RMSE(full, all ΔQ)", rmse(full, dqAll)],
        ["cos(full, top-k ΔQ)", cosine(full, selectedDq)],
        ["RMSE(full, top-k ΔQ)", rmse(full, selectedDq)],
        ["selected full energy ratio", l2(selectedFull) / Math.max(l2(full), 1e-12)],
        ["selected ΔQ energy ratio", l2(selectedDq) / Math.max(l2(dqAll), 1e-12)],
        ["residual ratio", l2(resDq) / Math.max(l2(full), 1e-12)],
      ];
      document.getElementById("metrics").innerHTML =
        "<tr><th>metric</th><th>value</th></tr>" +
        metrics.map(([name, val]) => `<tr><td>${name}</td><td>${Number(val).toPrecision(6)}</td></tr>`).join("");
    }
    function reshape(vals) {
      const out = [];
      for (let r = 0; r < DATA.gridH; r++) out.push(vals.slice(r * DATA.gridW, (r + 1) * DATA.gridW));
      return out;
    }
    topk.addEventListener("input", render);
    render();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()

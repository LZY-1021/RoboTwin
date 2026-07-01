#!/usr/bin/env python3
"""Render logits and P*V top-k block overlays across denoise steps or layers."""

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


def parse_steps(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(dict.fromkeys(out))


def parse_layers(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return [f"layer_{idx:02d}" for idx in range(18)]
    return [parse_layer(item) for item in value.split(",") if item.strip()]


def parse_heads(value: str, num_heads: int) -> list[int]:
    if value.strip().lower() in {"all", ""}:
        return list(range(num_heads))
    return [head for head in parse_steps(value) if 0 <= head < num_heads]


def load_pt(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def block_scores(values: np.ndarray, block_size: int, *, absolute: bool) -> tuple[np.ndarray, list[list[int]], int, int]:
    grid_h, grid_w = values.shape
    block_h = math.ceil(grid_h / block_size)
    block_w = math.ceil(grid_w / block_size)
    scores = np.zeros((block_h, block_w), dtype=np.float32)
    token_indices: list[list[int]] = []
    source = np.abs(values) if absolute else values
    for br in range(block_h):
        for bc in range(block_w):
            r0 = br * block_size
            r1 = min(grid_h, r0 + block_size)
            c0 = bc * block_size
            c1 = min(grid_w, c0 + block_size)
            scores[br, bc] = float(np.mean(source[r0:r1, c0:c1]))
            token_indices.append([r * grid_w + c for r in range(r0, r1) for c in range(c0, c1)])
    return scores, token_indices, block_h, block_w


def rank_blocks(values: np.ndarray, block_size: int, *, absolute: bool) -> tuple[list[int], int, int]:
    scores, _, block_h, block_w = block_scores(values, block_size, absolute=absolute)
    ranked = np.argsort(-scores.reshape(-1)).astype(int).tolist()
    return ranked, block_h, block_w


def round_nested(values: np.ndarray, digits: int = 7) -> Any:
    return np.round(values.astype(np.float32), digits).tolist()


def load_frame(
    infer_dir: Path,
    *,
    step: int,
    layer: str,
    image_key: str,
    heads: list[int],
    q_indices: list[int],
    block_size: int,
) -> dict[str, Any]:
    qk = load_pt(infer_dir / f"qk_denoise_step_{step:03d}.pt")
    attn = load_pt(infer_dir / f"attn_denoise_step_{step:03d}.pt")
    if layer not in qk["layers"]:
        raise KeyError(f"{layer} missing in {infer_dir}/qk_denoise_step_{step:03d}.pt")
    qk_layer = qk["layers"][layer]
    attn_layer = attn["full_layers"][layer]
    values = qk_layer.get("image_value_states", {}).get(image_key)
    if values is None:
        raise KeyError("image_value_states missing. Re-run trace after the P*V trace update.")
    heads_payload = {}
    block_h = block_w = None
    for head in heads:
        logits = qk_layer["image_logits"][image_key][head, q_indices].detach().to(torch.float32).mean(dim=0).numpy()
        probs = attn_layer[image_key][head, q_indices].detach().to(torch.float32).mean(dim=0).numpy()
        value_norm = values[head].detach().to(torch.float32).norm(dim=-1).numpy().reshape(logits.shape)
        pv = probs * value_norm
        logits_ranked, cur_block_h, cur_block_w = rank_blocks(logits, block_size, absolute=True)
        pv_ranked, _, _ = rank_blocks(pv, block_size, absolute=False)
        block_h = cur_block_h
        block_w = cur_block_w
        heads_payload[str(head)] = {
            "logits": round_nested(logits),
            "pv": round_nested(pv),
            "attn": round_nested(probs),
            "valueNorm": round_nested(value_norm),
            "logitsRanked": logits_ranked,
            "pvRanked": pv_ranked,
            "logitsMaxAbs": float(np.max(np.abs(logits))),
            "attnMax": float(np.max(probs)),
            "pvMax": float(np.max(pv)),
            "valueNormMax": float(np.max(value_norm)),
        }
    return {
        "step": step,
        "layer": layer,
        "label": f"step_{step:02d} / {layer}",
        "heads": heads_payload,
        "blockH": int(block_h or 0),
        "blockW": int(block_w or 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Trace root, episode dir, or infer dir.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--infer", type=int, default=0)
    parser.add_argument("--axis", choices=["step", "layer"], default="step")
    parser.add_argument("--steps", default="0-9")
    parser.add_argument("--layers", default="12")
    parser.add_argument("--fixed-step", type=int, default=9)
    parser.add_argument("--fixed-layer", default="12")
    parser.add_argument("--dynamic", action="store_true", help="Load all requested steps and layers for in-page switching.")
    parser.add_argument("--image-key", default="base_0_rgb")
    parser.add_argument("--head", type=int, default=0, help="Initial head selected in the page.")
    parser.add_argument("--heads", default="all", help="Heads to include in the page, e.g. all or 0,3,7.")
    parser.add_argument("--q", default="0", help="Single q or range like 0-31.")
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    infer_dir = args.path if args.path.name.startswith("infer_") else infer_dir_from_trace(args.path, args.episode, args.infer)
    q_indices = parse_steps(args.q)
    if args.dynamic:
        steps = parse_steps(args.steps)
        layers = parse_layers(args.layers)
    elif args.axis == "step":
        steps = parse_steps(args.steps)
        layers = [parse_layer(args.fixed_layer)]
    else:
        steps = [args.fixed_step]
        layers = parse_layers(args.layers)

    first_qk = load_pt(infer_dir / f"qk_denoise_step_{steps[0]:03d}.pt")
    first_layer_name = layers[0]
    if first_layer_name not in first_qk["layers"]:
        raise KeyError(f"{first_layer_name} missing in first qk file.")
    num_heads = int(first_qk["layers"][first_layer_name]["image_logits"][args.image_key].shape[0])
    heads = parse_heads(args.heads, num_heads)
    if args.head not in heads:
        heads = sorted(dict.fromkeys([args.head, *heads]))
    image_meta = find_image_meta(first_qk, args.image_key)
    obs_key = image_meta.get("obs_key", args.image_key)
    obs = np.load(infer_dir / "obs.npz")
    base_image = image_hwc(obs[obs_key])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.out_dir / f"{args.image_key}_reference.png"
    Image.fromarray(base_image).save(image_path)

    frames = []
    for step in steps:
        for layer in layers:
            frames.append(
                load_frame(
                    infer_dir,
                    step=step,
                    layer=layer,
                    image_key=args.image_key,
                    heads=heads,
                    q_indices=q_indices,
                    block_size=args.block_size,
                )
            )
    if not frames:
        raise ValueError("No frames loaded.")

    payload = {
        "inferDir": str(infer_dir),
        "axis": args.axis,
        "steps": steps,
        "layers": layers,
        "imageKey": args.image_key,
        "imageUri": image_data_uri(image_path),
        "head": args.head,
        "heads": heads,
        "qIndices": q_indices,
        "blockSize": args.block_size,
        "topk": args.topk,
        "gridH": len(frames[0]["heads"][str(heads[0])]["logits"]),
        "gridW": len(frames[0]["heads"][str(heads[0])]["logits"][0]),
        "blockH": frames[0]["blockH"],
        "blockW": frames[0]["blockW"],
        "frames": frames,
    }
    (args.out_dir / "data.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (args.out_dir / "index.html").write_text(
        HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(payload, ensure_ascii=False)),
        encoding="utf-8",
    )
    print(f"Wrote {args.out_dir / 'index.html'}")


HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>QK Logits / P*V Evolution</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; color: #20242b; background: #f6f7f9; }
    header { padding: 16px 20px; background: #fff; border-bottom: 1px solid #d8dde6; }
    h1 { margin: 0 0 6px; font-size: 24px; }
    h2 { margin: 18px 0 10px; font-size: 18px; }
    .subtle { color: #667085; font-size: 14px; line-height: 1.4; }
    .controls { display: flex; flex-wrap: wrap; gap: 14px; align-items: end; padding: 12px 20px; background: #eef2f8; border-bottom: 1px solid #d8dde6; }
    label { display: grid; gap: 4px; font-size: 12px; color: #667085; }
    select, input { min-width: 130px; border: 1px solid #cfd6e2; border-radius: 6px; padding: 7px 8px; background: #fff; color: #20242b; }
    input[type="range"] { min-width: 220px; padding: 0; }
    main { padding: 14px 20px 24px; }
    .strip { display: flex; gap: 12px; overflow-x: auto; padding: 8px 0 18px; }
    .card { flex: 0 0 310px; background: #fff; border: 1px solid #d8dde6; border-radius: 8px; padding: 10px; }
    .label { font-weight: 700; margin-bottom: 8px; }
    .overlay { position: relative; width: 288px; line-height: 0; }
    .overlay img { width: 288px; display: block; }
    .heat { position: absolute; inset: 0; display: grid; }
    .cell { border: 1px solid rgba(255,255,255,0.65); }
    .blocks { position: absolute; inset: 0; display: grid; pointer-events: none; }
    .block { border: 1px solid rgba(255,255,255,0.5); }
    .block.top { outline: 2px solid #1463ff; outline-offset: -2px; background: rgba(20, 99, 255, 0.08); }
    .stats { margin-top: 8px; font-size: 12px; color: #667085; line-height: 1.35; }
  </style>
</head>
<body>
  <header>
    <h1>QK Logits / P*V Evolution</h1>
    <div class="subtle" id="meta"></div>
  </header>
  <div class="controls">
    <label>View mode
      <select id="modeSelect">
        <option value="step">x-axis: denoise step</option>
        <option value="layer">x-axis: layer</option>
      </select>
    </label>
    <label>Fixed step
      <select id="stepSelect"></select>
    </label>
    <label>Fixed layer
      <select id="layerSelect"></select>
    </label>
    <label>Head
      <select id="headSelect"></select>
    </label>
    <label>Color scale
      <select id="colorScaleMode">
        <option value="global">all loaded frames</option>
        <option value="view">current view</option>
        <option value="frame">per frame</option>
      </select>
    </label>
    <label>Logits block ranking
      <select id="logitsRankMode">
        <option value="abs_logits">|logits| magnitude</option>
        <option value="positive_logits">positive logits only</option>
        <option value="negative_logits">negative logits magnitude</option>
        <option value="attn">softmax P</option>
        <option value="pv">P * ||V||</option>
        <option value="value_norm">||V|| only</option>
      </select>
    </label>
    <label>Logits heatmap value
      <select id="logitsHeatMode">
        <option value="follow">follow ranking mode</option>
        <option value="signed_logits">signed logits</option>
        <option value="abs_logits">|logits| magnitude</option>
        <option value="positive_logits">positive logits only</option>
        <option value="negative_logits">negative logits magnitude</option>
        <option value="attn">softmax P</option>
        <option value="pv">P * ||V||</option>
        <option value="value_norm">||V|| only</option>
      </select>
    </label>
    <label>P*V block ranking
      <select id="pvRankMode">
        <option value="pv">P * ||V||</option>
        <option value="attn">softmax P</option>
        <option value="value_norm">||V|| only</option>
        <option value="abs_logits">|logits| magnitude</option>
        <option value="positive_logits">positive logits only</option>
      </select>
    </label>
    <label>P*V heatmap value
      <select id="pvHeatMode">
        <option value="follow">follow ranking mode</option>
        <option value="pv">P * ||V||</option>
        <option value="attn">softmax P</option>
        <option value="value_norm">||V|| only</option>
        <option value="signed_logits">signed logits</option>
        <option value="abs_logits">|logits| magnitude</option>
        <option value="positive_logits">positive logits only</option>
        <option value="negative_logits">negative logits magnitude</option>
      </select>
    </label>
    <label>Top-k blocks
      <input id="topkInput" type="range" min="1" value="8">
    </label>
    <strong id="topkLabel"></strong>
  </div>
  <main>
    <h2>Logits top-k blocks</h2>
    <div id="logitsStrip" class="strip"></div>
    <h2>P*V top-k blocks</h2>
    <div id="pvStrip" class="strip"></div>
  </main>
  <script>
    const DATA = __DATA_JSON__;
    const framesByKey = new Map(DATA.frames.map(f => [`${f.step}|${f.layer}`, f]));
    const modeSelect = document.getElementById("modeSelect");
    const stepSelect = document.getElementById("stepSelect");
    const layerSelect = document.getElementById("layerSelect");
    const headSelect = document.getElementById("headSelect");
    const colorScaleMode = document.getElementById("colorScaleMode");
    const logitsRankMode = document.getElementById("logitsRankMode");
    const logitsHeatMode = document.getElementById("logitsHeatMode");
    const pvRankMode = document.getElementById("pvRankMode");
    const pvHeatMode = document.getElementById("pvHeatMode");
    const topkInput = document.getElementById("topkInput");
    topkInput.max = DATA.blockH * DATA.blockW;
    topkInput.value = Math.min(DATA.topk, DATA.blockH * DATA.blockW);
    for (const step of DATA.steps) {
      const opt = document.createElement("option");
      opt.value = step;
      opt.textContent = `step ${String(step).padStart(2, "0")}`;
      stepSelect.appendChild(opt);
    }
    for (const layer of DATA.layers) {
      const opt = document.createElement("option");
      opt.value = layer;
      opt.textContent = layer;
      layerSelect.appendChild(opt);
    }
    for (const head of DATA.heads) {
      const opt = document.createElement("option");
      opt.value = head;
      opt.textContent = `head ${head}`;
      headSelect.appendChild(opt);
    }
    stepSelect.value = DATA.steps.includes(9) ? 9 : DATA.steps[0];
    layerSelect.value = DATA.layers.includes("layer_12") ? "layer_12" : DATA.layers[0];
    headSelect.value = String(DATA.heads.includes(DATA.head) ? DATA.head : DATA.heads[0]);
    modeSelect.value = DATA.axis;
    function updateMeta() {
      document.getElementById("meta").textContent =
        `${DATA.inferDir} | image=${DATA.imageKey} | head=${headSelect.value} | q=[${DATA.qIndices.join(",")}]`;
    }
    function headData(frame) {
      return frame.heads[String(headSelect.value)];
    }
    function scaleFrames(frame) {
      if (colorScaleMode.value === "global") return DATA.frames;
      if (colorScaleMode.value === "frame") return [frame];
      return currentFrames();
    }
    function logitsScale(frame) {
      return Math.max(...scaleFrames(frame).map(f => headData(f).logitsMaxAbs), 1e-12);
    }
    function attnScale(frame) {
      return Math.max(...scaleFrames(frame).map(f => headData(f).attnMax || 0), 1e-12);
    }
    function pvScale(frame) {
      return Math.max(...scaleFrames(frame).map(f => headData(f).pvMax), 1e-12);
    }
    function valueNormScale(frame) {
      return Math.max(...scaleFrames(frame).map(f => headData(f).valueNormMax || 0), 1e-12);
    }
    function heatScale(frame, mode) {
      if (mode === "signed_logits" || mode === "abs_logits" || mode === "positive_logits" || mode === "negative_logits") return logitsScale(frame);
      if (mode === "attn") return attnScale(frame);
      if (mode === "pv") return pvScale(frame);
      if (mode === "value_norm") return valueNormScale(frame);
      return logitsScale(frame);
    }
    function flat(arr) { return arr.flat(); }
    function signed(v, scale) {
      const t = Math.min(1, Math.abs(v) / scale);
      if (v >= 0) return `rgba(220, 38, 38, ${0.05 + 0.72 * t})`;
      return `rgba(20, 99, 255, ${0.05 + 0.72 * t})`;
    }
    function positive(v, scale) {
      const t = Math.min(1, Math.max(0, v) / scale);
      return `rgba(34, 197, 94, ${0.04 + 0.75 * t})`;
    }
    function rankValue(mode, hd, r, c) {
      const logit = hd.logits[r][c];
      if (mode === "signed_logits") return logit;
      if (mode === "abs_logits") return Math.abs(logit);
      if (mode === "positive_logits") return Math.max(0, logit);
      if (mode === "negative_logits") return Math.max(0, -logit);
      if (mode === "attn") return hd.attn[r][c];
      if (mode === "pv") return hd.pv[r][c];
      if (mode === "value_norm") return hd.valueNorm[r][c];
      return Math.abs(logit);
    }
    function modeLabel(mode) {
      if (mode === "signed_logits") return "signed logits";
      if (mode === "abs_logits") return "|logits|";
      if (mode === "positive_logits") return "positive logits";
      if (mode === "negative_logits") return "negative logits";
      if (mode === "attn") return "softmax P";
      if (mode === "pv") return "P*||V||";
      if (mode === "value_norm") return "||V||";
      return mode;
    }
    function metricName(mode) {
      if (mode === "signed_logits") return "logit";
      if (mode === "abs_logits") return "|logits|";
      if (mode === "positive_logits") return "pos_logit";
      if (mode === "negative_logits") return "neg_logit";
      if (mode === "attn") return "P";
      if (mode === "pv") return "P*V";
      if (mode === "value_norm") return "Vnorm";
      return "score";
    }
    function tokenMetrics(hd, r, c) {
      return {
        logit: hd.logits[r][c],
        P: hd.attn[r][c],
        PV: hd.pv[r][c],
        Vnorm: hd.valueNorm[r][c],
      };
    }
    function heatValue(mode, hd, r, c) {
      return rankValue(mode, hd, r, c);
    }
    function heatColor(mode, value, scale, metric) {
      if (mode === "signed_logits" || mode === "abs_logits") return signed(metric.logit, scale);
      if (mode === "positive_logits") {
        const t = Math.min(1, Math.max(0, metric.logit) / scale);
        return `rgba(220, 38, 38, ${0.04 + 0.75 * t})`;
      }
      if (mode === "negative_logits") {
        const t = Math.min(1, Math.max(0, -metric.logit) / scale);
        return `rgba(20, 99, 255, ${0.04 + 0.75 * t})`;
      }
      return positive(value, scale);
    }
    function rankedBlocks(hd, mode) {
      const blockScores = [];
      const strictPositive = mode === "positive_logits" || mode === "negative_logits";
      for (let br = 0; br < DATA.blockH; br++) {
        for (let bc = 0; bc < DATA.blockW; bc++) {
          let sum = 0;
          let count = 0;
          const r0 = br * DATA.blockSize;
          const r1 = Math.min(DATA.gridH, r0 + DATA.blockSize);
          const c0 = bc * DATA.blockSize;
          const c1 = Math.min(DATA.gridW, c0 + DATA.blockSize);
          for (let r = r0; r < r1; r++) {
            for (let c = c0; c < c1; c++) {
              sum += rankValue(mode, hd, r, c);
              count += 1;
            }
          }
          const score = count ? sum / count : 0;
          if (!strictPositive || score > 0) {
            blockScores.push({ idx: br * DATA.blockW + bc, score });
          }
        }
      }
      blockScores.sort((a, b) => b.score - a.score);
      return blockScores;
    }
    function drawFrame(frame, kind) {
      const hd = headData(frame);
      const topk = Number(topkInput.value);
      const rankMode = kind === "logits" ? logitsRankMode.value : pvRankMode.value;
      const rawHeatMode = kind === "logits" ? logitsHeatMode.value : pvHeatMode.value;
      const heatMode = rawHeatMode === "follow" ? rankMode : rawHeatMode;
      const ranked = rankedBlocks(hd, rankMode);
      const tops = new Set(ranked.slice(0, topk).map(item => item.idx));
      const scale = heatScale(frame, heatMode);
      const card = document.createElement("div");
      card.className = "card";
      const label = document.createElement("div");
      label.className = "label";
      label.textContent = `${frame.label} · H${headSelect.value}`;
      const overlay = document.createElement("div");
      overlay.className = "overlay";
      const img = document.createElement("img");
      img.src = DATA.imageUri;
      const heat = document.createElement("div");
      heat.className = "heat";
      heat.style.gridTemplateColumns = `repeat(${DATA.gridW}, 1fr)`;
      heat.style.gridTemplateRows = `repeat(${DATA.gridH}, 1fr)`;
      for (let idx = 0; idx < DATA.gridH * DATA.gridW; idx++) {
        const cell = document.createElement("div");
        const r = Math.floor(idx / DATA.gridW);
        const c = idx % DATA.gridW;
        const metric = tokenMetrics(hd, r, c);
        const rankScore = rankValue(rankMode, hd, r, c);
        const heatScore = heatValue(heatMode, hd, r, c);
        cell.className = "cell";
        cell.style.background = heatColor(heatMode, heatScore, scale, metric);
        cell.title = `token=${idx} r=${r} c=${c}\nheat ${metricName(heatMode)}=${heatScore}\nrank ${metricName(rankMode)}=${rankScore}\nlogit=${metric.logit}\nP=${metric.P}\nP*V=${metric.PV}\nVnorm=${metric.Vnorm}`;
        heat.appendChild(cell);
      }
      const blocks = document.createElement("div");
      blocks.className = "blocks";
      blocks.style.gridTemplateColumns = `repeat(${DATA.blockW}, 1fr)`;
      blocks.style.gridTemplateRows = `repeat(${DATA.blockH}, 1fr)`;
      for (let i = 0; i < DATA.blockH * DATA.blockW; i++) {
        const block = document.createElement("div");
        block.className = "block" + (tops.has(i) ? " top" : "");
        blocks.appendChild(block);
      }
      overlay.appendChild(img);
      overlay.appendChild(heat);
      overlay.appendChild(blocks);
      const stats = document.createElement("div");
      stats.className = "stats";
      stats.textContent = `rank by ${modeLabel(rankMode)}: ${ranked.slice(0, topk).map(item => {
        const r = Math.floor(item.idx / DATA.blockW);
        const c = item.idx % DATA.blockW;
        const metric = tokenMetrics(hd, r, c);
        return `#${item.idx}(r=${r},c=${c},${metricName(rankMode)}=${item.score.toExponential(3)},logit=${metric.logit.toExponential(3)},P=${metric.P.toExponential(3)},P*V=${metric.PV.toExponential(3)})`;
      }).join(" ")}`;
      card.appendChild(label);
      card.appendChild(overlay);
      card.appendChild(stats);
      return card;
    }
    function currentFrames() {
      const mode = modeSelect.value;
      if (mode === "step") {
        const layer = layerSelect.value;
        return DATA.steps.map(step => framesByKey.get(`${step}|${layer}`)).filter(Boolean);
      }
      const step = Number(stepSelect.value);
      return DATA.layers.map(layer => framesByKey.get(`${step}|${layer}`)).filter(Boolean);
    }
    function render() {
      const k = Number(topkInput.value);
      document.getElementById("topkLabel").textContent = `${k} / ${DATA.blockH * DATA.blockW}`;
      updateMeta();
      stepSelect.disabled = modeSelect.value === "step";
      layerSelect.disabled = modeSelect.value === "layer";
      const logitsStrip = document.getElementById("logitsStrip");
      const pvStrip = document.getElementById("pvStrip");
      logitsStrip.innerHTML = "";
      pvStrip.innerHTML = "";
      for (const frame of currentFrames()) {
        logitsStrip.appendChild(drawFrame(frame, "logits"));
        pvStrip.appendChild(drawFrame(frame, "pv"));
      }
    }
    modeSelect.addEventListener("change", render);
    stepSelect.addEventListener("change", render);
    layerSelect.addEventListener("change", render);
    headSelect.addEventListener("change", render);
    colorScaleMode.addEventListener("change", render);
    logitsRankMode.addEventListener("change", render);
    logitsHeatMode.addEventListener("change", render);
    pvRankMode.addEventListener("change", render);
    pvHeatMode.addEventListener("change", render);
    topkInput.addEventListener("input", render);
    render();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()

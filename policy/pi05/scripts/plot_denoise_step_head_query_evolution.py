#!/usr/bin/env python3
"""Render how denoise head/query attention trajectories evolve over steps."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from plot_denoise_head_query_attention import (
    attention_centroids,
    draw_trajectory,
    find_image_meta,
    infer_dir_from_trace,
    parse_heads,
    parse_layer,
    save_contact_sheet,
)
from plot_patch_kv_overlay import image_hwc


def parse_steps(value: str) -> list[int]:
    steps: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            steps.extend(range(int(start), int(end) + 1))
        else:
            steps.append(int(part))
    return sorted(dict.fromkeys(steps))


def load_full_attention(
    infer_dir: Path,
    step: int,
    layer_name: str,
    image_key: str,
) -> tuple[dict[str, Any], np.ndarray]:
    path = infer_dir / f"attn_denoise_step_{step:03d}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    full_layers = payload.get("full_layers")
    if not full_layers:
        raise ValueError(f"{path} does not contain full_layers. Re-run trace with PI05_TRACE_SAVE_ATTN_FULL=1.")
    if layer_name not in full_layers:
        raise KeyError(f"{layer_name} not in {path}. Available: {sorted(full_layers)}")
    if image_key not in full_layers[layer_name]:
        raise KeyError(f"{image_key} not in {path}:{layer_name}. Available: {sorted(full_layers[layer_name])}")
    return payload, full_layers[layer_name][image_key].detach().to(torch.float32).numpy()


def trajectory_stats(centroids: np.ndarray) -> dict[str, float]:
    if centroids.shape[0] < 2:
        return {"path_len": 0.0, "straight_len": 0.0, "straight_over_path": 0.0}
    diffs = np.diff(centroids, axis=0)
    path_len = float(np.nansum(np.linalg.norm(diffs, axis=1)))
    disp = centroids[-1] - centroids[0]
    straight_len = float(np.linalg.norm(disp))
    return {
        "path_len": path_len,
        "straight_len": straight_len,
        "straight_over_path": straight_len / path_len if path_len > 0 else 0.0,
        "dx": float(disp[0]),
        "dy": float(disp[1]),
    }


def top_mass_and_entropy(action_heatmaps: np.ndarray) -> tuple[float, float]:
    masses = []
    entropies = []
    for heat in action_heatmaps:
        values = np.clip(np.asarray(heat, dtype=np.float64), 0.0, None)
        denom = float(values.sum())
        if denom <= 0:
            continue
        prob = values / denom
        masses.append(float(values.max() / denom))
        entropies.append(float(-(prob * np.log(prob + 1e-12)).sum() / np.log(values.size)))
    return (
        float(np.mean(masses)) if masses else float("nan"),
        float(np.mean(entropies)) if entropies else float("nan"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Trace root, episode dir, or infer dir.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--infer", type=int, default=0)
    parser.add_argument("--steps", default="0-9", help="Comma/range list, e.g. 0-9 or 0,3,9.")
    parser.add_argument("--layer", default="12")
    parser.add_argument("--image-key", default="base_0_rgb")
    parser.add_argument("--heads", default="all")
    parser.add_argument("--state-query-count", type=int, default=None)
    parser.add_argument("--max-action-queries", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--draw-grid", action="store_true")
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--thumb-width", type=int, default=220)
    args = parser.parse_args()

    infer_dir = args.path if args.path.name.startswith("infer_") else infer_dir_from_trace(args.path, args.episode, args.infer)
    layer_name = parse_layer(args.layer)
    steps = parse_steps(args.steps)
    if not steps:
        raise ValueError("--steps resolved to empty list")

    first_payload, first_full = load_full_attention(infer_dir, steps[0], layer_name, args.image_key)
    num_heads, suffix_query_len, grid_h, grid_w = first_full.shape
    query_axis = first_payload.get("query_axis", {})
    state_count = args.state_query_count
    if state_count is None:
        state_count = int(query_axis.get("state_query_count", max(0, suffix_query_len - args.max_action_queries)))
    state_count = max(0, min(state_count, suffix_query_len))
    action_start = state_count
    action_end = min(suffix_query_len, action_start + args.max_action_queries)
    heads = parse_heads(args.heads, num_heads)

    image_meta = find_image_meta(first_payload, args.image_key)
    obs_key = image_meta.get("obs_key", args.image_key)
    obs = np.load(infer_dir / "obs.npz")
    base_image = image_hwc(obs[obs_key])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_step: dict[int, dict[int, np.ndarray]] = {}
    per_head_step_images: dict[int, list[Path]] = {head: [] for head in heads}
    per_head_step_labels: dict[int, list[str]] = {head: [] for head in heads}
    rows = []
    stats_rows = []

    for step in steps:
        _, full = load_full_attention(infer_dir, step, layer_name, args.image_key)
        if full.shape[:2] != (num_heads, suffix_query_len):
            raise ValueError(f"Step {step} shape changed from {first_full.shape} to {full.shape}")
        step_centroids: dict[int, np.ndarray] = {}
        step_dir = args.out_dir / f"step_{step:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        for head in heads:
            action_heatmaps = full[head, action_start:action_end]
            centroids = attention_centroids(action_heatmaps)
            step_centroids[head] = centroids
            head_path = step_dir / f"step{step:03d}_h{head:02d}_trajectory.png"
            draw_trajectory(
                base_image,
                {head: centroids},
                head_path,
                grid_h=grid_h,
                grid_w=grid_w,
                per_query_color=True,
                draw_grid=args.draw_grid,
            )
            per_head_step_images[head].append(head_path)
            per_head_step_labels[head].append(f"step {step}")
            top_mass, entropy = top_mass_and_entropy(action_heatmaps)
            stats_rows.append(
                {
                    "step": step,
                    "head": head,
                    **trajectory_stats(centroids),
                    "top_mass_mean": top_mass,
                    "entropy_mean": entropy,
                    "centroid_start": centroids[0].tolist() if len(centroids) else None,
                    "centroid_end": centroids[-1].tolist() if len(centroids) else None,
                }
            )

        all_path = step_dir / f"step{step:03d}_all_heads_trajectory.png"
        draw_trajectory(
            base_image,
            step_centroids,
            all_path,
            grid_h=grid_h,
            grid_w=grid_w,
            per_query_color=False,
            draw_grid=args.draw_grid,
        )
        per_step[step] = step_centroids
        rows.append({"step": step, "all_heads": all_path.relative_to(args.out_dir).as_posix()})

    head_rows = []
    for head in heads:
        sheet_path = args.out_dir / f"head_{head:02d}_steps_sheet.png"
        save_contact_sheet(
            per_head_step_images[head],
            per_head_step_labels[head],
            sheet_path,
            columns=args.columns,
            thumb_width=args.thumb_width,
        )
        head_rows.append({"head": head, "steps_sheet": sheet_path.relative_to(args.out_dir).as_posix()})

    summary = {
        "infer_dir": str(infer_dir),
        "steps": steps,
        "layer": layer_name,
        "image_key": args.image_key,
        "obs_key": obs_key,
        "shape": list(first_full.shape),
        "state_query_count": state_count,
        "action_query_range": [action_start, action_end],
        "heads": heads,
        "per_step": rows,
        "per_head": head_rows,
        "stats": stats_rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    step_html = []
    for row in rows:
        img = html.escape(row["all_heads"])
        step_html.append(f"<tr><td>{row['step']}</td><td><a href='{img}'><img src='{img}'></a></td></tr>")

    head_html = []
    for row in head_rows:
        img = html.escape(row["steps_sheet"])
        head_html.append(f"<tr><td>H{row['head']}</td><td><a href='{img}'><img src='{img}'></a></td></tr>")

    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Denoise Step/Head/Query Evolution</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; margin-bottom: 28px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    img {{ max-width: 1200px; border: 1px solid #ddd; }}
    td:first-child {{ font-weight: 700; font-size: 18px; }}
  </style>
</head>
<body>
  <h1>Denoise Step/Head/Query Evolution</h1>
  <p><code>{html.escape(str(infer_dir))}</code></p>
  <p>steps=<code>{html.escape(str(steps))}</code>, layer=<code>{html.escape(layer_name)}</code>, image=<code>{html.escape(args.image_key)}</code></p>
  <p>shape=<code>[heads={num_heads}, queries={suffix_query_len}, grid={grid_h}x{grid_w}]</code>;
  action query range=<code>[{action_start}, {action_end})</code>.</p>
  <h2>Per-step all-head trajectory</h2>
  <table><tr><th>step</th><th>all heads</th></tr>{''.join(step_html)}</table>
  <h2>Per-head trajectory across denoise steps</h2>
  <table><tr><th>head</th><th>step 0 → 9 trajectory sheets</th></tr>{''.join(head_html)}</table>
</body>
</html>
"""
    (args.out_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"Wrote {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()

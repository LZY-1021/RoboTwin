#!/usr/bin/env python3
"""Render denoise attention by head and action-query time.

This script expects traces saved with:
  PI05_TRACE_SAVE_ATTN=1 PI05_TRACE_SAVE_ATTN_FULL=1

It reads attn_denoise_step_XXX.pt and creates one section per attention head:
state-query heatmap, if present, plus action-query heatmaps ordered by action
chunk time.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from plot_patch_kv_overlay import image_hwc, overlay_image, safe_name


def infer_dir_from_trace(path: Path, episode: int, infer: int) -> Path:
    root = path.parent if path.name.startswith("episode_") else path
    return root / f"episode_{episode:04d}" / f"infer_{infer:04d}"


def parse_heads(value: str | None, num_heads: int) -> list[int]:
    if value is None or value.strip().lower() in {"", "all"}:
        return list(range(num_heads))
    heads = [int(item) for item in value.split(",") if item.strip()]
    return [head for head in heads if 0 <= head < num_heads]


def parse_layer(value: str | int) -> str:
    if isinstance(value, int):
        return f"layer_{value:02d}"
    token = value.strip()
    if token.startswith("layer_"):
        return token
    return f"layer_{int(token):02d}"


def find_image_meta(payload: dict[str, Any], image_key: str) -> dict[str, Any]:
    for item in payload.get("image_tokens", []):
        if item.get("image_key") == image_key:
            return item
    keys = [item.get("image_key") for item in payload.get("image_tokens", [])]
    raise KeyError(f"Image key {image_key!r} not found. Available: {keys}")


def save_contact_sheet(
    image_paths: list[Path],
    labels: list[str],
    out_path: Path,
    *,
    columns: int,
    thumb_width: int,
) -> None:
    thumbs: list[Image.Image] = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        scale = thumb_width / img.width
        thumb_height = max(1, round(img.height * scale))
        thumbs.append(img.resize((thumb_width, thumb_height), Image.Resampling.BILINEAR))
    if not thumbs:
        return

    label_h = 20
    gap = 8
    rows = (len(thumbs) + columns - 1) // columns
    cell_w = thumb_width
    cell_h = max(img.height for img in thumbs) + label_h
    canvas = Image.new(
        "RGB",
        (columns * cell_w + (columns - 1) * gap, rows * cell_h + (rows - 1) * gap),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for idx, (img, label) in enumerate(zip(thumbs, labels)):
        row = idx // columns
        col = idx % columns
        x = col * (cell_w + gap)
        y = row * (cell_h + gap)
        draw.text((x + 4, y + 3), label, fill=(20, 20, 20))
        canvas.paste(img, (x, y + label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def attention_centroids(heatmaps: np.ndarray) -> np.ndarray:
    """Return [num_queries, 2] centroids in patch-grid coordinates as x, y."""
    if heatmaps.ndim != 3:
        raise ValueError(f"Expected [queries, H, W], got {heatmaps.shape}")
    _, grid_h, grid_w = heatmaps.shape
    ys, xs = np.mgrid[0:grid_h, 0:grid_w]
    centers = []
    for heat in heatmaps:
        values = np.asarray(heat, dtype=np.float64)
        values = np.clip(values, 0.0, None)
        denom = float(values.sum())
        if denom <= 0:
            centers.append((float("nan"), float("nan")))
            continue
        centers.append((float((values * xs).sum() / denom), float((values * ys).sum() / denom)))
    return np.asarray(centers, dtype=np.float32)


def gradient_color(index: int, total: int) -> tuple[int, int, int]:
    if total <= 1:
        return (40, 120, 255)
    t = index / (total - 1)
    # Blue -> cyan -> yellow -> red, readable on white robot scenes.
    if t < 0.33:
        u = t / 0.33
        return (round(40 * (1 - u)), round(120 + 120 * u), 255)
    if t < 0.66:
        u = (t - 0.33) / 0.33
        return (round(255 * u), 240, round(255 * (1 - u)))
    u = (t - 0.66) / 0.34
    return (255, round(240 * (1 - u) + 40 * u), 0)


def head_color(head: int) -> tuple[int, int, int]:
    palette = [
        (230, 25, 75),
        (60, 180, 75),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 180, 20),
    ]
    return palette[head % len(palette)]


def draw_trajectory(
    base_image: np.ndarray,
    centers_by_head: dict[int, np.ndarray],
    out_path: Path,
    *,
    grid_h: int,
    grid_w: int,
    per_query_color: bool,
    draw_grid: bool,
) -> None:
    img = Image.fromarray(image_hwc(base_image)).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    if draw_grid:
        for row in range(1, grid_h):
            y = round(row * height / grid_h)
            draw.line([(0, y), (width, y)], fill=(255, 255, 255), width=1)
        for col in range(1, grid_w):
            x = round(col * width / grid_w)
            draw.line([(x, 0), (x, height)], fill=(255, 255, 255), width=1)

    for head, centers in centers_by_head.items():
        points = []
        for cx, cy in centers:
            if not np.isfinite(cx) or not np.isfinite(cy):
                points.append(None)
                continue
            x = int(round((float(cx) + 0.5) * width / grid_w))
            y = int(round((float(cy) + 0.5) * height / grid_h))
            points.append((x, y))

        valid = [point for point in points if point is not None]
        if len(valid) < 1:
            continue
        for idx in range(len(points) - 1):
            p0 = points[idx]
            p1 = points[idx + 1]
            if p0 is None or p1 is None:
                continue
            color = gradient_color(idx, len(points) - 1) if per_query_color else head_color(head)
            draw.line([p0, p1], fill=color, width=3)
        for idx, point in enumerate(points):
            if point is None:
                continue
            color = gradient_color(idx, len(points)) if per_query_color else head_color(head)
            radius = 4 if idx not in {0, len(points) - 1} else 6
            draw.ellipse(
                [(point[0] - radius, point[1] - radius), (point[0] + radius, point[1] + radius)],
                fill=color,
                outline=(0, 0, 0),
                width=1,
            )
            if idx in {0, len(points) - 1}:
                label = f"H{head} q{idx}"
                draw.text((point[0] + radius + 2, point[1] - radius - 2), label, fill=(0, 0, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Trace root, episode dir, or infer dir.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--infer", type=int, default=0)
    parser.add_argument("--step", type=int, default=9)
    parser.add_argument("--layer", default="12", help="Layer index or layer_XX.")
    parser.add_argument("--image-key", default="base_0_rgb")
    parser.add_argument("--heads", default="all", help="Comma-separated heads or all.")
    parser.add_argument("--max-action-queries", type=int, default=50)
    parser.add_argument("--state-query-count", type=int, default=None, help="Override state query count. Auto uses trace metadata.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--draw-grid", action="store_true")
    parser.add_argument("--columns", type=int, default=10)
    parser.add_argument("--thumb-width", type=int, default=180)
    parser.add_argument("--no-trajectory", action="store_true", help="Do not generate centroid trajectory synthesis images.")
    args = parser.parse_args()

    if args.path.name.startswith("infer_"):
        infer_dir = args.path
    else:
        infer_dir = infer_dir_from_trace(args.path, args.episode, args.infer)

    attn_path = infer_dir / f"attn_denoise_step_{args.step:03d}.pt"
    if not attn_path.exists():
        raise FileNotFoundError(attn_path)
    payload = torch.load(attn_path, map_location="cpu")
    layer_name = parse_layer(args.layer)
    full_layers = payload.get("full_layers")
    if not full_layers:
        raise ValueError(
            f"{attn_path} does not contain full_layers. Re-run trace with PI05_TRACE_SAVE_ATTN_FULL=1."
        )
    if layer_name not in full_layers:
        raise KeyError(f"{layer_name} not in full_layers. Available: {sorted(full_layers)}")
    if args.image_key not in full_layers[layer_name]:
        raise KeyError(f"{args.image_key} not in {layer_name}. Available: {sorted(full_layers[layer_name])}")

    full = full_layers[layer_name][args.image_key].detach().to(torch.float32).numpy()
    # [heads, suffix_query_len, grid_h, grid_w]
    num_heads, suffix_query_len, grid_h, grid_w = full.shape
    query_axis = payload.get("query_axis", {})
    state_count = args.state_query_count
    if state_count is None:
        state_count = int(query_axis.get("state_query_count", max(0, suffix_query_len - args.max_action_queries)))
    state_count = max(0, min(state_count, suffix_query_len))
    action_start = state_count
    action_end = min(suffix_query_len, action_start + args.max_action_queries)
    heads = parse_heads(args.heads, num_heads)

    image_meta = find_image_meta(payload, args.image_key)
    obs_key = image_meta.get("obs_key", args.image_key)
    obs = np.load(infer_dir / "obs.npz")
    base_image = image_hwc(obs[obs_key])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    all_centroids: dict[int, np.ndarray] = {}
    summary = {
        "infer_dir": str(infer_dir),
        "attn_path": str(attn_path),
        "step": args.step,
        "layer": layer_name,
        "image_key": args.image_key,
        "obs_key": obs_key,
        "shape": list(full.shape),
        "state_query_count": state_count,
        "action_query_range": [action_start, action_end],
        "heads": heads,
        "trajectory_image": None,
    }

    for head in heads:
        head_dir = args.out_dir / f"head_{head:02d}"
        head_dir.mkdir(parents=True, exist_ok=True)
        state_overlay = None
        if state_count > 0:
            state_heat = full[head, :state_count].mean(axis=0)
            state_overlay_path = head_dir / f"h{head:02d}_state_queries.png"
            overlay_image(
                base_image,
                state_heat,
                state_overlay_path,
                percentile=args.percentile,
                vmax=args.vmax,
                alpha=args.alpha,
                draw_grid=args.draw_grid,
                mask_color=(255, 24, 0),
                top_blocks=None,
            )
            state_overlay = state_overlay_path.relative_to(args.out_dir).as_posix()

        action_paths: list[Path] = []
        action_labels: list[str] = []
        action_heatmaps = full[head, action_start:action_end]
        centroids = attention_centroids(action_heatmaps)
        all_centroids[head] = centroids
        trajectory_overlay = None
        if not args.no_trajectory:
            trajectory_path = head_dir / f"h{head:02d}_action_centroid_trajectory.png"
            draw_trajectory(
                base_image,
                {head: centroids},
                trajectory_path,
                grid_h=grid_h,
                grid_w=grid_w,
                per_query_color=True,
                draw_grid=args.draw_grid,
            )
            trajectory_overlay = trajectory_path.relative_to(args.out_dir).as_posix()
        for query_idx in range(action_start, action_end):
            action_t = query_idx - action_start
            heat = full[head, query_idx]
            out_path = head_dir / f"h{head:02d}_action_{action_t:03d}_q{query_idx:03d}.png"
            overlay_image(
                base_image,
                heat,
                out_path,
                percentile=args.percentile,
                vmax=args.vmax,
                alpha=args.alpha,
                draw_grid=args.draw_grid,
                mask_color=(255, 24, 0),
                top_blocks=None,
            )
            action_paths.append(out_path)
            action_labels.append(f"a{action_t:02d}/q{query_idx:02d}")

        sheet_path = head_dir / f"h{head:02d}_action_queries_sheet.png"
        save_contact_sheet(action_paths, action_labels, sheet_path, columns=args.columns, thumb_width=args.thumb_width)
        rows.append(
            {
                "head": head,
                "state_overlay": state_overlay,
                "trajectory_overlay": trajectory_overlay,
                "centroids": centroids.tolist(),
                "action_sheet": sheet_path.relative_to(args.out_dir).as_posix(),
                "action_images": [path.relative_to(args.out_dir).as_posix() for path in action_paths],
            }
        )

    if all_centroids and not args.no_trajectory:
        all_traj_path = args.out_dir / "all_heads_action_centroid_trajectory.png"
        draw_trajectory(
            base_image,
            all_centroids,
            all_traj_path,
            grid_h=grid_h,
            grid_w=grid_w,
            per_query_color=False,
            draw_grid=args.draw_grid,
        )
        summary["trajectory_image"] = all_traj_path.relative_to(args.out_dir).as_posix()

    summary["rows"] = rows
    (args.out_dir / "summary.json").write_text(
        __import__("json").dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    html_rows = []
    for row in rows:
        state_html = (
            f"<a href='{html.escape(row['state_overlay'])}'><img src='{html.escape(row['state_overlay'])}'></a>"
            if row["state_overlay"]
            else "none"
        )
        sheet = html.escape(row["action_sheet"])
        trajectory_html = (
            f"<a href='{html.escape(row['trajectory_overlay'])}'><img src='{html.escape(row['trajectory_overlay'])}'></a>"
            if row.get("trajectory_overlay")
            else "none"
        )
        html_rows.append(
            f"<tr><td>H{row['head']}</td><td>{state_html}</td>"
            f"<td>{trajectory_html}</td><td><a href='{sheet}'><img src='{sheet}'></a></td></tr>"
        )

    all_traj_html = ""
    if summary.get("trajectory_image"):
        traj = html.escape(str(summary["trajectory_image"]))
        all_traj_html = f"<h2>All-head attention centroid trajectories</h2><p><a href='{traj}'><img src='{traj}'></a></p>"

    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Denoise Head/Query Attention</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    img {{ max-width: 1000px; border: 1px solid #ddd; }}
    td:first-child {{ font-weight: 700; font-size: 20px; }}
  </style>
</head>
<body>
  <h1>Denoise Head/Query Attention</h1>
  <p><code>{html.escape(str(infer_dir))}</code></p>
  <p>step=<code>{args.step}</code>, layer=<code>{html.escape(layer_name)}</code>, image=<code>{html.escape(args.image_key)}</code></p>
  <p>full attention shape = <code>[heads={num_heads}, queries={suffix_query_len}, grid={grid_h}x{grid_w}]</code>.
  State queries: <code>{state_count}</code>; action queries shown: <code>{action_end - action_start}</code>.</p>
  {all_traj_html}
  <table>
    <tr><th>head</th><th>state query heatmap</th><th>action centroid trajectory</th><th>action query heatmaps in temporal order</th></tr>
    {''.join(html_rows)}
  </table>
</body>
</html>
"""
    (args.out_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"Wrote {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()

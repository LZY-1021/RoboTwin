#!/usr/bin/env python3
"""Render PI05 trace patch/prefix/KV differences as image overlays.

The script compares two inference trace directories and projects per-patch or
per-block cosine differences back onto the original 3-view observations.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_dir_from_trace(path: Path, episode: int, infer: int) -> Path:
    root = path.parent if path.name.startswith("episode_") else path
    return root / f"episode_{episode:04d}" / f"infer_{infer:04d}"


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size == 0 or b.size == 0:
        return float("nan")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def cosine_torch(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().to(torch.float32).reshape(-1)
    b = b.detach().to(torch.float32).reshape(-1)
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom) == 0.0:
        return float("nan")
    return float(torch.dot(a, b) / denom)


def cosine_diff(value: float) -> float:
    if math.isnan(value):
        return float("nan")
    return max(0.0, 1.0 - value)


def parse_layers(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(item) for item in value.split(",") if item.strip()]


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_").replace(".", "_")


def parse_color(value: str) -> tuple[int, int, int]:
    presets = {
        "red": (255, 24, 0),
        "green": (0, 230, 80),
        "cyan": (0, 220, 255),
        "blue": (40, 120, 255),
        "magenta": (255, 0, 220),
        "yellow": (255, 220, 0),
    }
    key = value.strip().lower()
    if key in presets:
        return presets[key]
    if key.startswith("#"):
        key = key[1:]
    if len(key) == 6:
        return tuple(int(key[i : i + 2], 16) for i in (0, 2, 4))
    parts = [part.strip() for part in value.split(",")]
    if len(parts) == 3:
        rgb = tuple(int(part) for part in parts)
        if all(0 <= part <= 255 for part in rgb):
            return rgb
    raise argparse.ArgumentTypeError(
        "Color must be a preset red/green/cyan/blue/magenta/yellow, #RRGGBB, or R,G,B."
    )


def image_hwc(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in {1, 3, 4}:
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        if image.max(initial=0) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] > 3:
        image = image[:, :, :3]
    return image


def block_ranges(grid_h: int, grid_w: int, block_size: int):
    for row_start in range(0, grid_h, block_size):
        row_end = min(grid_h, row_start + block_size)
        for col_start in range(0, grid_w, block_size):
            col_end = min(grid_w, col_start + block_size)
            offsets = [
                row * grid_w + col
                for row in range(row_start, row_end)
                for col in range(col_start, col_end)
            ]
            yield row_start, row_end, col_start, col_end, offsets


def obs_patch_vector(image: np.ndarray, grid_h: int, grid_w: int, offsets: list[int]) -> np.ndarray:
    height, width = image.shape[:2]
    patch_h = height // grid_h
    patch_w = width // grid_w
    patches = []
    for offset in offsets:
        row = offset // grid_w
        col = offset % grid_w
        patch = image[row * patch_h : (row + 1) * patch_h, col * patch_w : (col + 1) * patch_w]
        patches.append(patch.reshape(-1))
    return np.concatenate(patches, axis=0) if patches else np.empty((0,), dtype=np.float32)


def kv_tensor(layer_kv: Any, token_indices: list[int], kind: str) -> torch.Tensor:
    tensor = layer_kv[0] if kind == "k" else layer_kv[1]
    if tensor.ndim != 4:
        raise ValueError(f"Expected KV tensor [B,H,S,D], got {tuple(tensor.shape)}")
    return tensor[0, :, token_indices, :]


def normalize_heatmap(values: np.ndarray, percentile: float, vmax: float | None) -> np.ndarray:
    heat = np.asarray(values, dtype=np.float32)
    finite = heat[np.isfinite(heat)]
    if finite.size == 0:
        return np.zeros_like(heat, dtype=np.float32)
    scale = float(vmax) if vmax is not None else float(np.percentile(finite, percentile))
    if scale <= 0:
        scale = float(finite.max(initial=0.0))
    if scale <= 0:
        return np.zeros_like(heat, dtype=np.float32)
    return np.clip(np.nan_to_num(heat, nan=0.0) / scale, 0.0, 1.0)


def overlay_image(
    base: np.ndarray,
    heat: np.ndarray,
    out_path: Path,
    *,
    percentile: float,
    vmax: float | None,
    alpha: float,
    draw_grid: bool,
    mask_color: tuple[int, int, int],
    top_blocks: list[dict[str, Any]] | None,
) -> None:
    base_img = Image.fromarray(image_hwc(base)).convert("RGBA")
    height, width = base_img.height, base_img.width
    norm = normalize_heatmap(heat, percentile=percentile, vmax=vmax)
    heat_img = Image.fromarray((norm * 255).astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST)

    # Red mask with opacity proportional to normalized difference. Low-diff
    # background remains nearly transparent.
    mask_alpha = np.asarray(heat_img, dtype=np.float32) / 255.0
    mask_alpha = np.clip(mask_alpha * alpha * 255.0, 0, 255).astype(np.uint8)
    mask = np.zeros((height, width, 4), dtype=np.uint8)
    mask[:, :, 0] = mask_color[0]
    mask[:, :, 1] = mask_color[1]
    mask[:, :, 2] = mask_color[2]
    mask[:, :, 3] = mask_alpha
    overlay = Image.fromarray(mask)
    composed = Image.alpha_composite(base_img, overlay)

    if draw_grid:
        draw = ImageDraw.Draw(composed)
        grid_h, grid_w = heat.shape
        for row in range(1, grid_h):
            y = round(row * height / grid_h)
            draw.line([(0, y), (width, y)], fill=(255, 255, 255, 70), width=1)
        for col in range(1, grid_w):
            x = round(col * width / grid_w)
            draw.line([(x, 0), (x, height)], fill=(255, 255, 255, 70), width=1)

    if top_blocks:
        draw = ImageDraw.Draw(composed)
        grid_h, grid_w = heat.shape
        for block in top_blocks:
            row_start, row_end = block["rows"]
            col_start, col_end = block["cols"]
            x0 = round(col_start * width / grid_w)
            x1 = round(col_end * width / grid_w)
            y0 = round(row_start * height / grid_h)
            y1 = round(row_end * height / grid_h)
            draw.rectangle([(x0, y0), (x1, y1)], outline=(0, 90, 255, 255), width=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    composed.convert("RGB").save(out_path)


def token_grid_image(
    grid_h: int,
    grid_w: int,
    token_start: int,
    top_blocks: list[dict[str, Any]],
    out_path: Path,
    *,
    block_size: int,
) -> None:
    cell = 38 if grid_w <= 16 else 28
    margin = 34
    width = margin + grid_w * cell + 1
    height = margin + grid_h * cell + 1
    image = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(image)

    top_lookup: dict[tuple[int, int], int] = {}
    for rank, block in enumerate(top_blocks, start=1):
        for row in range(block["rows"][0], block["rows"][1]):
            for col in range(block["cols"][0], block["cols"][1]):
                top_lookup[(row, col)] = rank

    for row in range(grid_h):
        y0 = margin + row * cell
        draw.text((4, y0 + cell // 3), str(row), fill=(80, 80, 80))
    for col in range(grid_w):
        x0 = margin + col * cell
        draw.text((x0 + cell // 3, 8), str(col), fill=(80, 80, 80))

    for row in range(grid_h):
        for col in range(grid_w):
            x0 = margin + col * cell
            y0 = margin + row * cell
            x1 = x0 + cell
            y1 = y0 + cell
            rank = top_lookup.get((row, col))
            fill = (210, 255, 220) if rank is not None else (255, 255, 255)
            outline = (0, 180, 80) if rank is not None else (190, 190, 190)
            draw.rectangle([(x0, y0), (x1, y1)], fill=fill, outline=outline, width=2 if rank else 1)
            token_id = token_start + row * grid_w + col
            draw.text((x0 + 3, y0 + 3), str(token_id), fill=(20, 20, 20))
            if rank is not None:
                draw.text((x0 + 3, y0 + cell - 15), f"#{rank}", fill=(0, 110, 50))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def save_reference_image(image: np.ndarray, out_dir: Path, stem: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    Image.fromarray(image_hwc(image)).save(path)
    return path.name


def save_heatmap_artifacts(
    base_image: np.ndarray,
    heat: np.ndarray,
    out_dir: Path,
    stem: str,
    args: argparse.Namespace,
    summary: dict[str, Any],
    token_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"{stem}.npy"
    png_path = out_dir / f"{stem}_overlay.png"
    np.save(npy_path, heat)
    top_blocks = top_heatmap_blocks(heat, args.top_blocks, args.block_size)
    token_grid_name = None
    if token_meta is not None and args.save_token_grid:
        grid_h, grid_w = token_meta["grid"]
        token_grid_path = out_dir / f"{stem}_token_grid.png"
        token_grid_image(
            grid_h,
            grid_w,
            int(token_meta["token_start"]),
            top_blocks,
            token_grid_path,
            block_size=args.block_size,
        )
        token_grid_name = token_grid_path.name
    overlay_image(
        base_image,
        heat,
        png_path,
        percentile=args.percentile,
        vmax=args.vmax,
        alpha=args.alpha,
        draw_grid=args.draw_grid,
        mask_color=args.mask_color,
        top_blocks=top_blocks if args.annotate_top_blocks else None,
    )
    finite = heat[np.isfinite(heat)]
    stats = {
        "name": stem,
        "overlay": png_path.name,
        "npy": npy_path.name,
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "max": float(finite.max()) if finite.size else float("nan"),
        "p95": float(np.percentile(finite, 95)) if finite.size else float("nan"),
        "top_blocks": top_blocks,
    }
    if token_meta is not None:
        stats["token_start"] = int(token_meta["token_start"])
        stats["token_end"] = int(token_meta["token_end"])
        stats["grid"] = token_meta["grid"]
        for block in stats["top_blocks"]:
            token_indices = []
            for row in range(block["rows"][0], block["rows"][1]):
                for col in range(block["cols"][0], block["cols"][1]):
                    token_indices.append(int(token_meta["token_start"]) + row * int(token_meta["grid"][1]) + col)
            block["token_indices"] = token_indices
    if token_grid_name is not None:
        stats["token_grid"] = token_grid_name
    summary.setdefault("heatmaps", []).append(stats)
    return stats


def save_attention_compare_artifacts(
    base_image_a: np.ndarray,
    base_image_b: np.ndarray,
    heat_a: np.ndarray,
    heat_b: np.ndarray,
    out_dir: Path,
    stem: str,
    args: argparse.Namespace,
    summary: dict[str, Any],
    token_meta: dict[str, Any],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    abs_diff_heat = np.abs(heat_a - heat_b)
    if args.attn_diff_mode in {"relative", "weighted_relative"}:
        denom = np.maximum(np.maximum(np.abs(heat_a), np.abs(heat_b)), args.attn_relative_eps)
        relative_diff_heat = abs_diff_heat / denom
        if args.attn_diff_mode == "weighted_relative":
            importance = np.maximum(np.abs(heat_a), np.abs(heat_b))
            importance_scale = float(np.nanmax(importance))
            if importance_scale > 0:
                importance = importance / importance_scale
            diff_heat = relative_diff_heat * importance
        else:
            diff_heat = relative_diff_heat
    else:
        diff_heat = abs_diff_heat
    np.save(out_dir / f"{stem}_score_a.npy", heat_a)
    np.save(out_dir / f"{stem}_score_b.npy", heat_b)
    np.save(out_dir / f"{stem}_diff.npy", diff_heat)
    np.save(out_dir / f"{stem}_abs_diff.npy", abs_diff_heat)

    top_blocks = top_heatmap_blocks(diff_heat, args.top_blocks, args.block_size)
    token_grid_name = None
    if args.save_token_grid:
        grid_h, grid_w = token_meta["grid"]
        token_grid_path = out_dir / f"{stem}_token_grid.png"
        token_grid_image(
            grid_h,
            grid_w,
            int(token_meta["token_start"]),
            top_blocks,
            token_grid_path,
            block_size=args.block_size,
        )
        token_grid_name = token_grid_path.name

    a_overlay = out_dir / f"{stem}_score_a_overlay.png"
    b_overlay = out_dir / f"{stem}_score_b_overlay.png"
    diff_overlay = out_dir / f"{stem}_diff_overlay.png"
    overlay_image(
        base_image_a,
        heat_a,
        a_overlay,
        percentile=args.percentile,
        vmax=args.attn_score_vmax if args.attn_score_vmax is not None else args.vmax,
        alpha=args.alpha,
        draw_grid=args.draw_grid,
        mask_color=(255, 24, 0),
        top_blocks=None,
    )
    overlay_image(
        base_image_b,
        heat_b,
        b_overlay,
        percentile=args.percentile,
        vmax=args.attn_score_vmax if args.attn_score_vmax is not None else args.vmax,
        alpha=args.alpha,
        draw_grid=args.draw_grid,
        mask_color=(255, 24, 0),
        top_blocks=None,
    )
    overlay_image(
        base_image_a,
        diff_heat,
        diff_overlay,
        percentile=args.percentile,
        vmax=args.attn_diff_vmax if args.attn_diff_vmax is not None else args.vmax,
        alpha=args.alpha,
        draw_grid=args.draw_grid,
        mask_color=(0, 230, 80),
        top_blocks=top_blocks if args.annotate_top_blocks else None,
    )

    finite = diff_heat[np.isfinite(diff_heat)]
    stats = {
        "name": stem,
        "score_a_overlay": a_overlay.name,
        "score_b_overlay": b_overlay.name,
        "diff_overlay": diff_overlay.name,
        "score_a_npy": f"{stem}_score_a.npy",
        "score_b_npy": f"{stem}_score_b.npy",
        "diff_npy": f"{stem}_diff.npy",
        "abs_diff_npy": f"{stem}_abs_diff.npy",
        "diff_mode": args.attn_diff_mode,
        "relative_eps": args.attn_relative_eps,
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "max": float(finite.max()) if finite.size else float("nan"),
        "p95": float(np.percentile(finite, 95)) if finite.size else float("nan"),
        "top_blocks": top_blocks,
        "token_start": int(token_meta["token_start"]),
        "token_end": int(token_meta["token_end"]),
        "grid": token_meta["grid"],
    }
    for block in stats["top_blocks"]:
        token_indices = []
        for row in range(block["rows"][0], block["rows"][1]):
            for col in range(block["cols"][0], block["cols"][1]):
                token_indices.append(int(token_meta["token_start"]) + row * int(token_meta["grid"][1]) + col)
        block["token_indices"] = token_indices
    if token_grid_name is not None:
        stats["token_grid"] = token_grid_name
    summary.setdefault("attn_compare_heatmaps", []).append(stats)
    return stats


def top_heatmap_blocks(heat: np.ndarray, top_k: int, block_size: int) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    grid_h, grid_w = heat.shape
    records = []
    for row_start, row_end, col_start, col_end, _ in block_ranges(grid_h, grid_w, max(1, block_size)):
        block = heat[row_start:row_end, col_start:col_end]
        finite = block[np.isfinite(block)]
        if finite.size == 0:
            continue
        records.append(
            {
                "rows": [int(row_start), int(row_end)],
                "cols": [int(col_start), int(col_end)],
                "mean": float(finite.mean()),
                "max": float(finite.max()),
            }
        )
    records.sort(key=lambda item: item["mean"], reverse=True)
    return records[:top_k]


def load_attention_trace(infer_dir: Path) -> dict[tuple[int, str, str], np.ndarray]:
    out: dict[tuple[int, str, str], np.ndarray] = {}
    prefix_path = infer_dir / "attn_prefix.pt"
    if prefix_path.exists():
        payload = torch.load(prefix_path, map_location="cpu")
        for layer_name, layer_payload in payload.get("layers", {}).items():
            for image_key, heat in layer_payload.items():
                out[(-1, layer_name, image_key)] = np.asarray(heat, dtype=np.float32)
    for path in sorted(infer_dir.glob("attn_denoise_step_*.pt")):
        try:
            step = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        payload = torch.load(path, map_location="cpu")
        for layer_name, layer_payload in payload.get("layers", {}).items():
            for image_key, heat in layer_payload.items():
                out[(step, layer_name, image_key)] = np.asarray(heat, dtype=np.float32)
    return out


def load_mlp_trace(infer_dir: Path, layers: list[int] | None) -> dict[int, dict[str, torch.Tensor]]:
    mlp_dir = infer_dir / "mlp_prefix"
    if not mlp_dir.exists():
        mlp_dir = infer_dir / "mlp"
    out: dict[int, dict[str, torch.Tensor]] = {}
    if not mlp_dir.exists():
        return out
    selected = set(layers) if layers is not None else None
    for path in sorted(mlp_dir.glob("layer_*.pt")):
        try:
            layer = int(path.stem.split("_", 1)[1])
        except ValueError:
            continue
        if selected is not None and layer not in selected:
            continue
        payload = torch.load(path, map_location="cpu")
        out[layer] = payload
    return out


def make_html(out_dir: Path, summary: dict[str, Any]) -> None:
    rows = []
    camera_images = summary.get("camera_images", {})
    label_a = summary.get("label_a", "infer A")
    label_b = summary.get("label_b", "infer B")
    for item in summary.get("heatmaps", []):
        refs = camera_images.get(item.get("camera", ""), {})
        image_a = refs.get("infer_a", "")
        image_b = refs.get("infer_b", "")
        top_text = "<br>".join(
            f"#{idx}: r{block['rows'][0]}:{block['rows'][1]} c{block['cols'][0]}:{block['cols'][1]} "
            f"tok={block.get('token_indices', [])[:6]}{'...' if len(block.get('token_indices', [])) > 6 else ''} "
            f"mean={block['mean']:.4g} max={block['max']:.4g}"
            for idx, block in enumerate(item.get("top_blocks", []), start=1)
        )
        token_grid = item.get("token_grid", "")
        token_grid_html = (
            f"<a href='{html.escape(token_grid)}'><img src='{html.escape(token_grid)}'></a>" if token_grid else ""
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['name'])}</td>"
            f"<td>{item['mean']:.6g}</td>"
            f"<td>{item['p95']:.6g}</td>"
            f"<td>{item['max']:.6g}</td>"
            f"<td>{top_text}</td>"
            f"<td>{token_grid_html}</td>"
            f"<td><a href='{html.escape(image_a)}'><img src='{html.escape(image_a)}'></a></td>"
            f"<td><a href='{html.escape(image_b)}'><img src='{html.escape(image_b)}'></a></td>"
            f"<td><a href='{html.escape(item['overlay'])}'><img src='{html.escape(item['overlay'])}'></a></td>"
            "</tr>"
        )
    attn_rows = []
    for item in summary.get("attn_compare_heatmaps", []):
        top_text = "<br>".join(
            f"#{idx}: r{block['rows'][0]}:{block['rows'][1]} c{block['cols'][0]}:{block['cols'][1]} "
            f"tok={block.get('token_indices', [])[:6]}{'...' if len(block.get('token_indices', [])) > 6 else ''} "
            f"mean={block['mean']:.4g} max={block['max']:.4g}"
            for idx, block in enumerate(item.get("top_blocks", []), start=1)
        )
        token_grid = item.get("token_grid", "")
        token_grid_html = (
            f"<a href='{html.escape(token_grid)}'><img src='{html.escape(token_grid)}'></a>" if token_grid else ""
        )
        attn_rows.append(
            "<tr>"
            f"<td>{html.escape(item['name'])}</td>"
            f"<td>{item['mean']:.6g}</td>"
            f"<td>{item['p95']:.6g}</td>"
            f"<td>{item['max']:.6g}</td>"
            f"<td>{top_text}</td>"
            f"<td>{token_grid_html}</td>"
            f"<td><a href='{html.escape(item['score_a_overlay'])}'><img src='{html.escape(item['score_a_overlay'])}'></a></td>"
            f"<td><a href='{html.escape(item['score_b_overlay'])}'><img src='{html.escape(item['score_b_overlay'])}'></a></td>"
            f"<td><a href='{html.escape(item['diff_overlay'])}'><img src='{html.escape(item['diff_overlay'])}'></a></td>"
            "</tr>"
        )
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>PI05 Patch/KV Overlay</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f7f7f7; color: #111; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #eee; position: sticky; top: 0; }}
    img {{ max-width: 240px; height: auto; display: block; }}
    code {{ background: #eee; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>PI05 Patch/KV Overlay</h1>
  <p><code>{html.escape(summary['pair'])}</code></p>
  <p>metric = <code>1 - cosine</code>; stronger mask opacity means larger difference or score.</p>
  <p>Attention rows combine three overlays: A score in red, B score in red, and attention diff in green. Current attention diff mode: <code>{html.escape(str(summary.get('attn_diff_mode', 'abs')))}</code>.</p>
  <p><code>abs</code> diff is <code>abs(attn A - attn B)</code>; <code>relative</code> diff is <code>abs(attn A - attn B) / max(abs(attn A), abs(attn B), eps)</code>; <code>weighted_relative</code> multiplies relative diff by normalized <code>max(abs(attn A), abs(attn B))</code>.</p>
  <p>Use <code>--attn-diff-vmax</code> to render attention diff with a fixed scale instead of per-heatmap normalization.</p>
  <p>KV overlays are projected by each image patch token's original position. Deeper transformer layers mix information across tokens, so this is a source-token view, not pixel-level attribution.</p>
  <h2>Attention score/diff</h2>
  <table>
    <tr><th>attention</th><th>diff mean</th><th>diff p95</th><th>diff max</th><th>top diff patch/block</th><th>token grid</th><th>{html.escape(label_a)} score</th><th>{html.escape(label_b)} score</th><th>diff overlay on {html.escape(label_a)}</th></tr>
    {''.join(attn_rows)}
  </table>
  <h2>Obs / Prefix / KV diff</h2>
  <table>
    <tr><th>heatmap</th><th>mean</th><th>p95</th><th>max</th><th>top patch/block</th><th>token grid</th><th>{html.escape(label_a)}</th><th>{html.escape(label_b)}</th><th>diff overlay on {html.escape(label_a)}</th></tr>
    {''.join(rows)}
  </table>
</body>
</html>
"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def compare_pair(
    infer_a: Path,
    infer_b: Path,
    out_dir: Path,
    args: argparse.Namespace,
    *,
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    meta_a = load_json(infer_a / "prefix_token_meta.json")
    meta_b = load_json(infer_b / "prefix_token_meta.json")
    embeds_a = torch.load(infer_a / "prefix_image_embeds.pt", map_location="cpu")
    embeds_b = torch.load(infer_b / "prefix_image_embeds.pt", map_location="cpu")
    kv_a = None if args.no_kv else torch.load(infer_a / "kv_prefix_current.pt", map_location="cpu")
    kv_b = None if args.no_kv else torch.load(infer_b / "kv_prefix_current.pt", map_location="cpu")
    obs_a = np.load(infer_a / "obs.npz")
    obs_b = np.load(infer_b / "obs.npz")
    attn_a = {} if args.no_attn else load_attention_trace(infer_a)
    attn_b = {} if args.no_attn else load_attention_trace(infer_b)
    mlp_a = {} if args.no_mlp else load_mlp_trace(infer_a, args.mlp_layers)
    mlp_b = {} if args.no_mlp else load_mlp_trace(infer_b, args.mlp_layers)

    if kv_a is None or kv_b is None:
        layers = args.layers if args.layers is not None else []
    else:
        layers = args.layers if args.layers is not None else list(range(min(len(kv_a), len(kv_b))))
    meta_by_key_b = {item["image_key"]: item for item in meta_b["image_tokens"]}
    summary: dict[str, Any] = {
        "pair": f"{infer_a} -> {infer_b}",
        "block_size": args.block_size,
        "layers": layers,
        "label_a": label_a,
        "label_b": label_b,
        "camera_images": {},
        "heatmaps": [],
        "attn_diff_mode": args.attn_diff_mode,
        "attn_relative_eps": args.attn_relative_eps,
    }

    for item_a in meta_a["image_tokens"]:
        image_key = item_a["image_key"]
        item_b = meta_by_key_b.get(image_key)
        if item_b is None:
            continue
        if item_a["grid"] != item_b["grid"] or item_a["num_tokens"] != item_b["num_tokens"]:
            continue

        grid_h, grid_w = item_a["grid"]
        obs_key = item_a.get("obs_key", image_key)
        base_image = image_hwc(obs_a[obs_key])
        image_b = image_hwc(obs_b[obs_key])
        emb_a = embeds_a[image_key][0]
        emb_b = embeds_b[image_key][0]
        token_start = int(item_a["token_start"])
        camera = safe_name(image_key)
        summary["camera_images"][camera] = {
            "infer_a": save_reference_image(base_image, out_dir, f"{camera}_infer_a"),
            "infer_b": save_reference_image(image_b, out_dir, f"{camera}_infer_b"),
        }

        obs_heat = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
        prefix_heat = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
        mlp_heats = {
            (layer, name): np.full((grid_h, grid_w), np.nan, dtype=np.float32)
            for layer in sorted(set(mlp_a).intersection(mlp_b))
            for name in ("x", "y")
            if name in mlp_a[layer] and name in mlp_b[layer]
        }
        kv_heats = {}
        if kv_a is not None and kv_b is not None:
            kv_heats = {
                (layer, kind): np.full((grid_h, grid_w), np.nan, dtype=np.float32)
                for layer in layers
                for kind in ("k", "v")
                if 0 <= layer < min(len(kv_a), len(kv_b))
            }

        for row_start, row_end, col_start, col_end, offsets in block_ranges(grid_h, grid_w, args.block_size):
            token_indices = [token_start + offset for offset in offsets]
            obs_value = cosine_diff(
                cosine_np(
                    obs_patch_vector(base_image, grid_h, grid_w, offsets),
                    obs_patch_vector(image_b, grid_h, grid_w, offsets),
                )
            )
            prefix_value = cosine_diff(cosine_torch(emb_a[offsets], emb_b[offsets]))
            obs_heat[row_start:row_end, col_start:col_end] = obs_value
            prefix_heat[row_start:row_end, col_start:col_end] = prefix_value

            for layer in sorted(set(mlp_a).intersection(mlp_b)):
                for name in ("x", "y"):
                    if (layer, name) not in mlp_heats:
                        continue
                    tensor_a = mlp_a[layer][name]
                    tensor_b = mlp_b[layer][name]
                    if tensor_a.ndim != 3 or tensor_b.ndim != 3:
                        continue
                    token_indices = [token_start + offset for offset in offsets]
                    if max(token_indices, default=-1) >= tensor_a.shape[1] or max(token_indices, default=-1) >= tensor_b.shape[1]:
                        continue
                    mlp_value = cosine_diff(cosine_torch(tensor_a[0, token_indices, :], tensor_b[0, token_indices, :]))
                    mlp_heats[(layer, name)][row_start:row_end, col_start:col_end] = mlp_value

            if kv_a is not None and kv_b is not None:
                for layer in layers:
                    if layer < 0 or layer >= min(len(kv_a), len(kv_b)):
                        continue
                    for kind in ("k", "v"):
                        kv_value = cosine_diff(
                            cosine_torch(
                                kv_tensor(kv_a[layer], token_indices, kind),
                                kv_tensor(kv_b[layer], token_indices, kind),
                            )
                        )
                        kv_heats[(layer, kind)][row_start:row_end, col_start:col_end] = kv_value

        save_heatmap_artifacts(base_image, obs_heat, out_dir, f"{camera}_obs", args, summary, item_a)
        summary["heatmaps"][-1]["camera"] = camera
        save_heatmap_artifacts(base_image, prefix_heat, out_dir, f"{camera}_prefix_embed", args, summary, item_a)
        summary["heatmaps"][-1]["camera"] = camera
        for layer in sorted(set(mlp_a).intersection(mlp_b)):
            for name in ("x", "y"):
                heat = mlp_heats.get((layer, name))
                if heat is None:
                    continue
                save_heatmap_artifacts(base_image, heat, out_dir, f"{camera}_mlp_layer{layer:02d}_{name}", args, summary, item_a)
                summary["heatmaps"][-1]["camera"] = camera
        for layer in layers:
            for kind in ("k", "v"):
                heat = kv_heats.get((layer, kind))
                if heat is None:
                    continue
                save_heatmap_artifacts(base_image, heat, out_dir, f"{camera}_kv_layer{layer:02d}_{kind}", args, summary, item_a)
                summary["heatmaps"][-1]["camera"] = camera

        common_attn_keys = sorted(
            key for key in set(attn_a).intersection(attn_b) if key[2] == image_key
        )
        for step, layer_name, _ in common_attn_keys:
            heat_a = attn_a[(step, layer_name, image_key)]
            heat_b = attn_b[(step, layer_name, image_key)]
            if heat_a.shape != heat_b.shape:
                continue
            save_attention_compare_artifacts(
                base_image,
                image_b,
                heat_a,
                heat_b,
                out_dir,
                f"{camera}_attn_prefix_{safe_name(layer_name)}" if step < 0 else f"{camera}_attn_step{step:03d}_{safe_name(layer_name)}",
                args,
                summary,
                item_a,
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    make_html(out_dir, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Traces directory or episode directory.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index for adjacent mode.")
    parser.add_argument("--infer-a", type=int, default=0, help="First inference index.")
    parser.add_argument("--infer-b", type=int, default=1, help="Second inference index.")
    parser.add_argument("--episode-a", type=int, default=None, help="Cross-episode first episode index.")
    parser.add_argument("--episode-b", type=int, default=None, help="Cross-episode second episode index.")
    parser.add_argument("--block-size", type=int, default=1, help="Patch tokens per block side. Use 4 for 4x4 blocks.")
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("0,3,6,9,12,15,17"))
    parser.add_argument("--mlp-layers", type=parse_layers, default=parse_layers("0,3,6,9,12,15,17"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--percentile", type=float, default=95.0, help="Percentile used to normalize each heatmap.")
    parser.add_argument("--vmax", type=float, default=None, help="Fixed raw diff value mapped to max red.")
    parser.add_argument("--attn-score-vmax", type=float, default=None, help="Fixed raw attention score mapped to max red for A/B attention score overlays.")
    parser.add_argument("--attn-diff-vmax", type=float, default=None, help="Fixed attention diff value mapped to max green for attention diff overlays.")
    parser.add_argument(
        "--attn-diff-mode",
        choices=("abs", "relative", "weighted_relative"),
        default="abs",
        help="Attention diff metric: abs(a-b), relative, or relative weighted by normalized max attention score.",
    )
    parser.add_argument("--attn-relative-eps", type=float, default=1e-12, help="Epsilon denominator for --attn-diff-mode relative.")
    parser.add_argument("--alpha", type=float, default=0.65, help="Maximum overlay alpha in [0, 1].")
    parser.add_argument(
        "--mask-color",
        type=parse_color,
        default=parse_color("red"),
        help="Overlay color: red/green/cyan/blue/magenta/yellow, #RRGGBB, or R,G,B.",
    )
    parser.add_argument("--draw-grid", action="store_true", help="Draw patch/block grid over the overlay.")
    parser.add_argument("--top-blocks", type=int, default=5, help="Record top-k highest patch/block regions per heatmap.")
    parser.add_argument("--annotate-top-blocks", action="store_true", help="Draw numbered boxes for top patch/block regions.")
    parser.add_argument("--save-token-grid", action="store_true", help="Save token-grid visualization for each heatmap.")
    parser.add_argument("--no-attn", action="store_true", help="Do not include attn_denoise_step_*.pt overlays.")
    parser.add_argument("--no-kv", action="store_true", help="Do not load or render kv_prefix_current.pt overlays.")
    parser.add_argument("--no-mlp", action="store_true", help="Do not load or render mlp/layer_*.pt overlays.")
    args = parser.parse_args()

    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")

    if args.episode_a is not None or args.episode_b is not None:
        if args.episode_a is None or args.episode_b is None:
            raise ValueError("--episode-a and --episode-b must be passed together.")
        infer_a = infer_dir_from_trace(args.path, args.episode_a, args.infer_a)
        infer_b = infer_dir_from_trace(args.path, args.episode_b, args.infer_b)
        label_a = f"episode {args.episode_a} / infer {args.infer_a}"
        label_b = f"episode {args.episode_b} / infer {args.infer_b}"
    else:
        root = args.path.parent if args.path.name.startswith("episode_") else args.path
        infer_a = root / f"episode_{args.episode:04d}" / f"infer_{args.infer_a:04d}"
        infer_b = root / f"episode_{args.episode:04d}" / f"infer_{args.infer_b:04d}"
        label_a = f"episode {args.episode} / infer {args.infer_a}"
        label_b = f"episode {args.episode} / infer {args.infer_b}"

    summary = compare_pair(infer_a, infer_b, args.out_dir, args, label_a=label_a, label_b=label_b)
    print(f"Pair: {summary['pair']}")
    print(f"Wrote overlays: {args.out_dir}")
    print(f"Open summary: {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()

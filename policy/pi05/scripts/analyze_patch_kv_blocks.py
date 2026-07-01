#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_dirs(episode_dir: Path) -> list[Path]:
    return sorted(path for path in episode_dir.glob("infer_*") if path.is_dir())


def infer_dir_from_trace(path: Path, episode: int, infer: int) -> Path:
    root = path.parent if path.name.startswith("episode_") else path
    return root / f"episode_{episode:04d}" / f"infer_{infer:04d}"


def resolve_episode_dir(path: Path, episode: int | None) -> Path:
    if path.name.startswith("episode_"):
        return path
    if episode is None:
        raise ValueError("Pass an episode directory, or pass a traces directory with --episode.")
    return path / f"episode_{episode:04d}"


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


def block_ranges(grid_h: int, grid_w: int, block_size: int):
    for row_start in range(0, grid_h, block_size):
        row_end = min(grid_h, row_start + block_size)
        for col_start in range(0, grid_w, block_size):
            col_end = min(grid_w, col_start + block_size)
            token_offsets = [
                row * grid_w + col
                for row in range(row_start, row_end)
                for col in range(col_start, col_end)
            ]
            yield (
                row_start // block_size,
                col_start // block_size,
            ), (row_start, row_end, col_start, col_end), token_offsets


def obs_patch_block(obs_npz, obs_key: str, grid_h: int, grid_w: int, token_offsets: list[int]) -> np.ndarray:
    image = np.asarray(obs_npz[obs_key])
    if image.ndim == 3 and image.shape[0] in {1, 3, 4}:
        image = np.transpose(image, (1, 2, 0))
    height, width = image.shape[:2]
    patch_h = height // grid_h
    patch_w = width // grid_w
    patches = []
    for offset in token_offsets:
        row = offset // grid_w
        col = offset % grid_w
        patch = image[row * patch_h : (row + 1) * patch_h, col * patch_w : (col + 1) * patch_w]
        patches.append(patch.reshape(-1))
    return np.concatenate(patches, axis=0) if patches else np.empty((0,), dtype=np.float32)


def kv_block(layer_kv, token_indices: list[int], kind: str) -> torch.Tensor:
    tensor = layer_kv[0] if kind == "k" else layer_kv[1]
    if tensor.ndim != 4:
        raise ValueError(f"Expected KV tensor [B,H,S,D], got shape {tuple(tensor.shape)}")
    return tensor[0, :, token_indices, :]


def summarize_layers(kv_a, kv_b, token_indices: list[int], layers: list[int] | None) -> dict[str, float]:
    total_layers = min(len(kv_a), len(kv_b))
    selected_layers = layers if layers is not None else list(range(total_layers))
    out = {}
    for layer in selected_layers:
        if layer < 0 or layer >= total_layers:
            continue
        out[f"layer_{layer:02d}_k"] = cosine_torch(
            kv_block(kv_a[layer], token_indices, "k"),
            kv_block(kv_b[layer], token_indices, "k"),
        )
        out[f"layer_{layer:02d}_v"] = cosine_torch(
            kv_block(kv_a[layer], token_indices, "v"),
            kv_block(kv_b[layer], token_indices, "v"),
        )
    return out


def mean_selected(values: dict[str, float], suffix: str) -> float:
    selected = [value for key, value in values.items() if key.endswith(suffix) and not np.isnan(value)]
    return float(np.mean(selected)) if selected else float("nan")


def compare_pair(infer_a: Path, infer_b: Path, block_size: int, layers: list[int] | None) -> list[dict[str, Any]]:
    meta_a = load_json(infer_a / "prefix_token_meta.json")
    meta_b = load_json(infer_b / "prefix_token_meta.json")
    embeds_a = torch.load(infer_a / "prefix_image_embeds.pt", map_location="cpu")
    embeds_b = torch.load(infer_b / "prefix_image_embeds.pt", map_location="cpu")
    kv_a = torch.load(infer_a / "kv_prefix_current.pt", map_location="cpu")
    kv_b = torch.load(infer_b / "kv_prefix_current.pt", map_location="cpu")
    obs_a = np.load(infer_a / "obs.npz")
    obs_b = np.load(infer_b / "obs.npz")

    records = []
    meta_by_key_b = {item["image_key"]: item for item in meta_b["image_tokens"]}
    for item_a in meta_a["image_tokens"]:
        image_key = item_a["image_key"]
        item_b = meta_by_key_b.get(image_key)
        if item_b is None:
            continue
        if item_a["grid"] != item_b["grid"] or item_a["num_tokens"] != item_b["num_tokens"]:
            continue

        grid_h, grid_w = item_a["grid"]
        emb_a = embeds_a[image_key][0]
        emb_b = embeds_b[image_key][0]
        token_start = int(item_a["token_start"])
        obs_key = item_a.get("obs_key", image_key)

        for block_id, patch_range, offsets in block_ranges(grid_h, grid_w, block_size):
            token_indices = [token_start + offset for offset in offsets]
            obs_cos = cosine_np(
                obs_patch_block(obs_a, obs_key, grid_h, grid_w, offsets),
                obs_patch_block(obs_b, obs_key, grid_h, grid_w, offsets),
            )
            emb_cos = cosine_torch(emb_a[offsets], emb_b[offsets])
            layer_values = summarize_layers(kv_a, kv_b, token_indices, layers)
            record = {
                "camera": image_key,
                "obs_key": obs_key,
                "block": block_id,
                "patch_rows": [patch_range[0], patch_range[1]],
                "patch_cols": [patch_range[2], patch_range[3]],
                "token_indices": token_indices,
                "num_tokens": len(token_indices),
                "obs_patch_cos": obs_cos,
                "prefix_embed_cos": emb_cos,
                "kv_k_mean": mean_selected(layer_values, "_k"),
                "kv_v_mean": mean_selected(layer_values, "_v"),
                "layers": layer_values,
            }
            records.append(record)
    return records


def parse_layers(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(item) for item in value.split(",") if item.strip()]


def fmt(value: float) -> str:
    return "nan" if np.isnan(value) else f"{value:.6f}"


def print_records(records: list[dict[str, Any]], top_k: int) -> None:
    ranked = sorted(records, key=lambda item: item["kv_k_mean"])
    for item in ranked[:top_k]:
        block_row, block_col = item["block"]
        row_start, row_end = item["patch_rows"]
        col_start, col_end = item["patch_cols"]
        print(
            f"  {item['camera']} block=({block_row},{block_col}) "
            f"patch_rows={row_start}:{row_end} patch_cols={col_start}:{col_end} "
            f"tokens={item['num_tokens']} "
            f"obs={fmt(item['obs_patch_cos'])} "
            f"emb={fmt(item['prefix_embed_cos'])} "
            f"kv_k_mean={fmt(item['kv_k_mean'])} "
            f"kv_v_mean={fmt(item['kv_v_mean'])}"
        )
        for key in sorted(item["layers"]):
            print(f"    {key}: {fmt(item['layers'][key])}")


def print_pair_summary(records: list[dict[str, Any]]) -> None:
    def mean(key: str) -> float:
        values = [item[key] for item in records if not np.isnan(item[key])]
        return float(np.mean(values)) if values else float("nan")

    print(
        "Summary: "
        f"obs={fmt(mean('obs_patch_cos'))} "
        f"emb={fmt(mean('prefix_embed_cos'))} "
        f"kv_k={fmt(mean('kv_k_mean'))} "
        f"kv_v={fmt(mean('kv_v_mean'))}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze PI05 obs patch, prefix image token, and prefix KV block cosine within one trace episode."
    )
    parser.add_argument("path", type=Path, help="Episode directory, or traces directory with --episode.")
    parser.add_argument("--episode", type=int, default=None, help="Episode index if path is a traces directory.")
    parser.add_argument("--episode-a", type=int, default=None, help="Specific pair mode: first episode index.")
    parser.add_argument("--infer-a", type=int, default=0, help="Specific pair mode: first inference index.")
    parser.add_argument("--episode-b", type=int, default=None, help="Specific pair mode: second episode index.")
    parser.add_argument("--infer-b", type=int, default=0, help="Specific pair mode: second inference index.")
    parser.add_argument("--start", type=int, default=0, help="First inference index.")
    parser.add_argument("--end", type=int, default=None, help="Exclusive end inference index.")
    parser.add_argument("--block-size", type=int, default=4, help="Patch tokens per block side.")
    parser.add_argument("--layers", type=str, default="0,3,6,9,12,15,17", help="Comma-separated KV layers to compare.")
    parser.add_argument("--top-k", type=int, default=8, help="Print lowest-kv-similarity blocks per adjacent pair.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to write full JSON records.")
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    all_records = []

    if args.episode_a is not None or args.episode_b is not None:
        if args.episode_a is None or args.episode_b is None:
            raise ValueError("--episode-a and --episode-b must be passed together.")
        infer_a = infer_dir_from_trace(args.path, args.episode_a, args.infer_a)
        infer_b = infer_dir_from_trace(args.path, args.episode_b, args.infer_b)
        records = compare_pair(infer_a, infer_b, args.block_size, layers)
        for record in records:
            record["infer_a"] = str(infer_a.relative_to(args.path if not args.path.name.startswith("episode_") else args.path.parent))
            record["infer_b"] = str(infer_b.relative_to(args.path if not args.path.name.startswith("episode_") else args.path.parent))
        all_records.extend(records)
        print(f"Pair: {infer_a} -> {infer_b}")
        print(f"Block size: {args.block_size}x{args.block_size} patch tokens")
        print(f"Layers: {layers if layers is not None else 'all'}")
        print_pair_summary(records)
        print(f"=== most changed KV blocks ===")
        print_records(records, args.top_k)
        print()
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            with args.json_out.open("w", encoding="utf-8") as f:
                json.dump(all_records, f, ensure_ascii=False, indent=2)
            print(f"Wrote JSON: {args.json_out}")
        return

    episode_dir = resolve_episode_dir(args.path, args.episode)
    infers = infer_dirs(episode_dir)
    end = len(infers) if args.end is None else min(args.end, len(infers))

    print(f"Episode: {episode_dir}")
    print(f"Adjacent comparisons: {max(0, end - args.start - 1)}")
    print(f"Block size: {args.block_size}x{args.block_size} patch tokens")
    print(f"Layers: {layers if layers is not None else 'all'}")
    print()

    for idx in range(args.start, max(args.start, end - 1)):
        infer_a = infers[idx]
        infer_b = infers[idx + 1]
        records = compare_pair(infer_a, infer_b, args.block_size, layers)
        for record in records:
            record["infer_a"] = infer_a.name
            record["infer_b"] = infer_b.name
        all_records.extend(records)
        print(f"=== {infer_a.name} -> {infer_b.name} | most changed KV blocks ===")
        print_records(records, args.top_k)
        print()

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)
        print(f"Wrote JSON: {args.json_out}")


if __name__ == "__main__":
    main()

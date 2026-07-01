#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def cosine(a: Any, b: Any) -> float | None:
    a_tensor = to_tensor(a)
    b_tensor = to_tensor(b)
    if a_tensor is None or b_tensor is None or a_tensor.numel() == 0 or b_tensor.numel() == 0:
        return None
    n = min(a_tensor.numel(), b_tensor.numel())
    a_tensor = a_tensor.reshape(-1)[:n].float()
    b_tensor = b_tensor.reshape(-1)[:n].float()
    a_norm = torch.linalg.vector_norm(a_tensor)
    b_norm = torch.linalg.vector_norm(b_tensor)
    if a_norm.item() == 0.0 or b_norm.item() == 0.0:
        return None
    value = float(torch.dot(a_tensor, b_tensor) / (a_norm * b_norm))
    return max(-1.0, min(1.0, value))


def to_tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, (list, tuple)):
        pieces = [to_tensor(item) for item in value]
        pieces = [item.reshape(-1) for item in pieces if item is not None]
        if not pieces:
            return None
        return torch.cat(pieces)
    if isinstance(value, dict):
        pieces = [to_tensor(value[key]) for key in sorted(value)]
        pieces = [item.reshape(-1) for item in pieces if item is not None]
        if not pieces:
            return None
        return torch.cat(pieces)
    return torch.as_tensor(value)


def fmt(value: float | None) -> str:
    if value is None:
        return "nan"
    return f"{value:.6f}"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_torch(path: Path) -> Any:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def episode_dir(trace_dir: Path, episode_index: int) -> Path:
    path = trace_dir / f"episode_{episode_index:04d}"
    if not path.exists():
        raise FileNotFoundError(f"Episode directory not found: {path}")
    return path


def infer_dirs(ep_dir: Path) -> list[Path]:
    return sorted(path for path in ep_dir.glob("infer_*") if path.is_dir())


def compare_obs(infer_a: Path, infer_b: Path) -> dict[str, float | None]:
    obs_a = load_npz(infer_a / "obs.npz")
    obs_b = load_npz(infer_b / "obs.npz")
    keys = sorted(set(obs_a) & set(obs_b))
    result = {f"obs.{key}": cosine(obs_a[key], obs_b[key]) for key in keys}
    if keys:
        result["obs.all"] = cosine([obs_a[key] for key in keys], [obs_b[key] for key in keys])
    return result


def compare_action(infer_a: Path, infer_b: Path) -> dict[str, float | None]:
    action_a = load_npz(infer_a / "action.npz")
    action_b = load_npz(infer_b / "action.npz")
    keys = sorted(set(action_a) & set(action_b))
    return {f"action.{key}": cosine(action_a[key], action_b[key]) for key in keys}


def compare_kv_file(path_a: Path, path_b: Path, label: str) -> dict[str, float | None]:
    kv_a = load_torch(path_a)
    kv_b = load_torch(path_b)
    if kv_a is None or kv_b is None:
        return {}
    result: dict[str, float | None] = {f"{label}.all": cosine(kv_a, kv_b)}
    layer_scores = []
    for layer_idx, (layer_a, layer_b) in enumerate(zip(kv_a, kv_b, strict=False)):
        key_score = cosine(layer_a[0], layer_b[0])
        value_score = cosine(layer_a[1], layer_b[1])
        result[f"{label}.layer_{layer_idx:03d}.key"] = key_score
        result[f"{label}.layer_{layer_idx:03d}.value"] = value_score
        if key_score is not None:
            layer_scores.append(key_score)
        if value_score is not None:
            layer_scores.append(value_score)
    result[f"{label}.layer_mean"] = float(np.mean(layer_scores)) if layer_scores else None
    return result


def compare_kv(infer_a: Path, infer_b: Path) -> dict[str, float | None]:
    result = {}
    names = sorted({path.name for path in infer_a.glob("kv_*.pt")} & {path.name for path in infer_b.glob("kv_*.pt")})
    for name in names:
        label = name[:-3]
        result.update(compare_kv_file(infer_a / name, infer_b / name, label))
    return result


def compare_mlp(infer_a: Path, infer_b: Path) -> dict[str, float | None]:
    mlp_a = infer_a / "mlp"
    mlp_b = infer_b / "mlp"
    if not mlp_a.exists() or not mlp_b.exists():
        return {}
    names = sorted({path.name for path in mlp_a.glob("layer_*.pt")} & {path.name for path in mlp_b.glob("layer_*.pt")})
    result: dict[str, float | None] = {}
    y_scores = []
    x_scores = []
    for name in names:
        layer_a = load_torch(mlp_a / name)
        layer_b = load_torch(mlp_b / name)
        layer = name.removesuffix(".pt")
        x_score = cosine(layer_a.get("x"), layer_b.get("x"))
        y_score = cosine(layer_a.get("y"), layer_b.get("y"))
        changed_score = cosine(layer_a.get("changed"), layer_b.get("changed"))
        result[f"mlp.{layer}.x"] = x_score
        result[f"mlp.{layer}.y"] = y_score
        result[f"mlp.{layer}.changed"] = changed_score
        if x_score is not None:
            x_scores.append(x_score)
        if y_score is not None:
            y_scores.append(y_score)
    result["mlp.x_mean"] = float(np.mean(x_scores)) if x_scores else None
    result["mlp.y_mean"] = float(np.mean(y_scores)) if y_scores else None
    return result


def action_trajectory(infers: list[Path], key: str) -> torch.Tensor | None:
    pieces = []
    for infer_dir in infers:
        action = load_npz(infer_dir / "action.npz")
        if key in action:
            pieces.append(to_tensor(action[key]).reshape(-1))
    if not pieces:
        return None
    return torch.cat(pieces)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def print_section(title: str, values: dict[str, float | None], max_layers: int | None) -> None:
    print(title)
    for key in sorted(values):
        if max_layers is not None and ".layer_" in key:
            layer_text = key.split(".layer_", 1)[1][:3]
            if layer_text.isdigit() and int(layer_text) >= max_layers:
                continue
        print(f"  {key}: {fmt(values[key])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two PI05 trace episodes with cosine similarity.")
    parser.add_argument("trace_dir", type=Path, help="Path to a traces directory.")
    parser.add_argument("--episode-a", type=int, required=True)
    parser.add_argument("--episode-b", type=int, required=True)
    parser.add_argument("--max-layers", type=int, default=None, help="Limit printed per-layer KV/MLP rows.")
    args = parser.parse_args()

    ep_a = episode_dir(args.trace_dir, args.episode_a)
    ep_b = episode_dir(args.trace_dir, args.episode_b)
    infers_a = infer_dirs(ep_a)
    infers_b = infer_dirs(ep_b)
    pair_count = min(len(infers_a), len(infers_b))

    print(f"Trace: {args.trace_dir}")
    print(f"Episode A: {ep_a.name} {read_json(ep_a / 'meta.json')}")
    print(f"Episode B: {ep_b.name} {read_json(ep_b / 'meta.json')}")
    print(f"Aligned inferences: {pair_count}")
    print()

    for idx in range(pair_count):
        infer_a = infers_a[idx]
        infer_b = infers_b[idx]
        print(f"=== infer_{idx:04d} ===")
        print_section("obs", compare_obs(infer_a, infer_b), args.max_layers)
        print_section("action", compare_action(infer_a, infer_b), args.max_layers)
        print_section("kv", compare_kv(infer_a, infer_b), args.max_layers)
        print_section("mlp", compare_mlp(infer_a, infer_b), args.max_layers)
        print()

    print("=== episode action trajectory ===")
    for key in ("action_executed", "action_chunk_full"):
        traj_a = action_trajectory(infers_a[:pair_count], key)
        traj_b = action_trajectory(infers_b[:pair_count], key)
        print(f"  trajectory.{key}: {fmt(cosine(traj_a, traj_b))}")


if __name__ == "__main__":
    main()

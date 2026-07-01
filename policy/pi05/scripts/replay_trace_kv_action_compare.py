#!/usr/bin/env python3
"""Replay PI05 trace observations with donor prefix KV and compare actions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
PI05_DIR = REPO_ROOT / "policy/pi05"
if str(PI05_DIR) not in sys.path:
    sys.path.insert(0, str(PI05_DIR))
SRC_DIR = PI05_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pi_model import PI0  # noqa: E402


def infer_dir(trace_dir: Path, episode: int, infer: int) -> Path:
    return trace_dir / f"episode_{episode:04d}" / f"infer_{infer:04d}"


def resolve_infer_index(trace_dir: Path, episode: int, value: str) -> int:
    ep_dir = trace_dir / f"episode_{episode:04d}"
    infers = sorted(path for path in ep_dir.glob("infer_*") if path.is_dir())
    if not infers:
        raise FileNotFoundError(f"No infer_* directories found in {ep_dir}")
    last_idx = int(infers[-1].name.split("_", 1)[1])
    token = value.strip().lower()
    if token == "last":
        return last_idx
    if token.startswith("last-"):
        return last_idx - int(token.split("-", 1)[1])
    idx = int(token)
    if idx < 0:
        return last_idx + 1 + idx
    return idx


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_observation(path: Path) -> dict[str, Any]:
    obs_npz = np.load(path / "obs.npz")
    meta = load_json(path / "obs_meta.json")
    prompt = meta.get("prompt")
    if prompt is None:
        raise ValueError(f"Missing prompt in {path / 'obs_meta.json'}")
    return {
        "state": np.asarray(obs_npz["state"]),
        "images": {
            "cam_high": np.asarray(obs_npz["cam_high"]),
            "cam_left_wrist": np.asarray(obs_npz["cam_left_wrist"]),
            "cam_right_wrist": np.asarray(obs_npz["cam_right_wrist"]),
        },
        "prompt": prompt,
    }


def load_trace_action(path: Path) -> np.ndarray | None:
    action_path = path / "action.npz"
    if not action_path.exists():
        return None
    action_npz = np.load(action_path)
    if "action_chunk_full" in action_npz:
        return np.asarray(action_npz["action_chunk_full"])
    if "action_executed" in action_npz:
        return np.asarray(action_npz["action_executed"])
    return None


def cosine_np(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    n = min(a.size, b.size)
    if n == 0:
        return None
    a = a[:n]
    b = b[:n]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return None
    return float(np.dot(a, b) / denom)


def l2_np(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    n = min(a.size, b.size)
    if n == 0:
        return None
    return float(np.linalg.norm(a[:n] - b[:n]) / max(n, 1) ** 0.5)


def action_diff_summary(stale: np.ndarray, fresh: np.ndarray, top_k: int) -> dict[str, Any]:
    stale = np.asarray(stale, dtype=np.float32)
    fresh = np.asarray(fresh, dtype=np.float32)
    steps = min(stale.shape[0], fresh.shape[0])
    dims = min(stale.shape[1], fresh.shape[1])
    stale = stale[:steps, :dims]
    fresh = fresh[:steps, :dims]
    diff = stale - fresh
    abs_diff = np.abs(diff)
    per_step_rmse = np.sqrt(np.mean(diff * diff, axis=1))
    per_dim_rmse = np.sqrt(np.mean(diff * diff, axis=0))
    per_step_max_abs = np.max(abs_diff, axis=1)
    per_dim_max_abs = np.max(abs_diff, axis=0)

    flat = abs_diff.reshape(-1)
    top_indices = np.argsort(flat)[::-1][: max(0, top_k)]
    top_errors = []
    for flat_idx in top_indices:
        step = int(flat_idx // dims)
        dim = int(flat_idx % dims)
        top_errors.append(
            {
                "step": step,
                "dim": dim,
                "abs_diff": float(abs_diff[step, dim]),
                "signed_diff": float(diff[step, dim]),
                "fresh": float(fresh[step, dim]),
                "stale": float(stale[step, dim]),
            }
        )

    return {
        "per_step_rmse": per_step_rmse.tolist(),
        "per_step_max_abs": per_step_max_abs.tolist(),
        "per_dim_rmse": per_dim_rmse.tolist(),
        "per_dim_max_abs": per_dim_max_abs.tolist(),
        "max_step_rmse": {
            "step": int(np.argmax(per_step_rmse)),
            "value": float(np.max(per_step_rmse)),
        },
        "max_dim_rmse": {
            "dim": int(np.argmax(per_dim_rmse)),
            "value": float(np.max(per_dim_rmse)),
        },
        "top_errors": top_errors,
    }


def pair_name(target_dir: Path, donor_dir: Path) -> str:
    target_ep = target_dir.parent.name
    target_infer = target_dir.name
    donor_ep = donor_dir.parent.name
    donor_infer = donor_dir.name
    return f"target_{target_ep}_{target_infer}__donor_{donor_ep}_{donor_infer}"


def move_kv_to_device(kv: Any, device: str) -> Any:
    if torch.is_tensor(kv):
        return kv.to(device)
    if isinstance(kv, tuple):
        return tuple(move_kv_to_device(item, device) for item in kv)
    if isinstance(kv, list):
        return [move_kv_to_device(item, device) for item in kv]
    return kv


def clear_reuse_state(torch_model: Any) -> None:
    for attr in ("_last_prefix_pad_masks", "_last_past_key_values", "_last_kv_mode_stats"):
        if hasattr(torch_model, attr):
            setattr(torch_model, attr, None)


def infer_action(policy: Any, torch_model: Any, obs: dict[str, Any], noise: np.ndarray, mode: str) -> np.ndarray:
    torch_model._denoise_kv_mode = mode
    return policy.infer(obs, noise=noise)["actions"]


def run_one(
    model: PI0,
    target_dir: Path,
    donor_dir: Path,
    mode: str,
    noise_seed: int,
    cutoff: int,
    layers_per_step: int,
    initial_current_layers: int,
    save_action_details_dir: Path | None,
    top_error_k: int,
) -> dict[str, Any]:
    os.environ["PI05_DENOISE_KV_CUTOFF_STEP"] = str(cutoff)
    os.environ["PI05_DENOISE_KV_LAYERS_PER_STEP"] = str(layers_per_step)
    os.environ["PI05_DENOISE_KV_INITIAL_CURRENT_LAYERS"] = str(initial_current_layers)

    policy = model.policy
    torch_model = policy._model
    obs = load_observation(target_dir)
    trace_action = load_trace_action(target_dir)

    # Noise is consumed before output transforms, so it must use the model's
    # padded internal action dimension, not the task action dimension saved in
    # trace action.npz after postprocessing.
    action_shape = (
        int(torch_model.config.action_horizon),
        int(torch_model.config.action_dim),
    )
    rng = np.random.default_rng(noise_seed)
    noise = rng.normal(size=action_shape).astype(np.float32)

    clear_reuse_state(torch_model)
    fresh_action = infer_action(policy, torch_model, obs, noise, "fresh")

    # Prime current prefix masks for the target observation, then replace only KV
    # with the donor KV. This keeps masks/sequence shape aligned with the target.
    clear_reuse_state(torch_model)
    _ = infer_action(policy, torch_model, obs, noise, mode)
    donor_kv = torch.load(donor_dir / "kv_prefix_current.pt", map_location="cpu")
    torch_model._last_past_key_values = move_kv_to_device(donor_kv, policy._pytorch_device)

    stale_action = infer_action(policy, torch_model, obs, noise, mode)

    result = {
        "target": str(target_dir),
        "donor": str(donor_dir),
        "mode": mode,
        "noise_seed": noise_seed,
        "cos_stale_vs_fresh": cosine_np(stale_action, fresh_action),
        "l2_stale_vs_fresh": l2_np(stale_action, fresh_action),
        "cos_fresh_vs_trace": cosine_np(fresh_action, trace_action),
        "cos_stale_vs_trace": cosine_np(stale_action, trace_action),
        "action_shape": list(stale_action.shape),
    }
    if save_action_details_dir is not None:
        save_action_details_dir.mkdir(parents=True, exist_ok=True)
        stem = pair_name(target_dir, donor_dir)
        detail_path = save_action_details_dir / f"{stem}.npz"
        np.savez_compressed(
            detail_path,
            fresh_action=np.asarray(fresh_action),
            stale_action=np.asarray(stale_action),
            diff=np.asarray(stale_action) - np.asarray(fresh_action),
            abs_diff=np.abs(np.asarray(stale_action) - np.asarray(fresh_action)),
            trace_action=np.asarray(trace_action) if trace_action is not None else np.asarray([]),
        )
        result["action_detail_npz"] = str(detail_path)
        result["action_diff_summary"] = action_diff_summary(stale_action, fresh_action, top_error_k)
    return result


def fmt(value: float | None) -> str:
    return "nan" if value is None else f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--train-config-name", default="pi05_base_finetune_on_robotwin_clean_randomized_joint_training")
    parser.add_argument("--model-name", default="pi05_robotwin2")
    parser.add_argument("--checkpoint-id", default="30000")
    parser.add_argument("--pi0-step", type=int, default=32)
    parser.add_argument("--mode", choices=["step_cutoff", "layer_accumulate"], default="step_cutoff")
    parser.add_argument("--cutoff", type=int, default=5)
    parser.add_argument("--layers-per-step", type=int, default=2)
    parser.add_argument("--initial-current-layers", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Target pair EP,INFER. INFER supports 0, 1, -1, last, last-1. Can repeat.",
    )
    parser.add_argument(
        "--donor",
        action="append",
        default=[],
        help="Donor pair EP,INFER. INFER supports 0, 1, -1, last, last-1. Can repeat.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--save-action-details", type=Path, default=None, help="Directory for per-pair fresh/stale action npz and detailed error summaries.")
    parser.add_argument("--top-error-k", type=int, default=12, help="Number of largest per-step/per-dim action errors to store.")
    args = parser.parse_args()

    if not args.target:
        args.target = ["0,1", "1,1"]
    if not args.donor:
        args.donor = ["0,0", "1,0"]

    def parse_pair(value: str) -> tuple[int, int]:
        ep, inf = value.split(",", 1)
        episode = int(ep)
        return episode, resolve_infer_index(args.trace_dir, episode, inf)

    os.environ.setdefault("PI05_TORCH_COMPILE", "0")
    model = PI0(args.train_config_name, args.model_name, args.checkpoint_id, args.pi0_step)

    results = []
    for target_text in args.target:
        target_ep, target_inf = parse_pair(target_text)
        target_dir = infer_dir(args.trace_dir, target_ep, target_inf)
        for donor_text in args.donor:
            donor_ep, donor_inf = parse_pair(donor_text)
            donor_dir = infer_dir(args.trace_dir, donor_ep, donor_inf)
            result = run_one(
                model,
                target_dir,
                donor_dir,
                args.mode,
                args.noise_seed,
                args.cutoff,
                args.layers_per_step,
                args.initial_current_layers,
                args.save_action_details,
                args.top_error_k,
            )
            results.append(result)
            print(
                f"target=e{target_ep}/i{target_inf} donor=e{donor_ep}/i{donor_inf} "
                f"mode={args.mode} cos_stale_vs_fresh={fmt(result['cos_stale_vs_fresh'])} "
                f"l2={fmt(result['l2_stale_vs_fresh'])} "
                f"fresh_vs_trace={fmt(result['cos_fresh_vs_trace'])} "
                f"stale_vs_trace={fmt(result['cos_stale_vs_trace'])}"
            )

    payload = {
        "trace_dir": str(args.trace_dir),
        "mode": args.mode,
        "cutoff": args.cutoff,
        "layers_per_step": args.layers_per_step,
        "initial_current_layers": args.initial_current_layers,
        "noise_seed": args.noise_seed,
        "results": results,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()

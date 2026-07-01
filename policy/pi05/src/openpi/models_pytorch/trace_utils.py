import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


_TRACE_ROOT: Path | None = None
_EPISODE_DIR: Path | None = None
_INFER_DIR: Path | None = None
_EPISODE_INDEX = -1
_INFER_INDEX = -1


def enabled() -> bool:
    return os.environ.get("PI05_TRACE_ENABLE", "0") == "1"


def _to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(item) for item in value)
    return value


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return str(value)


def configure(root: str | os.PathLike[str], metadata: dict[str, Any] | None = None) -> None:
    global _TRACE_ROOT
    if not enabled():
        return
    _TRACE_ROOT = Path(root)
    _TRACE_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(_TRACE_ROOT / "trace_meta.json", metadata or {})


def start_episode(index: int, metadata: dict[str, Any] | None = None) -> None:
    global _EPISODE_DIR, _INFER_DIR, _EPISODE_INDEX, _INFER_INDEX
    if not enabled() or _TRACE_ROOT is None:
        return
    _EPISODE_INDEX = index
    _INFER_INDEX = -1
    _INFER_DIR = None
    _EPISODE_DIR = _TRACE_ROOT / f"episode_{index:04d}"
    _EPISODE_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(_EPISODE_DIR / "meta.json", metadata or {})


def finish_episode(metadata: dict[str, Any] | None = None) -> None:
    if not enabled() or _EPISODE_DIR is None:
        return
    if metadata:
        _write_json(_EPISODE_DIR / "result.json", metadata)


def start_infer(obs: dict[str, Any]) -> Path | None:
    global _INFER_DIR, _INFER_INDEX
    if not enabled() or _EPISODE_DIR is None:
        return None
    _INFER_INDEX += 1
    _INFER_DIR = _EPISODE_DIR / f"infer_{_INFER_INDEX:04d}"
    _INFER_DIR.mkdir(parents=True, exist_ok=True)
    save_obs(obs)
    return _INFER_DIR


def current_infer_dir() -> Path | None:
    if not enabled():
        return None
    return _INFER_DIR


def save_obs(obs: dict[str, Any]) -> None:
    if not enabled() or _INFER_DIR is None:
        return
    images = obs.get("images", {})
    np.savez_compressed(
        _INFER_DIR / "obs.npz",
        state=np.asarray(obs.get("state")),
        cam_high=np.asarray(images.get("cam_high")),
        cam_left_wrist=np.asarray(images.get("cam_left_wrist")),
        cam_right_wrist=np.asarray(images.get("cam_right_wrist")),
    )
    _write_json(_INFER_DIR / "obs_meta.json", {"prompt": obs.get("prompt")})


def save_action(action_chunk: Any, executed_action: Any) -> None:
    if not enabled() or _INFER_DIR is None:
        return
    np.savez_compressed(
        _INFER_DIR / "action.npz",
        action_chunk_full=np.asarray(action_chunk),
        action_executed=np.asarray(executed_action),
    )


def save_prefix_tokens(meta: dict[str, Any], image_embeds: dict[str, torch.Tensor]) -> None:
    if not enabled() or _INFER_DIR is None or os.environ.get("PI05_TRACE_SAVE_PREFIX", "1") != "1":
        return
    _write_json(_INFER_DIR / "prefix_token_meta.json", meta)
    torch.save(_to_cpu(image_embeds), _INFER_DIR / "prefix_image_embeds.pt")


def save_kv(label: str, past_key_values: Any) -> None:
    if not enabled() or _INFER_DIR is None or os.environ.get("PI05_TRACE_SAVE_KV", "1") != "1":
        return
    torch.save(_to_cpu(past_key_values), _INFER_DIR / f"kv_{label}.pt")


def save_attention(label: str, payload: dict[str, Any]) -> None:
    if not enabled() or _INFER_DIR is None or os.environ.get("PI05_TRACE_SAVE_ATTN", "0") != "1":
        return
    torch.save(_to_cpu(payload), _INFER_DIR / f"attn_{label}.pt")


def save_qk_logits(label: str, payload: dict[str, Any]) -> None:
    if not enabled() or _INFER_DIR is None or os.environ.get("PI05_TRACE_SAVE_QK_LOGITS", "0") != "1":
        return
    torch.save(_to_cpu(payload), _INFER_DIR / f"qk_{label}.pt")


def save_mlp(layer: int, x: torch.Tensor, y: torch.Tensor, changed: torch.Tensor | None, stats: dict[str, Any]) -> None:
    if not enabled() or _INFER_DIR is None or os.environ.get("PI05_TRACE_SAVE_MLP", "1") != "1":
        return
    prefix_seq_len = 0
    meta_path = _INFER_DIR / "prefix_token_meta.json"
    if meta_path.exists():
        try:
            prefix_seq_len = int(json.loads(meta_path.read_text(encoding="utf-8")).get("prefix_seq_len", 0))
        except Exception:
            prefix_seq_len = 0
    seq_len = int(x.shape[1]) if torch.is_tensor(x) and x.ndim >= 2 else 0
    mlp_kind = "prefix" if prefix_seq_len > 0 and seq_len >= prefix_seq_len else "denoise"
    mlp_dir = _INFER_DIR / f"mlp_{mlp_kind}"
    mlp_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "x": _to_cpu(x),
        "y": _to_cpu(y),
        "changed": _to_cpu(changed) if changed is not None else None,
        "stats": stats,
        "mlp_kind": mlp_kind,
        "seq_len": seq_len,
        "prefix_seq_len": prefix_seq_len,
    }
    torch.save(payload, mlp_dir / f"layer_{layer:03d}.pt")
    # Backward-compatible location for prefix MLP only. Older traces may have
    # denoise MLP here because the file was overwritten later in the forward.
    if mlp_kind == "prefix":
        legacy_dir = _INFER_DIR / "mlp"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, legacy_dir / f"layer_{layer:03d}.pt")

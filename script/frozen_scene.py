"""Frozen initial-scene snapshot helpers used by external evaluators."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def _pose_component(value: Any, *, entity_name: str, component: str, size: int) -> list[float]:
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"cannot serialize {component} for scene entity {entity_name!r}"
        ) from exc
    if len(result) != size or not all(math.isfinite(item) for item in result):
        raise RuntimeError(
            f"invalid {component} for scene entity {entity_name!r}: "
            f"expected {size} finite values"
        )
    return result


def _pose_payload(pose: Any, *, entity_name: str) -> dict[str, list[float]]:
    if pose is None:
        raise RuntimeError(f"scene entity {entity_name!r} has no pose")
    return {
        "p": _pose_component(
            getattr(pose, "p", None),
            entity_name=entity_name,
            component="position",
            size=3,
        ),
        "q": _pose_component(
            getattr(pose, "q", None),
            entity_name=entity_name,
            component="quaternion",
            size=4,
        ),
    }


def _entity_name(entity: Any) -> str | None:
    get_name = getattr(entity, "get_name", None)
    if not callable(get_name):
        return None
    name = str(get_name()).strip()
    return name or None


def _capture_frozen_scene_snapshot(task_env: Any) -> dict[str, Any]:
    """Capture named actor and articulation poses in simulator enumeration order.

    Call after demo setup and scene overrides, before policy actions move any
    objects. The list ordering is part of the historical fingerprint contract.
    """
    scene = getattr(task_env, "scene", None)
    get_actors = getattr(scene, "get_all_actors", None)
    if not callable(get_actors):
        raise RuntimeError("scene snapshot requires scene.get_all_actors()")

    actors: list[dict[str, Any]] = []
    for entity in get_actors():
        name = _entity_name(entity)
        get_pose = getattr(entity, "get_pose", None)
        if name is None or not callable(get_pose):
            continue
        actors.append({"name": name, "pose": _pose_payload(get_pose(), entity_name=name)})

    articulations: list[dict[str, Any]] = []
    get_articulations = getattr(scene, "get_all_articulations", None)
    if callable(get_articulations):
        for entity in get_articulations():
            name = _entity_name(entity)
            get_pose = getattr(entity, "get_root_pose", None)
            if not callable(get_pose):
                get_pose = getattr(entity, "get_pose", None)
            if name is None or not callable(get_pose):
                continue
            articulations.append(
                {"name": name, "root_pose": _pose_payload(get_pose(), entity_name=name)}
            )

    if not actors and not articulations:
        raise RuntimeError("scene snapshot captured no named entities")
    return {
        "schema_version": 1,
        "actors": actors,
        "articulations": articulations,
    }


def _scene_snapshot_fingerprint(snapshot: dict[str, Any]) -> str:
    """Hash a scene snapshot with the established ACC-compatible JSON contract.

    Dictionary keys are sorted, while actor/articulation list order is retained
    to remain compatible with existing frozen manifests.
    """
    if not isinstance(snapshot, dict) or not snapshot:
        raise RuntimeError("scene fingerprint requires a non-empty snapshot object")
    try:
        encoded = json.dumps(
            snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("scene snapshot is not finite JSON data") from exc
    return hashlib.sha256(encoded).hexdigest()


# The archived evaluator uses this name for the same live-snapshot hash.
_frozen_snapshot_sha256 = _scene_snapshot_fingerprint

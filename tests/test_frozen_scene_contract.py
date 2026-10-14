import hashlib
import json
import math
from types import SimpleNamespace
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from script.frozen_scene import (
    _capture_frozen_scene_snapshot,
    _frozen_snapshot_sha256,
    _scene_snapshot_fingerprint,
)


class Entity:
    def __init__(self, name, position=(0.0, 1.0, 2.0), quaternion=(1.0, 0.0, 0.0, 0.0)):
        self.name = name
        self.pose = SimpleNamespace(p=position, q=quaternion)

    def get_name(self):
        return self.name

    def get_pose(self):
        return self.pose


class Articulation(Entity):
    def get_root_pose(self):
        return self.pose


def make_env(actors=None, articulations=None):
    scene = SimpleNamespace(
        get_all_actors=lambda: list(actors or []),
        get_all_articulations=lambda: list(articulations or []),
    )
    return SimpleNamespace(scene=scene)


def test_snapshot_keeps_legacy_schema_and_enumeration_order():
    env = make_env(
        [Entity("z_object"), Entity("a_object")],
        [Articulation("robot")],
    )

    snapshot = _capture_frozen_scene_snapshot(env)

    assert snapshot == {
        "schema_version": 1,
        "actors": [
            {"name": "z_object", "pose": {"p": [0.0, 1.0, 2.0], "q": [1.0, 0.0, 0.0, 0.0]}},
            {"name": "a_object", "pose": {"p": [0.0, 1.0, 2.0], "q": [1.0, 0.0, 0.0, 0.0]}},
        ],
        "articulations": [
            {"name": "robot", "root_pose": {"p": [0.0, 1.0, 2.0], "q": [1.0, 0.0, 0.0, 0.0]}}
        ],
    }


def test_fingerprint_matches_existing_acc_json_sha256_contract():
    snapshot = {
        "schema_version": 1,
        "actors": [{"name": "object", "pose": {"p": [1.0, 2.0, 3.0], "q": [1.0, 0.0, 0.0, 0.0]}}],
        "articulations": [],
    }
    encoded = json.dumps(
        snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    assert _scene_snapshot_fingerprint(snapshot) == hashlib.sha256(encoded).hexdigest()
    assert _frozen_snapshot_sha256(snapshot) == _scene_snapshot_fingerprint(snapshot)


def test_fingerprint_is_stable_and_detects_scene_changes():
    snapshot = _capture_frozen_scene_snapshot(make_env([Entity("object")]))

    assert _scene_snapshot_fingerprint(snapshot) == _scene_snapshot_fingerprint(snapshot)
    moved = json.loads(json.dumps(snapshot))
    moved["actors"][0]["pose"]["p"][0] += 0.01
    assert _scene_snapshot_fingerprint(moved) != _scene_snapshot_fingerprint(snapshot)
    renamed = json.loads(json.dumps(snapshot))
    renamed["actors"][0]["name"] = "other_object"
    assert _scene_snapshot_fingerprint(renamed) != _scene_snapshot_fingerprint(snapshot)


def test_fingerprint_preserves_order_sensitive_legacy_list_semantics():
    snapshot = _capture_frozen_scene_snapshot(
        make_env([Entity("first"), Entity("second")])
    )
    reversed_snapshot = json.loads(json.dumps(snapshot))
    reversed_snapshot["actors"].reverse()

    assert _scene_snapshot_fingerprint(reversed_snapshot) != _scene_snapshot_fingerprint(snapshot)


def test_capture_fails_closed_without_scene_api_or_named_entities():
    with pytest.raises(RuntimeError, match="get_all_actors"):
        _capture_frozen_scene_snapshot(SimpleNamespace(scene=SimpleNamespace()))
    with pytest.raises(RuntimeError, match="no named entities"):
        _capture_frozen_scene_snapshot(make_env())


def test_capture_rejects_non_finite_or_malformed_pose_values():
    with pytest.raises(RuntimeError, match="finite values"):
        _capture_frozen_scene_snapshot(
            make_env([Entity("bad", position=(math.nan, 0.0, 0.0))])
        )
    with pytest.raises(RuntimeError, match="expected 3 finite values"):
        _capture_frozen_scene_snapshot(make_env([Entity("bad", position=(0.0, 0.0))]))


def test_fingerprint_rejects_non_json_or_non_finite_snapshot():
    with pytest.raises(RuntimeError, match="finite JSON"):
        _scene_snapshot_fingerprint({"actors": [math.inf]})

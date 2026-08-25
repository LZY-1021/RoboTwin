"""Small, dependency-free gates for optional RoboTwin planner paths."""

from __future__ import annotations

import os


def should_skip_curobo_for_qpos(env: dict[str, str] | None = None) -> bool:
    """Return whether the opt-in frozen qpos path may omit CuRobo.

    The action type is part of the guard so an accidental environment flag
    cannot disable CuRobo for an EE-action policy.
    """

    values = os.environ if env is None else env
    raw_flag = str(
        values.get("ROBOTWIN_FROZEN_QPOS_NO_CUROBO")
        or values.get("LEROBOT_ROBOTWIN_SKIP_CUROBO", "")
    ).strip().lower()
    action_type = str(values.get("ROBOTWIN_EVAL_ACTION_TYPE", "")).strip().lower()
    return raw_flag in {"1", "true", "yes", "on"} and action_type == "qpos"

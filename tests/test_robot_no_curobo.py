import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robotwin_curobo_gate import should_skip_curobo_for_qpos


def test_no_curobo_flag_is_limited_to_qpos_eval(monkeypatch):
    monkeypatch.setenv("ROBOTWIN_FROZEN_QPOS_NO_CUROBO", "1")
    monkeypatch.setenv("ROBOTWIN_EVAL_ACTION_TYPE", "qpos")
    assert should_skip_curobo_for_qpos()

    monkeypatch.setenv("ROBOTWIN_EVAL_ACTION_TYPE", "ee")
    assert not should_skip_curobo_for_qpos()


def test_no_curobo_flag_is_opt_in(monkeypatch):
    monkeypatch.delenv("ROBOTWIN_FROZEN_QPOS_NO_CUROBO", raising=False)
    monkeypatch.setenv("ROBOTWIN_EVAL_ACTION_TYPE", "qpos")
    assert not should_skip_curobo_for_qpos()

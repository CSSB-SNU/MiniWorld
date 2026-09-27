"""Prevent the old patch stack from silently rewriting a consolidated engine."""

import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "engine_install_check",
    Path(__file__).resolve().parents[1] / "scripts/check_engine_install.py",
)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def test_old_engine_requires_reinstallation(tmp_path):
    with pytest.raises(RuntimeError, match="predates the consolidated main"):
        checker.check_installation(tmp_path, "1.0.0", {}, "a" * 40)
    assert list(tmp_path.iterdir()) == []


def test_wrong_pinned_revision_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="revision mismatch"):
        checker.check_installation(
            tmp_path, "2.0.0", {"vcs_info": {"commit_id": "b" * 40}}, "a" * 40,
        )


def test_incomplete_new_engine_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="incomplete engine installation"):
        checker.check_installation(tmp_path, "2.0.0", {}, "a" * 40)


def test_source_checkout_contains_required_engine_files():
    package = Path(__file__).resolve().parents[1] / ".engine-release-2.0.0/src/miniworld_engine"
    if not package.exists():
        pytest.skip("local engine source checkout is optional")
    checker.check_installation(package, "2.0.0", {}, "a" * 40)

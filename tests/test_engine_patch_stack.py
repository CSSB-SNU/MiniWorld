"""Reinstall, upgrade and refusal behavior for overlapping engine patches."""

import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "engine_patch_applier",
    Path(__file__).resolve().parents[1] / "scripts/apply_engine_audit_patches.py",
)
applier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(applier)


@pytest.mark.parametrize("initial", ["base\n", "first\n", "second\n"])
def test_overlapping_patch_stack_is_idempotent(tmp_path, initial):
    package, patches = tmp_path / "package", tmp_path / "patches"
    package.mkdir()
    patches.mkdir()
    (package / "module.py").write_text(initial)
    for name, before, after in [
        ("one.patch", "base", "first"),
        ("two.patch", "first", "second"),
    ]:
        (patches / name).write_text(
            "--- a/src/miniworld_engine/module.py\n+++ b/src/miniworld_engine/module.py\n"
            "@@ -1 +1 @@\n-" + before + "\n+" + after + "\n"
        )
    names = ("one.patch", "two.patch")
    applier.apply_patches(package, patches, names, check_only=True)
    assert (package / "module.py").read_text() == initial
    applier.apply_patches(package, patches, names)
    assert (package / "module.py").read_text() == "second\n"
    assert applier.apply_patches(package, patches, names) == []


def test_mismatch_leaves_all_installed_files_untouched(tmp_path):
    package, patches = tmp_path / "package", tmp_path / "patches"
    package.mkdir()
    patches.mkdir()
    (package / "a.py").write_text("base\n")
    (package / "b.py").write_text("user edit\n")
    for name, file in [("one.patch", "a.py"), ("two.patch", "b.py")]:
        (patches / name).write_text(
            f"--- a/src/miniworld_engine/{file}\n+++ b/src/miniworld_engine/{file}\n"
            "@@ -1 +1 @@\n-base\n+patched\n"
        )
    with pytest.raises(RuntimeError, match="Patch does not match"):
        applier.apply_patches(package, patches, ("one.patch", "two.patch"))
    assert (package / "a.py").read_text() == "base\n"
    assert (package / "b.py").read_text() == "user edit\n"

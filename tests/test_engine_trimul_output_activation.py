"""Activation preserves the running writer and rejects stale validation."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "trimul_activation", ROOT / "scripts/activate_trimul_output.py"
)
activation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(activation)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = tmp_path / "project"
    package = project / "package"
    build = project / "build"
    for d in [
        "scripts",
        "patches",
        "libs/team-gm/scripts",
        "libs/team-gm/patches",
        "package",
        "build",
    ]:
        (project / d).mkdir(parents=True)
    source = (ROOT / "scripts/apply_engine_audit_patches.py").read_text()
    _, node = activation.names_node(source)
    lines = source.splitlines(keepends=True)
    source = (
        "".join(lines[: node.lineno - 1])
        + "PATCH_NAMES = ()\n"
        + "".join(lines[node.end_lineno :])
    )
    for d in ["scripts", "libs/team-gm/scripts"]:
        (project / d / "apply_engine_audit_patches.py").write_text(source)
    target = package / "marker.py"
    target.write_text("old\n")
    cache = package / "triton-cache.json"
    cache.write_text("cache v1\n")
    patch = "--- a/src/miniworld_engine/marker.py\n+++ b/src/miniworld_engine/marker.py\n@@ -1 +1 @@\n-old\n+new\n"
    for d in ["patches", "libs/team-gm/patches"]:
        (project / d / "test.patch").write_text(patch)
    digest = lambda x: __import__("hashlib").sha256(x.encode()).hexdigest()
    manifest = project / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "patch": "test.patch",
                "patch_sha256": digest(patch),
                "files": {
                    "marker.py": {"before": digest("old\n"), "after": digest("new\n")}
                },
                "before_identity": "old",
                "after_identity": "new",
                "base_patches": [],
                "base_patch_hashes": {},
            }
        )
    )
    monkeypatch.setattr(
        activation, "identity", lambda p: (p / "marker.py").read_text().strip()
    )
    return project, package, manifest, build, cache


def test_waits_for_writer(setup):
    project, package, manifest, build, cache = setup
    with pytest.raises(RuntimeError, match="has not exited"):
        activation.activate(project, package, manifest, after_build=build)
    assert (package / "marker.py").read_text() == "old\n"


def test_install_preserves_newly_published_cache_and_is_idempotent(setup):
    project, package, manifest, build, cache = setup
    cache.write_text("published cache v2\n")
    (build / "job_exit.json").write_text('{"exit_code":0}')
    result = activation.activate(project, package, manifest, after_build=build)
    assert result["installed"]
    json.dumps(result)
    assert cache.read_text() == "published cache v2\n"
    assert activation.activate(project, package, manifest)["already_active"]
    assert (project / "scripts/apply_engine_audit_patches.py").read_bytes() == (
        project / "libs/team-gm/scripts/apply_engine_audit_patches.py"
    ).read_bytes()


def test_changed_source_rejected(setup):
    project, package, manifest, build, cache = setup
    (package / "marker.py").write_text("concurrent source edit\n")
    with pytest.raises(RuntimeError, match="changed file"):
        activation.activate(project, package, manifest)
    assert (package / "marker.py").read_text() == "concurrent source edit\n"


def test_dry_run_does_not_mutate(setup):
    project, package, manifest, build, cache = setup
    before = (project / "scripts/apply_engine_audit_patches.py").read_bytes()
    assert activation.activate(project, package, manifest, check_only=True)["check_only"]
    assert (package / "marker.py").read_text() == "old\n"
    assert (project / "scripts/apply_engine_audit_patches.py").read_bytes() == before

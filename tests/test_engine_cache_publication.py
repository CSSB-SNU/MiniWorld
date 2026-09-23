"""Nightly cache publication must preserve measurements and concurrent edits."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def build(tmp_path):
    project = tmp_path / "project"
    run = project / "run"
    installed = project / "installed/miniworld_engine"
    source = Path(__file__).resolve().parents[1]
    for directory in (
        project / "scripts",
        project / "patches",
        project / "libs/team-gm/scripts",
        project / "libs/team-gm/patches",
    ):
        directory.mkdir(parents=True)
    installer = (source / "scripts/apply_engine_audit_patches.py").read_text()
    start = installer.index("PATCH_NAMES = (")
    end = installer.index("\n)", start) + 2
    installer = (
        installer[:start] + 'PATCH_NAMES = (\n    "base.patch",\n)' + installer[end:]
    )
    for path in (project / "scripts", project / "libs/team-gm/scripts"):
        (path / "apply_engine_audit_patches.py").write_text(installer)
    shutil.copy2(source / "scripts/publish_engine_build_cache.py", project / "scripts")
    for rel, text in {
        "__init__.py": "",
        "marker.py": "patched\n",
        "autotune/__init__.py": "",
        "autotune/native.py": 'def source_identity(): return "same-source"\n',
        "autotune/cache.py": 'def measurement_mismatch(op, data, identity): return ""\n',
        "autotune/cache_status.py": 'def _current_op_identity(op): return "same-source"\n',
    }.items():
        p = installed / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    gpu = "Test H100 (sm90)"
    relative = Path("example") / (gpu + ".json")
    target = installed / "autotune/data" / relative
    target.parent.mkdir(parents=True)
    target.write_text('{"version": 1}\n')
    snapshot = run / "package/miniworld_engine"
    shutil.copytree(installed, snapshot)
    baseline = run / "baseline_data" / relative
    baseline.parent.mkdir(parents=True)
    shutil.copy2(target, baseline)
    (snapshot / "autotune/data" / relative).write_text('{"version": 2}\n')
    (project / "patches/base.patch").write_text(
        "--- a/src/miniworld_engine/marker.py\n+++ b/src/miniworld_engine/marker.py\n"
        "@@ -1 +1 @@\n-base\n+patched\n"
    )
    metadata = dict(
        project=str(project),
        installed_package=str(installed),
        gpu=gpu,
        patches=["base.patch"],
        native_source_identity="same-source",
        cache_patch="built.patch",
    )
    (run / "metadata.json").write_text(json.dumps(metadata))
    return project, run, installed, snapshot, target


def publish(build, code=0):
    project, run, _, snapshot, _ = build
    return subprocess.run(
        [
            sys.executable,
            str(project / "scripts/publish_engine_build_cache.py"),
            str(run),
            "--build-exit",
            str(code),
        ],
        env={**os.environ, "PYTHONPATH": str(snapshot.parent)},
        cwd=run,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("code,state", [(0, "complete"), (1, "partial")])
def test_cache_patch_publication_and_reinstall(build, code, state):
    result = publish(build, code)
    assert result.returncode == 0, result.stderr
    project, run, installed, _, target = build
    assert json.loads(target.read_text()) == {"version": 2}
    assert json.loads((run / "publication.json").read_text())["state"] == state
    reinstall = subprocess.run(
        [sys.executable, str(project / "scripts/apply_engine_audit_patches.py")],
        env={**os.environ, "PYTHONPATH": str(installed.parent)},
        cwd=run,
        capture_output=True,
        text=True,
    )
    assert reinstall.returncode == 0, reinstall.stderr
    assert "Updated 0 engine files" in reinstall.stdout
    assert (project / "scripts/apply_engine_audit_patches.py").read_bytes() == (
        project / "libs/team-gm/scripts/apply_engine_audit_patches.py"
    ).read_bytes()


@pytest.mark.parametrize("edit", ["source", "cache"])
def test_concurrent_edits_stop_publication(build, edit):
    project, run, installed, _, target = build
    if edit == "source":
        (installed / "autotune/native.py").write_text(
            'def source_identity(): return "edited"\n'
        )
    else:
        target.write_text('{"version": "user edit"}\n')
    before = target.read_bytes()
    result = publish(build)
    assert result.returncode != 0
    assert target.read_bytes() == before
    assert not (project / "patches/built.patch").exists()
    assert (
        json.loads((run / "publication.json").read_text())["state"]
        == "publication_failed"
    )


def test_multiple_unterminated_json_files_roundtrip(build):
    project, run, installed, snapshot, target = build
    expected = {}
    for op, old, new in [
        ('example', '{"version": 1}', '{"version": 2}'),
        ('second', '{"version": 1}\n', '{"version": 3}'),
        ('third', None, '{"version": 4}'),
    ]:
        rel = Path(op) / target.name
        for root in (installed / 'autotune/data', run / 'baseline_data'):
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if old is not None:
                path.write_text(old)
        path = snapshot / 'autotune/data' / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new)
        expected[rel] = new.encode()
    result = publish(build)
    assert result.returncode == 0, result.stderr
    for rel, content in expected.items():
        assert (installed / 'autotune/data' / rel).read_bytes() == content
    result = subprocess.run(
        [sys.executable, str(project / 'scripts/apply_engine_audit_patches.py')],
        env={**os.environ, 'PYTHONPATH': str(installed.parent)}, cwd=run,
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'Updated 0 engine files' in result.stdout

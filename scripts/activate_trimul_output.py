"""Activate the validated TriMul patch after the existing cache writer exits."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def identity(package):
    return (
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                "from miniworld_engine.autotune.native import source_identity; print(source_identity())",
            ],
            env={**os.environ, "PYTHONPATH": str(package.parent)},
            text=True,
        )
        .strip()
        .splitlines()[-1]
    )


def names_node(text):
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PATCH_NAMES" for t in node.targets
        ):
            return tuple(ast.literal_eval(node.value)), node
    raise ValueError("PATCH_NAMES missing")


def activate(project, package, manifest_path, *, after_build=None, check_only=False):
    manifest = json.loads(manifest_path.read_text())
    if after_build is not None and not (after_build / "job_exit.json").exists():
        raise RuntimeError("cache build has not exited; preserve its publication guards")
    installers = [
        project / "scripts/apply_engine_audit_patches.py",
        project / "libs/team-gm/scripts/apply_engine_audit_patches.py",
    ]
    texts = [p.read_text() for p in installers]
    if texts[0] != texts[1]:
        raise RuntimeError("root and team-gm installer stacks differ")
    names, node = names_node(texts[0])
    patch = manifest["patch"]
    for directory in [project / "patches", project / "libs/team-gm/patches"]:
        if sha(directory / patch) != manifest["patch_sha256"]:
            raise RuntimeError("patch changed after validation")
    already = patch in names
    expected = "after" if already else "before"
    for relative, hashes in manifest["files"].items():
        if sha(package / relative) != hashes[expected]:
            raise RuntimeError(f"changed file: {relative}")
    if identity(package) != manifest[expected + "_identity"]:
        raise RuntimeError("engine implementation changed since validation")
    if already:
        return {"installed": True, "already_active": True}
    prefix = tuple(manifest["base_patches"])
    if names[: len(prefix)] != prefix:
        raise RuntimeError("baseline patch order changed")
    for name, digest in manifest["base_patch_hashes"].items():
        if sha(project / "patches" / name) != digest:
            raise RuntimeError(f"baseline patch changed: {name}")
    spec = importlib.util.spec_from_file_location("trimul_patch_applier", installers[0])
    applier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(applier)
    new_names = (*names, patch)
    changes = applier.apply_patches(
        package, project / "patches", new_names, check_only=True
    )
    changes = [str(path) for path in changes]
    if check_only:
        return {"installed": False, "check_only": True, "files": changes}
    if any(p.read_text() != t for p, t in zip(installers, texts)):
        raise RuntimeError("installer changed concurrently")
    applier.apply_patches(package, project / "patches", new_names)
    lines = texts[0].splitlines(keepends=True)
    assignment = (
        "PATCH_NAMES = (\n" + "".join(f'    "{name}",\n' for name in new_names) + ")\n"
    )
    updated = (
        "".join(lines[: node.lineno - 1])
        + assignment
        + "".join(lines[node.end_lineno :])
    )
    for p in installers:
        temp = p.with_suffix(".tmp")
        temp.write_text(updated)
        os.replace(temp, p)
    if identity(package) != manifest["after_identity"]:
        raise RuntimeError("installed source identity differs")
    assert applier.apply_patches(package, project / "patches", new_names) == [], (
        "reinstallation is not idempotent"
    )
    return {
        "installed": True,
        "patch": patch,
        "native_identity": manifest["after_identity"],
        "files": changes,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--package", type=Path)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--after-build", type=Path)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    if a.package is None:
        import miniworld_engine

        a.package = Path(miniworld_engine.__file__).parent
    try:
        result = activate(
            a.project,
            a.package,
            a.manifest,
            after_build=a.after_build,
            check_only=a.check_only,
        )
    except Exception as exc:
        a.report.write_text(
            json.dumps({"installed": False, "error": str(exc)}, indent=2) + "\n"
        )
        raise
    a.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

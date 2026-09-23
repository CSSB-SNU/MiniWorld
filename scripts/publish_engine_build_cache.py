"""Publish an isolated build's valid H100 caches as a reproducible engine patch.

The build snapshot and original installation must still have identical sources.
Concurrent edits to the original cache or patch stack stop publication; measured
shards and merged snapshot caches remain available for recovery.
"""

import argparse
import difflib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from apply_engine_audit_patches import PATCH_NAMES, apply_patches, unified_file_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--build-exit", type=int, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    metadata = json.loads((run / "metadata.json").read_text())
    project = Path(metadata["project"])
    installed = Path(metadata["installed_package"])
    snapshot = run / "package/miniworld_engine"
    report = {"build_exit": args.build_exit, "published": False}
    try:
        from miniworld_engine.autotune import cache, cache_status, native

        expected = metadata["native_source_identity"]
        assert native.source_identity() == expected, "build snapshot source changed"
        env = {**os.environ, "PYTHONPATH": str(installed.parent)}
        current = (
            subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    "from miniworld_engine.autotune.native import source_identity; print(source_identity())",
                ],
                cwd=run,
                env=env,
                text=True,
            )
            .strip()
            .splitlines()[-1]
        )
        assert current == expected, "installed source changed; keep results in snapshot"
        assert list(PATCH_NAMES) == metadata["patches"], (
            "patch stack changed during build"
        )
        changes, rejected = [], []
        for path in sorted(
            (snapshot / "autotune/data").glob("*/" + metadata["gpu"] + ".json")
        ):
            relative = path.relative_to(snapshot / "autotune/data")
            baseline = run / "baseline_data" / relative
            old = baseline.read_bytes() if baseline.exists() else None
            new = path.read_bytes()
            if old == new:
                continue
            data = json.loads(new)
            identity = cache_status._current_op_identity(relative.parts[0])
            reason = (
                cache.measurement_mismatch(relative.parts[0], data, identity)
                if identity
                else "cannot resolve live kernel identity"
            )
            if reason:
                rejected.append({"op": relative.parts[0], "reason": reason})
                continue
            target = installed / "autotune/data" / relative
            assert (target.read_bytes() if target.exists() else None) == old, (
                f"installed cache changed during build: {relative}"
            )
            changes.append((relative, old, new))
        report.update(
            files=len(changes), rejected=rejected, native_source_identity=expected
        )
        if changes:
            name = metadata["cache_patch"]
            patch = []
            for relative, old, new in changes:
                suffix = "src/miniworld_engine/autotune/data/" + relative.as_posix()
                patch.append(unified_file_diff(
                    old.decode() if old is not None else None, new.decode(), suffix))
            destination = project / "patches" / name
            assert not destination.exists(), f"refuse to replace existing {destination}"
            destination.write_text("".join(patch))
            shutil.copy2(destination, project / "libs/team-gm/patches" / name)
            names = (*PATCH_NAMES, name)
            apply_patches(installed, project / "patches", names=names)
            assert apply_patches(installed, project / "patches", names=names) == []
            for relative, _, new in changes:
                target = installed / "autotune/data" / relative
                assert target.read_bytes() == new, f"published bytes differ: {relative}"
                json.loads(target.read_bytes())
            for script in (
                project / "scripts/apply_engine_audit_patches.py",
                project / "libs/team-gm/scripts/apply_engine_audit_patches.py",
            ):
                text = script.read_text()
                anchor = f'    "{PATCH_NAMES[-1]}",\n'
                assert text.count(anchor) == 1, f"cannot update patch stack: {script}"
                script.write_text(text.replace(anchor, anchor + f'    "{name}",\n'))
            report["patch"] = str(destination)
        report["published"] = True
        report["state"] = (
            "complete" if args.build_exit == 0 and not rejected else "partial"
        )
    except Exception as exc:
        report.update(state="publication_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (run / "publication.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

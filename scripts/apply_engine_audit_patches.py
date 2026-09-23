"""Reapply the reviewed local engine fixes after reinstalling the pinned package.

Validate the complete patch stack in a temporary tree before editing. Dependent
patches can overlap; partially and fully patched installations are both supported.
"""

from __future__ import annotations

import importlib.util
import difflib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

PATCH_NAMES = (
    'miniworld-engine-wheel-cute-path.patch',
    'miniworld-engine-cudagraph-amp.patch',
    'miniworld-engine-complete-wiring.patch',
    'miniworld-engine-trimul-release-20260917.patch',
)


def unified_file_diff(old, new, relative):
    """A GNU-patch compatible diff, including paths with spaces and unterminated lines."""
    lines = difflib.unified_diff(
        old.splitlines(keepends=True) if old is not None else [],
        new.splitlines(keepends=True),
        fromfile="a/" + relative + "\t" if old is not None else "/dev/null",
        tofile="b/" + relative + "\t",
    )
    return "".join(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
                   for line in lines)


def apply_patches(package, patches, names=PATCH_NAMES, *, check_only=False):
    package, patches = Path(package), Path(patches).resolve()
    affected = set()
    for name in names:
        for line in (patches / name).read_text().splitlines():
            if line.startswith(("--- ", "+++ ")):
                filename = line[4:].split("\t", 1)[0]
                if filename == "/dev/null":
                    continue
                parts = Path(filename).parts
                if (
                    parts[:3]
                    not in (
                        ("a", "src", "miniworld_engine"),
                        ("b", "src", "miniworld_engine"),
                    )
                    or ".." in parts
                ):
                    raise ValueError(f"Unexpected patch path: {filename}")
                affected.add(Path(*parts[3:]))
    original = {
        p: (package / p).read_bytes() if (package / p).exists() else None
        for p in affected
    }

    def patch(stage, name, reverse=False, dry_run=False):
        command = [
            "patch",
            "--batch",
            "--force",
            "--fuzz=0",
            "--no-backup-if-mismatch",
            "-p3",
            "-d",
            str(stage),
            "-i",
            str(patches / name),
            "--reverse" if reverse else "--forward",
        ]
        if dry_run:
            command.append("--dry-run")
        return subprocess.run(command, capture_output=True, text=True)

    with tempfile.TemporaryDirectory(prefix="miniworld-patches-") as directory:
        stage = Path(directory)
        for path, content in original.items():
            if content is not None:
                (stage / path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(package / path, stage / path)
        # Undo already present patches newest first, then replay the entire stack.
        # A not-yet-applicable dependent patch is resolved during the forward pass.
        for name in reversed(names):
            if patch(stage, name, reverse=True, dry_run=True).returncode == 0:
                result = patch(stage, name, reverse=True)
                if result.returncode:
                    raise RuntimeError(result.stdout + result.stderr)
        for name in names:
            result = patch(stage, name)
            if result.returncode:
                raise RuntimeError(
                    f"Patch does not match {package}: {name}\n"
                    f"{result.stdout}\n{result.stderr}"
                )
        changed = [
            p
            for p in affected
            if original[p]
            != ((stage / p).read_bytes() if (stage / p).exists() else None)
        ]
        if check_only:
            return changed
        # Refuse to overwrite edits made while validation was in progress.
        for path, content in original.items():
            current = (
                (package / path).read_bytes() if (package / path).exists() else None
            )
            if current != content:
                raise RuntimeError(
                    f"Package changed during validation: {package / path}"
                )
        for path in sorted(changed):
            target, source = package / path, stage / path
            if not source.exists():
                target.unlink()
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temp = tempfile.mkstemp(prefix=".engine-patch-", dir=target.parent)
            os.close(fd)
            try:
                shutil.copy2(source, temp)
                os.replace(temp, target)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
        return changed


def main():
    spec = importlib.util.find_spec("miniworld_engine")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "miniworld_engine is not installed in this Python environment"
        )
    package = Path(spec.origin).parent
    patches = Path(__file__).resolve().parents[1] / "patches"
    changed = apply_patches(package, patches)
    print(f"Updated {len(changed)} engine files")
    print(f"Engine fixes ready: {package}")


if __name__ == "__main__":
    main()

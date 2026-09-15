"""Reapply the reviewed local engine fixes after reinstalling the pinned package.

Checks every patch before editing. A changed upstream source that no longer
matches raises an error; it is never replaced with a different checkout's file.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


def main():
    spec = importlib.util.find_spec("miniworld_engine")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "miniworld_engine is not installed in this Python environment"
        )
    package = Path(spec.origin).parent
    patches = Path(__file__).resolve().parents[1] / "patches"
    pending = []
    for name in (
        "miniworld-engine-wheel-cute-path.patch",
        "miniworld-engine-cudagraph-amp.patch",
        "miniworld-engine-complete-wiring.patch",
    ):
        command = [
            "patch",
            "--batch",
            "-p3",
            "-d",
            str(package),
            "-i",
            str(patches / name),
        ]
        forward = subprocess.run(
            [*command, "--dry-run", "--forward"], capture_output=True, text=True
        )
        if forward.returncode == 0:
            pending.append(command)
            continue
        reverse = subprocess.run(
            [*command, "--dry-run", "--reverse"], capture_output=True, text=True
        )
        if reverse.returncode == 0:
            print(f"Already applied: {name}")
            continue
        raise RuntimeError(
            f"Patch does not match {package}: {name}\n{forward.stdout}\n{forward.stderr}"
        )
    for command in pending:
        subprocess.run([*command, "--forward", "--no-backup-if-mismatch"], check=True)
    print(f"Engine fixes ready: {package}")


if __name__ == "__main__":
    main()

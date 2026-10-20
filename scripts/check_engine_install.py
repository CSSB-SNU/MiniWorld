"""Check the consolidated engine installation without applying legacy patches."""

import json
import re
import subprocess
from importlib import metadata, util
from pathlib import Path

#: The GPU architectures production runs on, each needing its own prebuilt kernels (scripts/prebuild_engine.py).
PREBUILD_ARCHES = {"sm90": "H100", "sm100": "B200"}


def check_installation(package, version, direct_url, expected_commit):
    if int(version.split(".", 1)[0]) < 2:
        raise RuntimeError(
            f"engine {version} predates the consolidated main; run pixi install "
            "on a compute/development machine to install the pinned engine"
        )
    actual = direct_url.get("vcs_info", {}).get("commit_id")
    if actual is not None and actual != expected_commit:
        raise RuntimeError(
            f"engine revision mismatch: {actual}; expected {expected_commit}"
        )
    required = (
        "ops/__init__.py",
        "kernels/transition/cuda/fused_sm90a.py",
        "kernels/layernorm/dispatch.py",
        "kernels/triangle_attention/cuda/ln_backward.py",
    )
    missing = [name for name in required if not (package / name).is_file()]
    if missing:
        raise RuntimeError(f"incomplete engine installation: {missing}")


def _git_head(package: Path) -> str | None:
    """HEAD of the git checkout an editable engine (the B200 boxes) is installed from, ``-dirty`` with local edits."""
    for parent in [package, *package.parents][:4]:
        if (parent / ".git").exists():
            try:
                head = subprocess.run(
                    ["git", "-C", str(parent), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=20,
                    check=True,
                ).stdout.strip()
                dirty = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(parent),
                        "status",
                        "--porcelain",
                        "--untracked-files=no",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=True,
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                return None
            return head + ("-dirty" if dirty else "")
    return None


def engine_identity(package: Path, version: str, direct_url: dict) -> str:
    """The engine revision the installed kernels come from: the recorded Git pin, else the source checkout's HEAD."""
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    return commit or _git_head(package) or f"version-{version}"


def installed_engine() -> tuple[str, str, Path]:
    """(version, revision identity, package directory) of the importable miniworld_engine."""
    dist = metadata.distribution("miniworld-engine")
    direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
    spec = util.find_spec("miniworld_engine")
    if spec is None or spec.origin is None:
        raise RuntimeError("miniworld_engine cannot be imported")
    package = Path(spec.origin).parent
    return dist.version, engine_identity(package, dist.version, direct_url), package


def prebuild_status(
    jit_root: Path, identity: str, arches=PREBUILD_ARCHES
) -> dict[str, str]:
    """Per architecture: ``ok``, ``missing`` or ``stale (<revision>)`` -- whether scripts/prebuild_engine.py has built and verified
    this engine revision's kernels in ``jit_root`` (its ``prebuild/<arch>.json`` stamp)."""
    status = {}
    for arch in arches:
        path = Path(jit_root) / "prebuild" / f"{arch}.json"
        try:
            stamp = json.loads(path.read_text())
        except (OSError, ValueError):
            status[arch] = "missing"
            continue
        built = stamp.get("engine")
        status[arch] = (
            "ok"
            if built == identity and stamp.get("verify") == "PASS"
            else f"stale ({built})"
        )
    return status


def shipped_arches(package: Path) -> set[str]:
    """Architectures whose kernels ship compiled inside the engine wheel (``_prebuilt/<arch>-cu<ver>/manifest.json``)."""
    return {
        path.parent.name.split("-", 1)[0]
        for path in (package / "_prebuilt").glob("*/manifest.json")
    }


def report_prebuild(identity: str, package: Path) -> None:
    try:
        from miniworld import engine_jit_root
    except ImportError:
        return
    root = engine_jit_root()
    if root is None:
        return
    status = prebuild_status(root, identity)
    for arch in shipped_arches(package) & status.keys():
        status[arch] = "ok (shipped in the engine wheel)"
    print(f"Engine JIT cache: {root}")
    for arch, state in status.items():
        print(f"  prebuilt kernels for {arch} ({PREBUILD_ARCHES[arch]}): {state}")
    todo = [
        f"{arch} ({PREBUILD_ARCHES[arch]})"
        for arch, state in status.items()
        if not state.startswith("ok")
    ]
    if todo:
        print(
            "HINT: the engine kernels are not prebuilt for this engine revision on "
            + ", ".join(todo)
            + ". The engine wheel does not ship them, so without it the first model call compiles them (14-35 min). Run once on a "
            "GPU node of that architecture: `python scripts/prebuild_engine.py --jobs 8` (or `pixi run -e cu128 engine-prebuild`)."
        )


def main():
    root = Path(__file__).resolve().parents[1]
    project = (root / "pyproject.toml").read_text()
    # The release pin (`miniworld-engine = "==2.3.0"`, from PyPI) or a Git pin (`{ git = ..., rev = "<sha>" }`).
    pinned_version = re.search(r'^miniworld-engine = "==([^"]+)"', project, re.M)
    pinned_rev = re.search(
        r'^miniworld-engine = .*rev = "([0-9a-f]{40})"', project, re.M
    )
    expected = pinned_rev.group(1) if pinned_rev else None
    dist = metadata.distribution("miniworld-engine")
    direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
    spec = util.find_spec("miniworld_engine")
    if spec is None or spec.origin is None:
        raise RuntimeError("miniworld_engine cannot be imported")
    package = Path(spec.origin).parent
    check_installation(package, dist.version, direct_url, expected)
    if pinned_version and not direct_url and dist.version != pinned_version.group(1):
        raise RuntimeError(
            f"engine {dist.version} installed; pyproject.toml pins {pinned_version.group(1)}"
        )
    print(f"Engine {dist.version}: {package}")
    if pinned_version:
        print(f"Pinned release: {pinned_version.group(1)} (PyPI)")
    else:
        print(f"Pinned main revision: {expected}")
        if not direct_url.get("vcs_info"):
            print(
                "Local/wheel install: VCS revision is not recorded in distribution metadata."
            )
    print(
        "No legacy patches applied. GPU dispatch and native rebuilds require separate validation."
    )
    report_prebuild(engine_identity(package, dist.version, direct_url), package)


if __name__ == "__main__":
    main()

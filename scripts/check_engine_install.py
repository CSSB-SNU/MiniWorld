"""Check the consolidated engine installation without applying legacy patches."""

from importlib import metadata, util
import json
from pathlib import Path
import re


def check_installation(package, version, direct_url, expected_commit):
    if int(version.split(".", 1)[0]) < 2:
        raise RuntimeError(
            f"engine {version} predates the consolidated main; run pixi install "
            "on a compute/development machine to install the pinned engine"
        )
    actual = direct_url.get("vcs_info", {}).get("commit_id")
    if actual is not None and actual != expected_commit:
        raise RuntimeError(f"engine revision mismatch: {actual}; expected {expected_commit}")
    required = (
        "ops/__init__.py",
        "kernels/transition/cuda/fused_sm90a.py",
        "kernels/layernorm/dispatch.py",
        "kernels/triangle_attention/cuda/ln_backward.py",
    )
    missing = [name for name in required if not (package / name).is_file()]
    if missing:
        raise RuntimeError(f"incomplete engine installation: {missing}")


def main():
    root = Path(__file__).resolve().parents[1]
    project = (root / "pyproject.toml").read_text()
    expected = re.search(r'^miniworld-engine = .*rev = "([0-9a-f]{40})"', project, re.M).group(1)
    dist = metadata.distribution("miniworld-engine")
    direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
    spec = util.find_spec("miniworld_engine")
    if spec is None or spec.origin is None:
        raise RuntimeError("miniworld_engine cannot be imported")
    package = Path(spec.origin).parent
    check_installation(package, dist.version, direct_url, expected)
    print(f"Engine {dist.version}: {package}")
    print(f"Pinned main revision: {expected}")
    if not direct_url.get("vcs_info"):
        print("Local/wheel install: VCS revision is not recorded in distribution metadata.")
    print("No legacy patches applied. GPU dispatch and native rebuilds require separate validation.")


if __name__ == "__main__":
    main()

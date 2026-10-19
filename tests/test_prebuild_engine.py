"""The engine prebuild covers the inference padding ladder, and engine-setup reads its stamp."""

import ast
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prebuild = _load("prebuild_engine")
checker = _load("check_engine_install")


def _ladder(name):
    tree = ast.parse((ROOT / "scripts/run_miniworld_diffusion_inference.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(name)


def test_ladders_match_the_inference_script():
    assert _ladder("_TOKEN_BUCKETS") == prebuild.TOKEN_BUCKETS
    assert _ladder("_ATOM_BUCKETS") == prebuild.ATOM_BUCKETS
    assert _ladder("_MSA_BUCKETS") == (prebuild.INFER_MSA,)
    assert _ladder("_N_TEMPLATE") == prebuild.INFER_TEMPLATES


def test_default_shapes_cover_every_rung():
    shapes = prebuild.default_infer_shapes()
    assert {t for t, _ in shapes} == set(prebuild.TOKEN_BUCKETS)
    assert {a for _, a in shapes} == set(prebuild.ATOM_BUCKETS)
    # the pairs FoldBench targets pad to
    for pair in [
        (128, 1024),
        (256, 2048),
        (384, 4096),
        (512, 4096),
        (768, 6144),
        (1024, 8192),
        (1536, 12288),
        (2560, 24576),
    ]:
        assert pair in shapes


def test_stamp_status(tmp_path):
    assert checker.prebuild_status(tmp_path, "a" * 40) == {
        "sm90": "missing",
        "sm100": "missing",
    }
    (tmp_path / "prebuild").mkdir()
    (tmp_path / "prebuild" / "sm90.json").write_text(
        json.dumps({"engine": "a" * 40, "verify": "PASS"})
    )
    (tmp_path / "prebuild" / "sm100.json").write_text(
        json.dumps({"engine": "b" * 40, "verify": "PASS"})
    )
    status = checker.prebuild_status(tmp_path, "a" * 40)
    assert status["sm90"] == "ok"
    assert status["sm100"].startswith("stale")


def test_engine_identity_prefers_the_recorded_pin(tmp_path):
    assert (
        checker.engine_identity(tmp_path, "2.2.0", {"vcs_info": {"commit_id": "c" * 40}})
        == "c" * 40
    )
    package = (
        tmp_path / "a" / "b" / "c" / "d" / "miniworld_engine"
    )  # no git checkout within reach
    package.mkdir(parents=True)
    assert checker.engine_identity(package, "2.2.0", {}) == "version-2.2.0"

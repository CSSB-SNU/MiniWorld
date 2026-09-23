"""Triton-only builds preserve the module plan and omit native driver searches."""
import importlib.util
from pathlib import Path
import pytest


@pytest.mark.parametrize('selected',[False,True])
def test_triton_driver_scope(monkeypatch,selected):
    from miniworld_engine import cli
    from miniworld_engine.autotune import derive,native
    filename=Path(__file__).resolve().parents[1]/'scripts/build_engine_triton_cache.py'
    spec=importlib.util.spec_from_file_location('triton_builder',filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=lambda *args,**kwargs: {'adaln_gemm_gate_triton','rmsnorm_fwd_triton','layernorm_fwd_cuda'}
    monkeypatch.setattr(derive,'uncovered_kernels',original)
    assert 'layernorm_fwd_cuda' in native.BUILD_OPS
    def invoke(argv):
        assert argv==['build','all','--gpus','7']
        expected={'adaln_gemm_gate_triton'} if selected else {'adaln_gemm_gate_triton','rmsnorm_fwd_triton'}
        assert derive.uncovered_kernels('sm90')==expected
        return 7
    monkeypatch.setattr(cli,'main',invoke)
    flags=['--driver-ops','adaln_gemm_gate_triton'] if selected else []
    assert module.main([*flags,'--gpus','7'])==7
    assert derive.uncovered_kernels is original

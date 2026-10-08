"""``train.forbid_triton``: engine / user Triton kernels are refused (launched or traced into a compiled graph), inductor's own kernels are not."""
import pytest
import torch

from miniworld.training import no_triton


def test_install_forbids_engine_triton_kernels_but_not_inductor():
    triton = pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("a Triton launch needs a GPU")
    import triton.language as tl

    @triton.jit
    def double(x_ptr, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(x_ptr + i, tl.load(x_ptr + i, mask=i < n) * 2, mask=i < n)

    def launch(t):
        double[(1,)](t, 8, BLOCK=8)
        return t

    x = torch.ones(8, device="cuda")
    launch(x)                                                   # works before the guard
    assert x.tolist() == [2.0] * 8
    no_triton.install()
    try:
        assert no_triton.active()
        with pytest.raises(no_triton.TritonForbidden, match="double"):                    # an eager launch
            launch(x)
        with pytest.raises(Exception, match="TritonForbidden: Triton kernel 'double' traced"):    # inside torch.compile
            torch.compile(launch, backend="inductor")(x)
        y = torch.compile(lambda t: t * 3 + 1)(torch.ones(8, device="cuda"))               # inductor's own kernels stay allowed
        assert y.tolist() == [4.0] * 8
    finally:
        no_triton.uninstall()
    assert not no_triton.active()
    launch(x)                                                   # and the kernel works again after it
    assert x.tolist() == [4.0] * 8

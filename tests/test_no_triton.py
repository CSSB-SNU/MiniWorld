"""``train.forbid_triton``: the guard refuses Triton launches and inductor compiles, init_msa runs eagerly under it, the v2 phase 1 configs turn it on."""
import pytest
import torch

from miniworld.data.features import MSAFeatures
from miniworld.modules.msa_util import _init_msa, init_msa
from miniworld.training import no_triton


def tiny_msa(device="cpu"):
    b, n, l = 1, 3, 4
    return MSAFeatures(
        aligned_sequences=torch.randint(0, 32, (b, n, l), device=device),
        mask=torch.tensor([[True, True, False]], device=device),
        has_deletion=torch.randint(0, 2, (b, n, l), device=device),
        deletion_value=torch.rand(b, n, l, device=device),
        profile=torch.rand(b, l, 32, device=device),
        deletion_mean=torch.rand(b, l, device=device),
    )


def test_init_msa_is_eager_under_the_guard_and_matches_the_one_hot_definition(monkeypatch):
    msa = tiny_msa()
    monkeypatch.setattr(no_triton, "_ACTIVE", True)           # what install() sets, without patching the process
    feat, mask = init_msa(msa, num_res_class=32, dtype=torch.float32)
    expected = torch.cat([torch.nn.functional.one_hot(msa.aligned_sequences.long(), 32), msa.has_deletion[..., None],
                          msa.deletion_value[..., None]], dim=-1).float() * msa.mask[:, :, None, None]
    torch.testing.assert_close(feat, expected)
    torch.testing.assert_close(feat, _init_msa(msa, 32, torch.float32)[0])
    assert mask.tolist() == msa.mask.tolist()


def test_install_forbids_triton_launches_and_inductor_compiles():
    triton = pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("a Triton launch needs a GPU")
    import triton.language as tl

    @triton.jit
    def double(x_ptr, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(x_ptr + i, tl.load(x_ptr + i, mask=i < n) * 2, mask=i < n)

    x = torch.ones(8, device="cuda")
    double[(1,)](x, 8, BLOCK=8)                                # works before the guard
    assert x.tolist() == [2.0] * 8
    no_triton.install()
    try:
        assert no_triton.active()
        with pytest.raises(no_triton.TritonForbidden, match="double"):
            double[(1,)](x, 8, BLOCK=8)
        with pytest.raises(Exception, match="TritonForbidden: inductor compile"):    # dynamo wraps it in BackendCompilerFailed
            torch.compile(lambda t: t * 2)(x)
    finally:
        no_triton.uninstall()
    assert not no_triton.active()
    double[(1,)](x, 8, BLOCK=8)                                # and again after it
    assert x.tolist() == [4.0] * 8

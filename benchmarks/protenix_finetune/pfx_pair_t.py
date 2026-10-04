"""Fast pair transpose [B, I, J, C] -> [B, J, I, C] (contiguous), for the engine TriAttn's ending direction.

The engine wraps the ending node as rearrange(pair, "B I J D -> B J I D").contiguous() around the starting-layout kernels (input,
output, and the same two in backward); torch's generic permuted copy runs at ~1.9 TB/s on B200. Each (b, i, j) row is C
contiguous elements (256 B for C = 128 bf16), so the transpose only moves whole rows: one thread per 16 B, reads and writes
both fully coalesced within a row. Installed by patching the einops rearrange the integration module imported."""
import os
import torch
from torch.utils.cpp_extension import load_inline


_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void __launch_bounds__(256) pair_t_kernel(const int4* __restrict__ x, int4* __restrict__ y, long long n_vec, int L, int vpr) {
    const long long LL = (long long)L * L;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < n_vec; t += (long long)gridDim.x * blockDim.x) {
        const long long row = t / vpr;
        const int v = (int)(t - row * vpr);
        const long long b = row / LL;
        const long long r = row - b * LL;
        const long long i = r / L, j = r - i * L;
        y[(b * LL + j * L + i) * vpr + v] = __ldg(x + t);
    }
}

void pair_t(at::Tensor x, at::Tensor y) {
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && x.dim() == 4 && x.size(1) == x.size(2));
    const int L = x.size(1);
    const long long row_bytes = x.size(3) * x.element_size();
    TORCH_CHECK(row_bytes % 16 == 0);
    const int vpr = row_bytes / 16;
    const long long n_vec = x.numel() * x.element_size() / 16;
    const int blocks = (int)std::min<long long>((n_vec + 255) / 256, 148LL * 16);
    pair_t_kernel<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const int4*>(x.data_ptr()), reinterpret_cast<int4*>(y.data_ptr()), n_vec, L, vpr);
}
"""
_mod = None


def _ext():
    global _mod
    if _mod is None:
        _mod = load_inline("pfx_pair_t", cpp_sources="void pair_t(at::Tensor x, at::Tensor y);", cuda_sources=_SRC,
                           functions=["pair_t"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a"], verbose=False)
    return _mod


def _t(x):
    x = x.contiguous()
    y = torch.empty_like(x)
    _ext().pair_t(x, y)
    return y


class _PairT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return _t(x)

    @staticmethod
    def backward(ctx, g):
        return _t(g)


def ok(t):
    return t.is_cuda and t.dim() == 4 and t.shape[1] == t.shape[2] and (t.shape[-1] * t.element_size()) % 16 == 0


def pair_transpose(t):
    return _PairT.apply(t)


def install():
    """swap the rearrange used by the engine's TriAttn B200 integration (ending direction) for the fast transpose"""
    import einops
    import miniworld_engine.integrations.triattn_b200 as TB
    orig = einops.rearrange

    def fast_rearrange(t, pattern, **kw):
        if not kw and pattern in ("B I J D -> B J I D", "B J I D -> B I J D") and ok(t):
            return _PairT.apply(t)
        return orig(t, pattern, **kw)
    TB.rearrange = fast_rearrange
    return TB

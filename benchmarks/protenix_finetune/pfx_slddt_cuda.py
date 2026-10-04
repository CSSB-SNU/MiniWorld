"""Sparse smooth-LDDT (Protenix SmoothLDDTLoss.sparse_forward) as one CUDA kernel: forward value and gradient together.

The pair list (l, m) is held as CSR rows (nonzero order). One warp per (sample, atom l) walks row l: the per-pair smooth-LDDT
term e(|d_lm - d_lm^true|) = 0.25 * sum_t sigmoid(t - .) is summed per row, and d(sum_pairs e)/dx_l is accumulated in
registers. Pair (l, m) also moves x_m; those contributions come from the transposed list (row l of the transpose = the
pairs (m, l)), so every atom's gradient is a row sum -- no atomics. When the list is symmetric (protein, DNA-free lists)
the transpose is the list itself and its row contribution equals the forward one (factor 2). The backward is a scale."""
import os
import torch
from torch.utils.cpp_extension import load_inline


_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ void term(float diff, float& e, float& de) {
    // e = 0.25 * sum_t sigmoid(t - |diff|), de = de / d(diff)
    const float dd = fabsf(diff);
    float s, ds = 0.f;
    e = 0.f;
    s = 1.f / (1.f + expf(-(0.5f - dd))); e += 0.25f * s; ds += s * (1.f - s);
    s = 1.f / (1.f + expf(-(1.0f - dd))); e += 0.25f * s; ds += s * (1.f - s);
    s = 1.f / (1.f + expf(-(2.0f - dd))); e += 0.25f * s; ds += s * (1.f - s);
    s = 1.f / (1.f + expf(-(4.0f - dd))); e += 0.25f * s; ds += s * (1.f - s);
    const float sg = (diff > 0.f) ? 1.f : ((diff < 0.f) ? -1.f : 0.f);
    de = -0.25f * ds * sg;
}

template <bool SYM>
__global__ void __launch_bounds__(256) slddt_kernel(const float* __restrict__ x, int N,
        const int* __restrict__ rp, const int* __restrict__ ci, const float* __restrict__ td,
        const int* __restrict__ rp2, const int* __restrict__ ci2, const float* __restrict__ td2,
        float* __restrict__ e_row, float* __restrict__ g) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int l = blockIdx.x * 8 + warp;
    if (l >= N) return;
    const float* xs = x + (size_t)blockIdx.y * N * 3;
    const float a0 = xs[3 * l], a1 = xs[3 * l + 1], a2 = xs[3 * l + 2];
    float es = 0.f, g0 = 0.f, g1 = 0.f, g2 = 0.f;
    for (int j = rp[l] + lane; j < rp[l + 1]; j += 32) {
        const int m = ci[j];
        const float d0 = a0 - xs[3 * m], d1 = a1 - xs[3 * m + 1], d2 = a2 - xs[3 * m + 2];
        const float pd = sqrtf(d0 * d0 + d1 * d1 + d2 * d2);
        float e, de;
        term(pd - td[j], e, de);
        es += e;
        const float c = pd > 0.f ? de / pd : 0.f;
        g0 += c * d0; g1 += c * d1; g2 += c * d2;
    }
    if (SYM) {
        g0 *= 2.f; g1 *= 2.f; g2 *= 2.f;
    } else {
        for (int j = rp2[l] + lane; j < rp2[l + 1]; j += 32) {
            const int m = ci2[j];
            const float d0 = a0 - xs[3 * m], d1 = a1 - xs[3 * m + 1], d2 = a2 - xs[3 * m + 2];
            const float pd = sqrtf(d0 * d0 + d1 * d1 + d2 * d2);
            float e, de;
            term(pd - td2[j], e, de);
            const float c = pd > 0.f ? de / pd : 0.f;
            g0 += c * d0; g1 += c * d1; g2 += c * d2;
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        es += __shfl_xor_sync(0xffffffffu, es, o);
        g0 += __shfl_xor_sync(0xffffffffu, g0, o);
        g1 += __shfl_xor_sync(0xffffffffu, g1, o);
        g2 += __shfl_xor_sync(0xffffffffu, g2, o);
    }
    if (lane == 0) {
        const size_t r = (size_t)blockIdx.y * N + l;
        e_row[r] = es;
        g[3 * r] = g0; g[3 * r + 1] = g1; g[3 * r + 2] = g2;
    }
}

std::vector<at::Tensor> slddt(at::Tensor x, at::Tensor rp, at::Tensor ci, at::Tensor td,
                              at::Tensor rp2, at::Tensor ci2, at::Tensor td2, bool sym) {
    TORCH_CHECK(x.dim() == 3 && x.size(2) == 3 && x.is_contiguous() && x.scalar_type() == at::kFloat);
    const int S = x.size(0), N = x.size(1);
    auto e = at::empty({S, N}, x.options());
    auto g = at::empty({S, N, 3}, x.options());
    dim3 grid((N + 7) / 8, S);
    auto st = at::cuda::getCurrentCUDAStream();
    if (sym)
        slddt_kernel<true><<<grid, 256, 0, st>>>(x.data_ptr<float>(), N, rp.data_ptr<int>(), ci.data_ptr<int>(), td.data_ptr<float>(),
            rp2.data_ptr<int>(), ci2.data_ptr<int>(), td2.data_ptr<float>(), e.data_ptr<float>(), g.data_ptr<float>());
    else
        slddt_kernel<false><<<grid, 256, 0, st>>>(x.data_ptr<float>(), N, rp.data_ptr<int>(), ci.data_ptr<int>(), td.data_ptr<float>(),
            rp2.data_ptr<int>(), ci2.data_ptr<int>(), td2.data_ptr<float>(), e.data_ptr<float>(), g.data_ptr<float>());
    return {e, g};
}
"""
_mod = None


def _ext():
    global _mod
    if _mod is None:
        _mod = load_inline("pfx_slddt", cpp_sources="std::vector<at::Tensor> slddt(at::Tensor x, at::Tensor rp, at::Tensor ci, at::Tensor td, "
                           "at::Tensor rp2, at::Tensor ci2, at::Tensor td2, bool sym);",
                           cuda_sources=_SRC, functions=["slddt"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a"],
                           verbose=False)
    return _mod


def build_csr(li, mi, td, n):
    """CSR rows of the pair list (li sorted, as torch.nonzero gives) and of its transpose; sym = the two are identical."""
    rp = torch.zeros(n + 1, dtype=torch.int32, device=li.device)
    rp[1:] = torch.bincount(li, minlength=n).cumsum(0)
    perm = torch.argsort(mi * n + li)
    rp2 = torch.zeros(n + 1, dtype=torch.int32, device=li.device)
    rp2[1:] = torch.bincount(mi, minlength=n).cumsum(0)
    ci, ci2 = mi.int(), li[perm].int()
    sym = bool(torch.equal(rp, rp2) and torch.equal(ci, ci2))
    return dict(rp=rp, ci=ci, td=td.float().contiguous(), rp2=rp2, ci2=ci2, td2=td[perm].float().contiguous(), sym=sym, n_pairs=int(li.numel()))


class _SLDDT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rp, ci, td, rp2, ci2, td2, sym, n_pairs):
        e, g = _ext().slddt(x.contiguous().float(), rp, ci, td, rp2, ci2, td2, sym)
        ctx.save_for_backward(g)
        ctx.n = n_pairs
        return e.sum(-1) / n_pairs              # per-sample mean of the smooth-LDDT term over the pairs

    @staticmethod
    def backward(ctx, go):
        (g,) = ctx.saved_tensors
        return (g * (go / ctx.n)[:, None, None]), None, None, None, None, None, None, None, None


@torch.compiler.disable
def smooth_lddt_loss(x, csr):
    """1 - mean_samples(mean_pairs(e)), x [S, N, 3] fp32"""
    lddt = _SLDDT.apply(x, csr["rp"], csr["ci"], csr["td"], csr["rp2"], csr["ci2"], csr["td2"], csr["sym"], csr["n_pairs"])
    return 1 - torch.mean(torch.mean(lddt, dim=-1))

"""The fast trainer's whole optimizer section in three CUDA launches: global-norm clip, Adam (torch's fused-Adam math, L2 weight
decay), fp32 master -> bf16 weight cast, EMA lerp.

torch's version is ~2 x 4170 multi-tensor launches over 368M parameters (master-grad cast, per-tensor norms, clip scale, fused
Adam, weight cast, EMA). Here every tensor is cut into fixed chunks; (1) per-chunk sum of squared grads, (2) one block: total
norm -> clip coefficient, step count, bias corrections, (3) per chunk: g * coef (+ wd * w) -> Adam moments and the fp32 master
update -> the bf16 weight (if the parameter is bf16) and the EMA shadow. The Adam moments and the EMA shadow are flat fp32
buffers owned here; the grad pointers (they change between eager steps; a graph captures them) go by value in the kernel
parameters, up to 2048 tensors per launch."""
import os
import torch
from torch.utils.cpp_extension import load_inline

CH = 1 << 16

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#define CH (1 << 16)
#define GMAX 2048
struct GPtrs { int64_t p[GMAX]; };       // this launch's grad pointers, by value in the kernel parameters (capturable)

__device__ __forceinline__ float ld_g(int64_t p, int bf, int64_t i) {
    return bf ? __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(p)[i]) : reinterpret_cast<const float*>(p)[i];
}

__global__ void __launch_bounds__(512) sumsq_kernel(const __grid_constant__ GPtrs gp, int t0, const int64_t* __restrict__ lp,
        const int64_t* __restrict__ numel, const int64_t* __restrict__ chunks, float* __restrict__ part) {
    const int64_t t = chunks[2 * blockIdx.x], s0 = chunks[2 * blockIdx.x + 1];
    const int64_t e = min(s0 + CH, numel[t]);
    const int64_t p = gp.p[t - t0]; const int bf = lp[t] != 0;
    float acc = 0.f;
    for (int64_t i = s0 + threadIdx.x; i < e; i += blockDim.x) { const float g = ld_g(p, bf, i); acc += g * g; }
    __shared__ float red[32];
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        acc = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (threadIdx.x == 0) part[blockIdx.x] = acc;
    }
}

// scal: [0] clip coefficient, [1] step, [2] step_size, [3] bias_correction2_sqrt, [4] total grad norm
__global__ void finalize_kernel(const float* __restrict__ part, int n, float* __restrict__ scal, float max_norm, float lr, float b1, float b2) {
    double acc = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += (double)part[i];
    __shared__ double red[32];
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        double s = 0.0;
        for (int w = 0; w < (blockDim.x >> 5); ++w) s += red[w];
        const float total = (float)sqrt(s);
        scal[4] = total;
        scal[0] = fminf(max_norm / (total + 1e-6f), 1.f);
        const float step = scal[1] + 1.f;
        scal[1] = step;
        scal[2] = lr / (1.f - powf(b1, step));
        scal[3] = sqrtf(1.f - powf(b2, step));
    }
}

__global__ void __launch_bounds__(512) update_kernel(const __grid_constant__ GPtrs gp, int t0, const int64_t* __restrict__ wp,
        const int64_t* __restrict__ lp, const int64_t* __restrict__ off, const int64_t* __restrict__ numel,
        const int64_t* __restrict__ chunks, float* __restrict__ M, float* __restrict__ V, float* __restrict__ S,
        const float* __restrict__ scal, float b1, float b2, float eps, float wd, float ema_w) {
    const int64_t t = chunks[2 * blockIdx.x], s0 = chunks[2 * blockIdx.x + 1];
    const int64_t e = min(s0 + CH, numel[t]);
    const int64_t p = gp.p[t - t0];
    float* w = reinterpret_cast<float*>(wp[t]);
    __nv_bfloat16* lw = reinterpret_cast<__nv_bfloat16*>(lp[t]);
    const int bf = lw != nullptr;
    const int64_t o = off[t];
    const float coef = scal[0], step_size = scal[2], bc2s = scal[3];
    for (int64_t i = s0 + threadIdx.x; i < e; i += blockDim.x) {
        float g = ld_g(p, bf, i) * coef;
        float x = w[i];
        if (wd != 0.f) g += wd * x;
        const float m = b1 * M[o + i] + (1.f - b1) * g;
        const float v = b2 * V[o + i] + (1.f - b2) * g * g;
        M[o + i] = m; V[o + i] = v;
        const float denom = sqrtf(v) / bc2s + eps;
        x -= step_size * m / denom;
        w[i] = x;
        if (bf) lw[i] = __float2bfloat16_rn(x);
        const float sh = S[o + i];
        S[o + i] = sh + ema_w * (x - sh);
    }
}

// groups: per group [t0, t1) tensor range, its chunk table and its slice of the partial sums
void fused_opt(std::vector<int64_t> gptrs, std::vector<int64_t> t0s, std::vector<at::Tensor> chunks, std::vector<int64_t> pofs,
               at::Tensor wp, at::Tensor lp, at::Tensor off, at::Tensor numel, at::Tensor M, at::Tensor V, at::Tensor S,
               at::Tensor part, at::Tensor scal, double max_norm, double lr, double b1, double b2, double eps, double wd, double ema_w) {
    auto st = at::cuda::getCurrentCUDAStream();
    const int G = t0s.size();
    std::vector<GPtrs> gp(G);
    for (int k = 0; k < G; ++k) {
        const int64_t a = t0s[k], b = (k + 1 < G) ? t0s[k + 1] : (int64_t)gptrs.size();
        TORCH_CHECK(b - a <= GMAX);
        for (int64_t t = a; t < b; ++t) gp[k].p[t - a] = gptrs[t];
    }
    for (int k = 0; k < G; ++k)
        sumsq_kernel<<<chunks[k].size(0), 512, 0, st>>>(gp[k], (int)t0s[k], lp.data_ptr<int64_t>(), numel.data_ptr<int64_t>(),
                                                        chunks[k].data_ptr<int64_t>(), part.data_ptr<float>() + pofs[k]);
    finalize_kernel<<<1, 1024, 0, st>>>(part.data_ptr<float>(), (int)part.numel(), scal.data_ptr<float>(), (float)max_norm, (float)lr,
                                        (float)b1, (float)b2);
    for (int k = 0; k < G; ++k)
        update_kernel<<<chunks[k].size(0), 512, 0, st>>>(gp[k], (int)t0s[k], wp.data_ptr<int64_t>(), lp.data_ptr<int64_t>(),
                                                         off.data_ptr<int64_t>(), numel.data_ptr<int64_t>(), chunks[k].data_ptr<int64_t>(),
                                                         M.data_ptr<float>(), V.data_ptr<float>(), S.data_ptr<float>(), scal.data_ptr<float>(),
                                                         (float)b1, (float)b2, (float)eps, (float)wd, (float)ema_w);
}
"""
_mod = None


def _ext():
    global _mod
    if _mod is None:
        _mod = load_inline("pfx_fused_opt", cpp_sources="void fused_opt(std::vector<int64_t> gptrs, std::vector<int64_t> t0s, "
                           "std::vector<at::Tensor> chunks, std::vector<int64_t> pofs, at::Tensor wp, at::Tensor lp, at::Tensor off, at::Tensor numel, "
                           "at::Tensor M, at::Tensor V, at::Tensor S, at::Tensor part, at::Tensor scal, "
                           "double max_norm, double lr, double b1, double b2, double eps, double wd, double ema_w);",
                           cuda_sources=_SRC, functions=["fused_opt"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a"],
                           verbose=False)
    return _mod


class FusedTrainerOpt:
    """params: the model's trainable tensors (fp32 or bf16), in order; masters: their fp32 tensors (the parameter itself when fp32).
    The moments start at zero and the EMA shadow at the masters (as torch's Adam / the trainer's EMA)."""

    def __init__(self, params, masters, lr, betas, eps, weight_decay, ema_decay, max_norm):
        dev = masters[0].device
        self.params, self.masters = params, masters
        self.hp = dict(lr=float(lr), b1=float(betas[0]), b2=float(betas[1]), eps=float(eps), wd=float(weight_decay), ema_w=1.0 - float(ema_decay),
                       max_norm=float(max_norm))
        for w in masters:
            assert w.dtype == torch.float32 and w.is_contiguous()
        for p, w in zip(params, masters):
            assert p.is_contiguous() and p.numel() == w.numel() and (p.dtype == torch.float32 and p is w or p.dtype == torch.bfloat16)
        numel = [w.numel() for w in masters]
        offs, o = [], 0
        for n in numel:
            offs.append(o); o += n
        self.n = o
        i64 = lambda xs: torch.tensor(xs, dtype=torch.int64, device=dev)
        self.wp = i64([w.data_ptr() for w in masters])
        self.lp = i64([p.data_ptr() if p.dtype == torch.bfloat16 else 0 for p in params])
        self.off, self.numel = i64(offs), i64(numel)
        GMAX = 2048
        self.t0s, self.chunks, self.pofs = [], [], []
        nc = 0
        for t0 in range(0, len(numel), GMAX):
            ch = [[t, s] for t in range(t0, min(t0 + GMAX, len(numel))) for s in range(0, numel[t], CH)]
            self.t0s.append(t0); self.pofs.append(nc); self.chunks.append(i64(ch).reshape(-1, 2)); nc += len(ch)
        self.M = torch.zeros(o, device=dev); self.V = torch.zeros(o, device=dev)
        self.S = torch.cat([w.detach().reshape(-1) for w in masters])
        self.part = torch.zeros(nc, device=dev)
        self.scal = torch.zeros(8, device=dev)
        self.shadow_views = [self.S[a:a + n].view_as(w) for a, n, w in zip(offs, numel, masters)]

    def state_tensors(self):
        return [self.M, self.V, self.scal]

    def step(self, grads):
        for g, p in zip(grads, self.params):
            assert g.is_contiguous() and g.numel() == p.numel() and g.dtype == p.dtype
        h = self.hp
        _ext().fused_opt([g.data_ptr() for g in grads], self.t0s, self.chunks, self.pofs, self.wp, self.lp, self.off, self.numel,
                         self.M, self.V, self.S, self.part, self.scal, h["max_norm"], h["lr"], h["b1"], h["b2"], h["eps"], h["wd"], h["ema_w"])

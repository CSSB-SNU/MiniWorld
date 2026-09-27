// ESMFold2 SWA atom block, out-projection backward (CUDA sm_90a, C = 128, H = 4, D = 32, bf16 operands, fp32 accumulate).
// Forward: q1 = q + gate_a * rn((sigmoid(g) * o) Wo^T).  Given dq1:
//   datt = rn(dq1 * rn(gate_a));  dgated = datt Wo;  dO = rn(dgated * sg);  dG = rn(dgated * o * sg * (1 - sg));  gated = rn(sg * o)
//   Dv[n, h, s] = sum_{d in head h} dO * o  (fp32);  d gate_a[b*S + s] += sum_aug dq1 * att
// (rounding points of the Triton _oproj_bwd).  datt / gated are the operands of dWo = datt^T gated (cuBLAS on the host).
// Structure: two consumer warpgroups ping-pong on alternate tiles (SP=8 augments x AT=8 atoms), Wo^T resident in smem, one producer warp.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cstdlib>

namespace {
constexpr int C = 128, H = 4, D = 32, BR = 64, SP = 8, AT = BR / SP;
constexpr int THREADS = 384;
constexpr int TB = BR * 128;
constexpr int SDQ = 0, SO = 2 * TB, SG = 4 * TB, SATT = 6 * TB, SMOD = 8 * TB, SRED = SMOD + 4 * AT * 128, SET = SRED + AT * C * 4;
constexpr int SW = 2 * SET, SEND = SW + 32768;
constexpr int NBAR = 5;
constexpr int BYTES = SEND + NBAR * 8;
#ifndef CONS_REGS
#define CONS_REGS "232"
#define PROD_REGS "40"
#endif

namespace wg {
__device__ __forceinline__ uint64_t desc(const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  uint64_t d = static_cast<uint64_t>((a >> 4) & 0x3FFFull);
  d |= (static_cast<uint64_t>(1) << 16);
  d |= (static_cast<uint64_t>(64) << 32);
  d |= (static_cast<uint64_t>(1) << 62);
  return d;
}
#include "gen/wgmma_bf16.inc"
__device__ __forceinline__ void fence()  { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int K> __device__ __forceinline__ void wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(K) : "memory"); }
__device__ __forceinline__ void proxy_fence() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
}  // namespace wg

namespace tma {
__device__ __forceinline__ uint32_t sa(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void bar_init(uint64_t* b, uint32_t count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(sa(b)), "r"(count) : "memory"); }
__device__ __forceinline__ void bar_init_fence() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
__device__ __forceinline__ void expect_tx(uint64_t* b, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(sa(b)), "r"(bytes) : "memory"); }
__device__ __forceinline__ void arrive(uint64_t* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(sa(b)) : "memory"); }
__device__ __forceinline__ void wait(uint64_t* b, uint32_t parity) {
  asm volatile("{\n.reg .pred p;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WAIT_%=;\n}\n" :: "r"(sa(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void load_2d(const void* map, uint32_t dst, uint64_t* bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               :: "r"(dst), "l"(map), "r"(sa(bar)), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void load_3d(const void* map, uint32_t dst, uint64_t* bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];\n"
               :: "r"(dst), "l"(map), "r"(sa(bar)), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void load_4d(const void* map, uint32_t dst, uint64_t* bar, int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];\n"
               :: "r"(dst), "l"(map), "r"(sa(bar)), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}
__device__ __forceinline__ void store_4d(const void* map, const void* src, int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [%0, {%2, %3, %4, %5}], [%1];\n"
               :: "l"(map), "r"(sa(src)), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}
__device__ __forceinline__ void store_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
template <int K> __device__ __forceinline__ void store_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;\n" :: "n"(K) : "memory"); }
}  // namespace tma

__device__ __forceinline__ float rcp_approx(float x) { float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
__device__ __forceinline__ float sigmoid(float x) { return rcp_approx(1.f + __expf(-x)); }
__device__ __forceinline__ uint32_t pack(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float rnb(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float2 unpack(uint32_t u) {
  const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&u);
  return make_float2(__low2float(v), __high2float(v));
}
__device__ __forceinline__ int offb(int r, int c) { return (c >> 6) * TB + r * 128 + ((((c & 63) >> 3) ^ (r & 7)) << 4) + (c & 7) * 2; }
__device__ __forceinline__ const float2* modp(const unsigned char* sm, int al, int c) {   // gate_a tile [4 col-blocks][AT][32] fp32, 128 B swizzle
  const int row = (c >> 5) * AT + al;
  return reinterpret_cast<const float2*>(sm + row * 128 + ((((c & 31) >> 2) ^ (row & 7)) << 4) + (c & 3) * 4);
}
__device__ __forceinline__ void red_add_v4(float* p, float4 v) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" :: "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}

struct Args {
  float *dv, *dmod;                                       // Dv [N, H, S] fp32; dmod [B*S, 6C] fp32 (accumulated, gate_a columns)
  int S, A, B, nab, nag, ntile;
};

__global__ void __launch_bounds__(THREADS, 1) oproj_bwd_kernel(Args args, const __grid_constant__ CUtensorMap mdq, const __grid_constant__ CUtensorMap mo,
                                                              const __grid_constant__ CUtensorMap mg, const __grid_constant__ CUtensorMap matt,
                                                              const __grid_constant__ CUtensorMap mmod, const __grid_constant__ CUtensorMap mwot,
                                                              const __grid_constant__ CUtensorMap mdo, const __grid_constant__ CUtensorMap mdg,
                                                              const __grid_constant__ CUtensorMap mdatt, const __grid_constant__ CUtensorMap mgated) {
  extern __shared__ __align__(1024) unsigned char smem[];
  if (tma::sa(smem) & 1023u) __trap();
  unsigned char* sW = smem + SW;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + SEND);
  uint64_t *xfull = bars, *xempty = bars + 2, *wfull = bars + 4;
  const int tid = threadIdx.x, lane = tid & 31, wgi = tid >> 7, warp = (tid >> 5) & 3, tq = tid & 127;
  if (tid == 0) {
    for (int i = 0; i < 2; ++i) { tma::bar_init(xfull + i, 1); tma::bar_init(xempty + i, 1); }
    tma::bar_init(wfull, 1);
    tma::bar_init_fence(); wg::proxy_fence();
  }
  __syncthreads();
  const int ntT = (int)blockIdx.x < args.ntile ? (args.ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  const int S = args.S, A = args.A, B = args.B;
  auto coords = [&](int t, int& b, int& a0, int& s0) {
    const int ab = t % args.nab, r = t / args.nab, ag = r % args.nag;
    b = r / args.nag; a0 = ag * SP; s0 = ab * AT;
  };
  if (wgi == 2) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 " PROD_REGS ";\n" ::: "memory");
    if (warp == 0 && lane == 0) {
      tma::expect_tx(wfull, 32768);
      tma::load_2d(&mwot, tma::sa(sW), wfull, 0, 0);
      tma::load_2d(&mwot, tma::sa(sW + 16384), wfull, 64, 0);
      for (int T = 0; T < ntT; ++T) {
        int b, a0, s0; coords(blockIdx.x + T * gridDim.x, b, a0, s0);
        const int st = T & 1;
        if (T >= 2) tma::wait(xempty + st, ((T >> 1) - 1) & 1);
        unsigned char* sx = smem + st * SET;
        tma::expect_tx(xfull + st, 8 * TB + 4 * AT * 128);
        for (int kb = 0; kb < 2; ++kb) {
          tma::load_4d(&mdq, tma::sa(sx + SDQ + kb * TB), xfull + st, kb * 64, s0, b, a0);
          tma::load_4d(&mo, tma::sa(sx + SO + kb * TB), xfull + st, kb * 64, s0, b, a0);
          tma::load_4d(&mg, tma::sa(sx + SG + kb * TB), xfull + st, kb * 64, s0, b, a0);
          tma::load_4d(&matt, tma::sa(sx + SATT + kb * TB), xfull + st, kb * 64, s0, b, a0);
        }
        tma::load_3d(&mmod, tma::sa(sx + SMOD), xfull + st, 0, b * S + s0, 8);   // col-blocks 8..11 = gate_a
      }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 " CONS_REGS ";\n" ::: "memory");
  const int cs = wgi;
  unsigned char* sx = smem + cs * SET;
  unsigned char *sDQ = sx + SDQ, *sO = sx + SO, *sG = sx + SG, *sAtt = sx + SATT, *sM = sx + SMOD;
  float* sRed = reinterpret_cast<float*>(sx + SRED);       // [AT][C] per-atom d gate_a over the tile's augments
  const int r0 = warp * 16 + (lane >> 2), cq = (lane & 3) * 2, al = r0 % AT;   // rows r0 / r0 + 8: augments 2*warp / 2*warp + 1, atom al
  const int bar_id = 1 + cs;
  auto bsync = [&]() { asm volatile("bar.sync %0, 128;\n" :: "r"(bar_id) : "memory"); };
  float acc[64];
  uint32_t fa[32];
  tma::wait(wfull, 0);
  for (int T = cs; T < ntT; T += 2) {
    int b, a0, s0; coords(blockIdx.x + T * gridDim.x, b, a0, s0);
    for (int i = tq; i < AT * C; i += 128) sRed[i] = 0.f;
    tma::wait(xfull + cs, (T >> 1) & 1);
    // ---- datt = rn(dq1 * rn(gate_a)) -> A fragments and D (in place); d gate_a partials ----
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int c = i * 8 + cq;
      const float2 ga = *modp(sM, al, c);
      float p0 = 0.f, p1 = 0.f;
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const int r = r0 + 8 * rr;
        uint32_t* pd = reinterpret_cast<uint32_t*>(sDQ + offb(r, c));
        const float2 dq = unpack(*pd), at = unpack(*reinterpret_cast<const uint32_t*>(sAtt + offb(r, c)));
        p0 += dq.x * at.x; p1 += dq.y * at.y;
        const uint32_t da = pack(dq.x * rnb(ga.x), dq.y * rnb(ga.y));
        fa[4 * (i >> 1) + 2 * (i & 1) + rr] = da;
        *pd = da;
      }
      acc[2 * i] = p0; acc[2 * i + 1] = p1;                 // (acc is free until the GEMM)
    }
    bsync();                                               // sRed zeroed
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      atomicAdd(sRed + al * C + i * 8 + cq, acc[2 * i]); atomicAdd(sRed + al * C + i * 8 + cq + 1, acc[2 * i + 1]);
    }
    // ---- dgated = datt Wo (B = Wo^T resident, K-major) ----
    wg::fence();
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const uint64_t d = wg::desc(sW + (s >> 2) * 16384 + (s & 3) * 32);
      if (s == 0) wg::mma_rs_n128_first(fa[0], fa[1], fa[2], fa[3], d, acc);
      else wg::mma_rs_n128(fa[4 * s], fa[4 * s + 1], fa[4 * s + 2], fa[4 * s + 3], d, acc, 1);
    }
    wg::commit(); wg::wait<0>();
    bsync();                                               // sRed complete
    for (int k = tq; k < AT * C / 4; k += 128) {          // one red.add per (atom, 4 channels)
      const int a_ = k / (C / 4), c4 = k % (C / 4);
      if (s0 + a_ < S) red_add_v4(args.dmod + ((long)b * S + s0 + a_) * (6 * C) + 2 * C + c4 * 4, *reinterpret_cast<const float4*>(sRed + a_ * C + c4 * 4));
    }
    // ---- dO, dG, gated, Dv ----
    float dv[2][H];
#pragma unroll
    for (int h = 0; h < H; ++h) { dv[0][h] = 0.f; dv[1][h] = 0.f; }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int c = i * 8 + cq;
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const int r = r0 + 8 * rr;
        uint32_t* pg = reinterpret_cast<uint32_t*>(sG + offb(r, c));
        uint32_t* po = reinterpret_cast<uint32_t*>(sO + offb(r, c));
        const float2 gv = unpack(*pg), ov = unpack(*po);
        const float s0_ = sigmoid(gv.x), s1_ = sigmoid(gv.y), d0 = acc[4 * i + 2 * rr], d1 = acc[4 * i + 2 * rr + 1];
        const float do0 = rnb(d0 * s0_), do1 = rnb(d1 * s1_);
        dv[rr][i >> 2] += do0 * ov.x + do1 * ov.y;
        *reinterpret_cast<uint32_t*>(sAtt + offb(r, c)) = pack(do0, do1);                 // dO -> att slot
        *pg = pack(d0 * ov.x * s0_ * (1.f - s0_), d1 * ov.y * s1_ * (1.f - s1_));         // dG in place
        *po = pack(s0_ * ov.x, s1_ * ov.y);                                               // gated in place
      }
    }
#pragma unroll
    for (int rr = 0; rr < 2; ++rr)
#pragma unroll
      for (int h = 0; h < H; ++h) { dv[rr][h] += __shfl_xor_sync(0xffffffffu, dv[rr][h], 1); dv[rr][h] += __shfl_xor_sync(0xffffffffu, dv[rr][h], 2); }
    {
      const int h = lane & 3;
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const int a = a0 + 2 * warp + rr, s = s0 + al;
        const float v = h == 0 ? dv[rr][0] : h == 1 ? dv[rr][1] : h == 2 ? dv[rr][2] : dv[rr][3];
        if (a < A && s < S) args.dv[(((long)a * B + b) * H + h) * S + s] = v;
      }
    }
    wg::proxy_fence();
    bsync();
    if (tq == 0) {
      for (int kb = 0; kb < 2; ++kb) {
        tma::store_4d(&mdatt, sDQ + kb * TB, kb * 64, s0, b, a0);
        tma::store_4d(&mdo, sAtt + kb * TB, kb * 64, s0, b, a0);
        tma::store_4d(&mdg, sG + kb * TB, kb * 64, s0, b, a0);
        tma::store_4d(&mgated, sO + kb * TB, kb * 64, s0, b, a0);
      }
      tma::store_commit();
      tma::store_wait_read<0>();
      tma::arrive(xempty + cs);
    }
    bsync();                                               // sRed reads done before the next tile zeroes it
  }
}

PFN_cuTensorMapEncodeTiled tma_encode() {
  static PFN_cuTensorMapEncodeTiled fn = nullptr;
  if (fn == nullptr) {
    void* p = nullptr; cudaDriverEntryPointQueryResult qr;
    C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &qr));
    TORCH_CHECK(p != nullptr && qr == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled>(p);
  }
  return fn;
}
CUtensorMap encode(CUtensorMapDataType dt, int rank, void* ptr, const uint64_t* gdim, const uint64_t* gstride, const uint32_t* bdim, CUtensorMapSwizzle sw) {
  alignas(64) CUtensorMap m{};
  uint32_t estride[5] = {1, 1, 1, 1, 1};
  CUresult r = tma_encode()(&m, dt, rank, ptr, gdim, gstride, bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}
CUtensorMap map4d(const torch::Tensor& t, long A, long B, long S) {
  uint64_t gdim[4] = {(uint64_t)C, (uint64_t)S, (uint64_t)B, (uint64_t)A};
  uint64_t gstride[3] = {(uint64_t)C * 2, (uint64_t)S * C * 2, (uint64_t)B * S * C * 2};
  uint32_t bdim[4] = {64, AT, 1, SP};
  return encode(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, t.data_ptr(), gdim, gstride, bdim, CU_TENSOR_MAP_SWIZZLE_128B);
}
CUtensorMap mapmod(const torch::Tensor& t, long rows) {
  uint64_t gdim[3] = {32, (uint64_t)rows, 24}; uint64_t gstride[2] = {6 * C * 4, 128};
  uint32_t bdim[3] = {32, AT, 4};
  return encode(CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 3, t.data_ptr(), gdim, gstride, bdim, CU_TENSOR_MAP_SWIZZLE_128B);
}
CUtensorMap map2d(const torch::Tensor& t, uint64_t rows, uint64_t cols, uint32_t brows) {
  uint64_t gdim[2] = {cols, rows}; uint64_t gstride[1] = {cols * 2};
  uint32_t bdim[2] = {64, brows};
  return encode(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), gdim, gstride, bdim, CU_TENSOR_MAP_SWIZZLE_128B);
}
int num_sms(int dev) { static int s = 0; if (s == 0) cudaDeviceGetAttribute(&s, cudaDevAttrMultiProcessorCount, dev); return s; }
}  // namespace

// dq1, o, g, att: [M, 128] bf16; mod / dmod [B*S, 768] fp32; wot = Wo^T [128, 128] bf16.  Returns [dO, dG, Dv [N, H, S] fp32, datt, gated].
std::vector<torch::Tensor> oproj_bwd(torch::Tensor dq1, torch::Tensor o, torch::Tensor g, torch::Tensor att, torch::Tensor mod, torch::Tensor dmod,
                                     torch::Tensor wot, int64_t A, int64_t B, int64_t S) {
  const long M = A * B * S;
  for (const auto* t : {&dq1, &o, &g, &att}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 && t->is_contiguous() && t->numel() == M * C, "rows: [M, 128] bf16");
  TORCH_CHECK(mod.scalar_type() == torch::kFloat && mod.is_contiguous() && mod.numel() == B * S * 6 * C, "mod [B*S, 768] fp32");
  TORCH_CHECK(dmod.scalar_type() == torch::kFloat && dmod.is_contiguous() && dmod.numel() == B * S * 6 * C, "dmod [B*S, 768] fp32");
  TORCH_CHECK(wot.is_contiguous() && wot.scalar_type() == torch::kBFloat16 && wot.sizes() == torch::IntArrayRef({C, C}), "wot [128, 128] bf16");
  auto opt = dq1.options();
  auto dO = torch::empty({M, C}, opt), dG = torch::empty({M, C}, opt), datt = torch::empty({M, C}, opt), gated = torch::empty({M, C}, opt);
  auto Dv = torch::empty({A * B, H, S}, opt.dtype(torch::kFloat));
  CUtensorMap mdq = map4d(dq1, A, B, S), mo = map4d(o, A, B, S), mg = map4d(g, A, B, S), matt = map4d(att, A, B, S);
  CUtensorMap mdo = map4d(dO, A, B, S), mdg = map4d(dG, A, B, S), mdatt = map4d(datt, A, B, S), mgated = map4d(gated, A, B, S);
  CUtensorMap mmod = mapmod(mod, B * S), mwot = map2d(wot, C, C, 128);
  Args args; args.dv = Dv.data_ptr<float>(); args.dmod = dmod.data_ptr<float>();
  args.S = (int)S; args.A = (int)A; args.B = (int)B;
  args.nab = (int)((S + AT - 1) / AT); args.nag = (int)((A + SP - 1) / SP); args.ntile = args.nab * args.nag * (int)B;
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(oproj_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES)); attr = true; }
  const int grid = std::min(args.ntile, num_sms(dq1.device().index()));
  oproj_bwd_kernel<<<grid, THREADS, BYTES, at::cuda::getCurrentCUDAStream()>>>(args, mdq, mo, mg, matt, mmod, mwot, mdo, mdg, mdatt, mgated);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dO, dG, Dv, datt, gated};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("oproj_bwd", &oproj_bwd, "ESMFold2 SWA atom block out-projection backward (bf16 wgmma)");
}

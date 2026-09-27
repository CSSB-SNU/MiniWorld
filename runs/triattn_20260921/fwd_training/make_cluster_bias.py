"""Multicast bias between outer-row CTAs without changing the score arithmetic."""
from pathlib import Path

r = Path(__file__).resolve().parent
base = (r / 'cooperative_q1_s2' / 'fused.cu').read_text()
for cr in (2, 4):
    s = base.replace('#include <cute/tensor.hpp>',
                     '#include <cute/tensor.hpp>\n#include <cute/arch/cluster_sm90.hpp>')
    s = s.replace('static constexpr int R=1;', 'static constexpr int R=1, CR=%d;' % cr)
    s = s.replace('using TB=decltype(make_tma_copy(SM90_TMA_LOAD{},',
                  'using TB=decltype(make_tma_copy(SM90_TMA_LOAD_MULTICAST{},')
    s = s.replace('auto tb=make_tma_copy(SM90_TMA_LOAD{},bg,',
                  'auto tb=make_tma_copy(SM90_TMA_LOAD_MULTICAST{},bg,')
    s = s.replace('s.bempty[b].init(Config::R);', 's.bempty[b].init(Config::CR);')
    s = s.replace('  __syncthreads();', '''  __syncthreads();
  if(tid==0) {
    s.bfull[0].arrive_and_expect_tx(4096*sizeof(Element));
    if(nt>1)s.bfull[1].arrive_and_expect_tx(4096*sizeof(Element));
  }
  // Both destination barriers must be armed before the initial multicast.
  cluster_arrive(); cluster_wait();''', 1)
    s = s.replace('    s.bfull[stage].arrive_and_expect_tx(4096*sizeof(Element));\n', '')
    old = '    tma_load(p.bias,bb,make_tensor(make_smem_ptr(s.bias[stage].data()),Config::BL{}),s.bfull[stage]);'
    assert old in s
    s = s.replace(old, '''    if(rg%Config::CR==0) {
      // Prior bias readers in every CTA have finished and rearmed their full barrier.
      if(kt>=2)s.bempty[stage].wait(((kt/2)-1)%2);
      auto c=p.bias.get_slice(_0{});
      auto dst=make_tensor(make_smem_ptr(s.bias[stage].data()),Config::BL{});
      copy(p.bias.with(reinterpret_cast<uint64_t&>(s.bfull[stage]),uint16_t((1<<Config::CR)-1)),
           c.partition_S(bb),c.partition_D(dst));
    }''')
    old = '        if(tid==0 && kt>0 && kt+1<nt)load(kt+1);'
    assert old in s
    s = s.replace(old, '''        if(tid==0 && kt>0 && kt+1<nt) {
          int free_stage=(kt+1)%2;
          s.bfull[free_stage].arrive_and_expect_tx(4096*sizeof(Element));
          s.bempty[free_stage].arrive(uint32_t(0),uint32_t(1));
          load(kt+1);
        }''')
    s = s.replace('\n}\nstd::vector<torch::Tensor> launch_forward',
                  '\n  // All multicast recipients must remain live through their final reads/stores.\n  cluster_arrive(); cluster_wait();\n}\nstd::vector<torch::Tensor> launch_forward', 1)
    old = '  if(L==64) {\n    C10_CUDA_CHECK(cudaFuncSetAttribute(training_fwd_stream<true>'
    assert old in s
    s = s.replace(old, '''  cudaLaunchAttribute attr{}; attr.id=cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x=1; attr.val.clusterDim.y=Config::CR; attr.val.clusterDim.z=1;
  cudaLaunchConfig_t cfg{}; cfg.gridDim=dim3(L/64,L,4); cfg.blockDim=dim3(128,1,1);
  cfg.dynamicSmemBytes=sizeof(Config::Shared); cfg.stream=at::cuda::getCurrentCUDAStream();
  cfg.attrs=&attr; cfg.numAttrs=1;
  if(L==64) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(training_fwd_stream<true>''')
    for single in ('true', 'false'):
        old = '    training_fwd_stream<%s><<<dim3(L/64,L/Config::R,4),128,sizeof(Config::Shared),at::cuda::getCurrentCUDAStream()>>>(p);' % single
        assert old in s
        s = s.replace(old, '    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg,training_fwd_stream<%s>,p));' % single)
    p = r / ('cooperative_cluster%d_v2' % cr)
    p.mkdir(exist_ok=True)
    (p / 'fused.cu').write_text(s)

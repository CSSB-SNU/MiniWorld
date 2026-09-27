"""All-head KV projection plus matched resident/streaming hot6t attention controls."""
from pathlib import Path
import argparse

R = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--artifact', required=True)
ap.add_argument('--consumers', type=int, default=2)
ap.add_argument('--resident', action='store_true')
ap.add_argument('--stages', type=int, choices=(1,2), default=2)
ap.add_argument('--head1', action='store_true')
ap.add_argument('--head1-local', action='store_true')
ap.add_argument('--local-queries', action='store_true')
a = ap.parse_args()
s = (R/'hot6t/fused.cu').read_text()
project = (R/'head4_kv_projection.cuh').read_text()
if a.head1:
    project = project.replace('using WL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_128,_128>{}));',
                              'using WL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_32,_128>{}));')
    project = project.replace('using OL=ZL;', 'using OL=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_64,_32>{}));')
    project = project.replace('WL{},Shape<_128,_128>', 'WL{},Shape<_32,_128>')
    project = project.replace('OL{},Shape<_64,_128>', 'OL{},Shape<_64,_32>')
    project = project.replace('Shape<_64,_128,_128>', 'Shape<_64,_32,_128>')
    project = project.replace('array_aligned<Element,8192,1024> z,out;',
                              'array_aligned<Element,8192,1024> z;array_aligned<Element,2048,1024> out;')
    project = project.replace('array_aligned<Element,16384,1024> w;', 'array_aligned<Element,4096,1024> w;')
    project = project.replace('make_identity_tensor(Shape<_64,_128>{})', 'make_identity_tensor(Shape<_64,_32>{})')
    project = project.replace('partition_fragment_C(mma,Shape<_64,_128>{})', 'partition_fragment_C(mma,Shape<_64,_32>{})')
    project = project.replace('(16384+(which==0?8192:0))', '(4096+(which==0?8192:0))')
    project = project.replace('tma_load(t,g,ws,s.full);',
                              'auto tile=local_tile(g,Shape<_32,_128>{},make_coord(blockIdx.y,0));tma_load(t,tile,ws,s.full);')
    project = project.replace('auto tile=local_tile(g,Shape<_64,_128>{},make_coord(blockIdx.x,0));\n      auto out=',
                              'auto tile=local_tile(g,Shape<_64,_32>{},make_coord(blockIdx.x,blockIdx.y));\n      auto out=')
    project = project.replace('<<<rows/64,128,', '<<<dim3(rows/64,4),128,')
    project = project.replace('// Four D32 heads are one dense N128 projection. Z is loaded once for K and V.',
                              '// Control: each CTA projects one D32 head. Identical two-pass K/V arithmetic.')
    if a.head1_local:
        project = project.replace('blockIdx.x', '(blockIdx.x/4)').replace('blockIdx.y', '(blockIdx.x%4)')
        project = project.replace('<<<dim3(rows/64,4),128,', '<<<(rows/64)*4,128,')
pos = s.index('template<int Capacity,int Consumers,int Stages,int Projectors>\n__global__')
s = s[:pos] + project + '\n' + s[pos:]
s = s.replace('TB bias; TO out;', 'TB bias; TQ key,value; TO out;')
s = s.replace('prefetch_tma_descriptor(p.wk.get_tma_descriptor());prefetch_tma_descriptor(p.wv.get_tma_descriptor());',
              'prefetch_tma_descriptor(p.key.get_tma_descriptor());prefetch_tma_descriptor(p.value.get_tma_descriptor());')
start = s.index('  // All KV remain')
end = s.index('  int bias_epoch=0;', start)
if a.resident:
    replacement = '''  // Matched control: same attention schedule, KV loaded rather than projected.
  for(int kt=c;kt<nt;kt+=Consumers) {
    if(lane==0) {
      s.zfull[c].arrive_and_expect_tx(8192);
      for(int which=0;which<2;++which) {
        auto const& t=which==0?p.key:p.value;
        auto g=t.get_tma_tensor(make_shape(L,_32{},_4{},L));
        auto src=local_tile(g(_,_,h,row),Shape<_64,_32>{},make_coord(kt,0));
        auto dst=make_tensor(make_smem_ptr(s.kv[which][kt].data()),typename C::QL{});
        tma_load(t,src,dst,s.zfull[c]);
      }
    }
    s.zfull[c].wait((kt/Consumers)%2);
  }
  __syncthreads();
'''
    # Reinitialize the reused transaction barrier after every KV transaction retires.
    replacement += '''  if(lane==0) { s.zfull[c].init(1); }
  cutlass::arch::fence_barrier_init();
  __syncthreads();
'''
else:
    replacement = ''
s = s[:start] + replacement + s[end:]
s = s.replace('s.wfull.wait((1+batch/Consumers)%2)', 's.wfull.wait((batch/Consumers)%2)')
if not a.resident:
    s = s.replace('array_aligned<Element,2048,1024> kv[2][Capacity/64];',
                  'array_aligned<Element,2048,1024> kv[Consumers][2][Stages];')
    s = s.replace('for(int batch=0;batch<nt;batch+=Consumers) {',
                  '{\n    int batch=int(blockIdx.z)*Consumers;')
    s = s.replace('s.wfull.wait((batch/Consumers)%2)', 's.wfull.wait(0)')
    s = s.replace('s.bfull[c][stage].arrive_and_expect_tx(4096*sizeof(Element));', '''s.bfull[c][stage].arrive_and_expect_tx(8192*sizeof(Element));
        for(int which=0;which<2;++which) {
          auto const& t=which==0?p.key:p.value;
          auto g=t.get_tma_tensor(make_shape(L,_32{},_4{},L));
          auto src=local_tile(g(_,_,h,row),Shape<_64,_32>{},make_coord(kt,0));
          auto dst=make_tensor(make_smem_ptr(s.kv[c][which][stage].data()),typename C::QL{});
          tma_load(t,src,dst,s.bfull[c][stage]);
        }''')
    s = s.replace('s.kv[0][kt]', 's.kv[c][0][stage]').replace('s.kv[1][kt]', 's.kv[c][1][stage]')
    s = s.replace('      int stage=kt%Stages;\n      auto sk=',
                  '      int stage=kt%Stages;\n      s.bfull[c][stage].wait((bias_epoch+kt/Stages)%2);\n      auto sk=')
    s = s.replace('''      if constexpr(Stages==2) {
        if(lane==0 && kt>0 && kt+1<nt)load_bias(kt+1);
      }
      s.bfull[c][stage].wait((bias_epoch+kt/Stages)%2);''', '')
    s = s.replace('''        if constexpr(Stages==1) {
          cutlass::arch::NamedBarrier::sync(128,c+1);
          if(lane==0 && kt+1<nt)load_bias(kt+1);
        }''', '')
    needle = '\n    }\n    warpgroup_wait<0>(); warpgroup_fence_operand(out);'
    assert needle in s
    s = s.replace(needle, '''
      // Retire K/V and bias readers before refilling this ring slot.
      cutlass::arch::NamedBarrier::sync(128,c+1);
      if(lane==0 && kt+Stages<nt)load_bias(kt+Stages);
    }
    warpgroup_wait<0>(); warpgroup_fence_operand(out);''')
    s = s.replace('<<<dim3(4,L),', '<<<dim3(4,L,(L/64+Consumers-1)/Consumers),')
    if a.local_queries:
        s = s.replace('h=blockIdx.x,row=blockIdx.y', 'h=blockIdx.x%4,row=blockIdx.y')
        s = s.replace('int batch=int(blockIdx.z)*Consumers;', 'int batch=(int(blockIdx.x)/4)*Consumers;')
        s = s.replace('int ep_lane=ep_tid%128,', 'ep_h%=4;\n    int ep_lane=ep_tid%128,')
        s = s.replace('<<<dim3(4,L,(L/64+Consumers-1)/Consumers),',
                      '<<<dim3(4*((L/64+Consumers-1)/Consumers),L),')

s = s.replace('torch::Tensor out,torch::Tensor counts) {',
              'torch::Tensor key,torch::Tensor value,torch::Tensor out,torch::Tensor counts) {')
pos = s.index('  typename C::Params p{tz,')
s = s[:pos] + '''  auto read_kv=[&](torch::Tensor const& t) {
    auto g=make_tensor(make_gmem_ptr((Element const*)t.data_ptr()),make_shape(L,_32{},_4{},L),typename C::QStride{128,_1{},_32{},int64_t(L)*128});
    return make_tma_copy(SM90_TMA_LOAD{},g,typename C::QL{},Shape<_64,_32>{},_1{});
  };
''' + s[pos:]
s = s.replace('tb,save(out),L,', 'tb,read_kv(key),read_kv(value),save(out),L,')
s = s.replace('  auto out=torch::empty_like(z);', '''  auto key=torch::empty_like(z),value=torch::empty_like(z);
  project_head4_kv(z,wk,wv,key,value);
  auto out=torch::empty_like(z);''')
s = s.replace('(z,wq,wk,wv,wg,b,out,counts)', '(z,wq,wk,wv,wg,b,key,value,out,counts)')
if not a.resident:
    for cap, oldc, stages in ((128,2,1),(128,2,2),(384,6,2),(768,6,1),(1024,6,1)):
        s = s.replace(f'launch<{cap},{oldc},{stages},{oldc}>',
                      f'launch<{cap},{a.consumers},{1 if stages==1 and cap==128 else a.stages},{a.consumers}>')
    s = s.replace('Config<1024,6,1,6>::Shared', f'Config<1024,{a.consumers},{a.stages},{a.consumers}>::Shared')
s = s.replace('void qkv_attention_inference(', 'void qg_attention_head4kv(')
s = s.replace('qkv_attention_inference<', 'qg_attention_head4kv<')
s = s.replace('  m.def("forward_audit",&launch_forward);', '''  m.def("forward_audit",&launch_forward);
  m.def("project_kv",[](torch::Tensor z,torch::Tensor k,torch::Tensor v){
    c10::cuda::CUDAGuard guard(z.device());
    auto key=torch::empty_like(z),value=torch::empty_like(z);
    project_head4_kv(z,k,v,key,value);
    return std::vector<torch::Tensor>{key,value};
  });''')
folder = R/a.artifact
assert not (folder/'build-ready.json').exists(), 'Never overwrite a built artifact'
folder.mkdir(exist_ok=True)
(folder/'fused.cu').write_text(s)

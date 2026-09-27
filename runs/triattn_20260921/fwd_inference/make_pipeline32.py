"""Resident single-head CTA, two32-key score buffers and lifetime-safe async overlap."""
from pathlib import Path
import argparse

ap=argparse.ArgumentParser();ap.add_argument('--artifact',required=True)
ap.add_argument('--consumers',type=int,choices=(2,4,6),default=4)
ap.add_argument('--serial',action='store_true')
a=ap.parse_args();r=Path(__file__).resolve().parent
s=(r/'hot4t/fused.cu').read_text()
s=s.replace('  using VT=', '  using KHL=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_32,_32>{}));\n  using VT=',1)
s=s.replace('GMMA::Layout_MN_SW64_Atom<Element>{},Shape<_32,_64>', 'GMMA::Layout_MN_SW64_Atom<Element>{},Shape<_32,_32>')
s=s.replace('GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_64>{}));\n  using ZL=',
            'GMMA::Layout_K_SW64_Atom<Element>{},Shape<_64,_32>{}));\n  using ZL=')
s=s.replace('BS{},BStride{}),BL{},Shape<_64,_64>', 'BS{},BStride{}),BL{},Shape<_64,_32>')
s=s.replace('float,Shape<_64,_64,_32>', 'float,Shape<_64,_32,_32>')
s=s.replace('float,Shape<_64,_32,_64>', 'float,Shape<_64,_32,_32>')
s=s.replace('array_aligned<Element,4096,1024> bias[Consumers][Stages]', 'array_aligned<Element,2048,1024> bias[Consumers][Stages]')
s=s.replace('s.bfull[c][stage].arrive_and_expect_tx(4096*sizeof(Element))', 's.bfull[c][stage].arrive_and_expect_tx(2048*sizeof(Element))')
s=s.replace('bg(_,_,h),Shape<_64,_64>', 'bg(_,_,h),Shape<_64,_32>')
s=s.replace('bg,typename C::BL{},Shape<_64,_64>', 'bg,typename C::BL{},Shape<_64,_32>')
s=s.replace('partition_fragment_C(smma,Shape<_64,_64>{})', 'partition_fragment_C(smma,Shape<_64,_32>{})')
s=s.replace('b<Stages && b<nt', 'b<Stages && b<nt*2')
s=s.replace('kt<nt;', 'kt<nt*2;')
s=s.replace('kt+1<nt', 'kt+1<nt*2')
s=s.replace('bias_epoch+=nt/Stages;', 'bias_epoch+=(nt*2)/Stages;')
s=s.replace('s.kv[0][kt].data()),typename C::QL{}', 's.kv[0][kt/2].data()+(kt%2)*1024),typename C::KHL{}')
s=s.replace('s.kv[1][kt].data()),typename C::VT{}', 's.kv[1][kt/2].data()+(kt%2)*1024),typename C::VT{}')
if not a.serial:
    lo=s.index('    for(int kt=0;kt<nt*2;++kt) {')
    hi=s.index('    warpgroup_wait<0>(); warpgroup_fence_operand(out);\n\n      if constexpr(!Safe)',lo)
    serial=s[lo:hi]
    # Stable retry stays serial; only the checked max-free path uses overlap.
    begin=serial.index('        uint32_t bp=cast_smem_ptr_to_uint(')
    end=serial.index('        if constexpr(Stages==1)',begin)
    reader=serial[begin:end]
    fast=r'''    } else {
      auto next_score=partition_fragment_C(smma,Shape<_64,_32>{});
      auto layout=flash::convert_layout_acc_Aregs<typename C::PV>(score.layout());
      auto ar0=make_tensor(score.data(),layout);
      auto pr=make_tensor_like<Element>(ar0);clear(pr);
      auto issue_qk=[&](auto& dst,int kt){
        auto sk=make_tensor(make_smem_ptr(s.kv[0][kt/2].data()+(kt%2)*1024),typename C::KHL{});
        auto kb=st.partition_fragment_B(sk);
        flash::gemm<true,-1>(smma,qa,kb,dst);
      };
      auto step=[&](auto& score,auto& next,int kt){
        // Queue invariant on entry: current QK, then previous PV (except the first step).
        if(kt==0)warpgroup_wait<0>();else warpgroup_wait<1>();
        warpgroup_fence_operand(score);
        cutlass::arch::NamedBarrier::sync(128,c+1);
        if(kt+1<nt*2)issue_qk(next,kt+1);
        int stage=kt%Stages;
        if constexpr(Stages==2){if(lane==0 && kt>0 && kt+1<nt*2)load_bias(kt+1);}
        s.bfull[c][stage].wait((bias_epoch+kt/Stages)%2);
        asm volatile("":::"memory");
@READER@
        if constexpr(Stages==1){
          cutlass::arch::NamedBarrier::sync(128,c+1);
          if(lane==0 && kt+1<nt*2)load_bias(kt+1);
        }
        #pragma unroll
        for(int x=0;x<size(score);++x)score(x)=ex2(score(x)*(SCALE*LOG2E)-16.f);
        // Retire previous PV before overwriting its RS operands or accumulators.
        // If next QK exists, it is the newer group and may stay outstanding.
        if(kt+1<nt*2)warpgroup_wait<1>();else warpgroup_wait<0>();
        warpgroup_fence_operand(pr);warpgroup_fence_operand(out);warpgroup_fence_operand(sums);
        auto ar=make_tensor(score.data(),layout);flash::convert_type_out(ar,pr);
        auto vv=make_tensor(make_smem_ptr(s.kv[1][kt/2].data()+(kt%2)*1024),typename C::VT{});
        auto vb=pt.partition_fragment_B(vv);
        warpgroup_fence_operand(pr);warpgroup_fence_operand(out);warpgroup_fence_operand(sums);
        warpgroup_arrive();
        pmma.accumulate_=GMMA::ScaleOut::One;sum_mma.accumulate_=GMMA::ScaleOut::One;
        #pragma unroll
        for(int kk=0;kk<size<2>(pr);++kk){
          cute::gemm(pmma,pr(_,_,kk),vb(_,_,kk),out);
          cute::gemm(sum_mma,pr(_,_,kk),ob(_,_,0),sums);
        }
        warpgroup_commit_batch();
      };
      issue_qk(score,0);
      for(int kt=0;kt<nt*2;kt+=2){step(score,next_score,kt);step(next_score,score,kt+1);}
      warpgroup_wait<0>();warpgroup_fence_operand(pr);warpgroup_fence_operand(sums);
    }
'''.replace('@READER@',reader)
    s=s[:lo]+'    if constexpr(Safe) {\n'+serial+fast+s[hi:]
for cap,cons,st in ((384,6,2),(768,4,2),(1024,4,2)):
    s=s.replace(f'launch<{cap},{cons},{st},{cons}>',f'launch<{cap},{a.consumers},{st},{a.consumers}>')
s=s.replace('Config<1024,4,2,4>::Shared',f'Config<1024,{a.consumers},2,{a.consumers}>::Shared')
s=s.replace('qkv_attention_inference', 'qkv_attention_n32_serial' if a.serial else 'qkv_attention_n32_pipeline')
folder=r/a.artifact;assert not (folder/'build-ready.json').exists()
folder.mkdir(exist_ok=True);(folder/'fused.cu').write_text(s)

"""Fuse P*V and P*1 with an N40 descriptor targeting a reused constant tile."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--artifact',required=True)
a=p.parse_args();r=Path(__file__).resolve().parent;s=(r/a.base/'fused.cu').read_text()
s=s.replace('#include <cute/tensor.hpp>','#define CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED\n#include <cute/tensor.hpp>')
s=s.replace('  using Sum=', '  using FusedPV=decltype(make_tiled_mma(SM90::GMMA::MMA_64x40x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>{}));\n  using Sum=',1)
s=s.replace('array_aligned<Element,128,128> ones;', 'array_aligned<Element,512,1024> ones;')
s=s.replace('if(tid<128)s.ones[tid]=Element(1.f);', 'for(int i=tid;i<512;i+=Consumers*128)s.ones[i]=Element(1.f);')
s=s.replace('    auto out=partition_fragment_C(pmma,Shape<_64,_32>{});',
'''    typename C::FusedPV fused_mma;
    auto fused_acc=partition_fragment_C(fused_mma,Shape<_64,_40>{});
    auto out_layout=partition_fragment_C(pmma,Shape<_64,_32>{}).layout();
    auto out=make_tensor(fused_acc.data(),out_layout);''')
s=s.replace('auto sums=partition_fragment_C(sum_mma,Shape<_64,_8>{});clear(sums);',
'''auto sum_layout=partition_fragment_C(sum_mma,Shape<_64,_8>{}).layout();
      auto sums=make_tensor(fused_acc.data()+16,sum_layout);clear(sums);''')
s=s.replace('      auto ones=make_tensor(make_smem_ptr(s.ones.data()),typename C::Ones{});\n      auto ob=sum_mma.get_slice(lane).partition_fragment_B(ones);\n','')
old='''          warpgroup_fence_operand(pr);warpgroup_fence_operand(out);warpgroup_fence_operand(sums);
          warpgroup_arrive();
          pmma.accumulate_=GMMA::ScaleOut::One;
          sum_mma.accumulate_=GMMA::ScaleOut::One;
          #pragma unroll
          for(int kk=0;kk<size<2>(pr);++kk) {
            cute::gemm(pmma,pr(_,_,kk),vb(_,_,kk),out);
            cute::gemm(sum_mma,pr(_,_,kk),ob(_,_,0),sums);
          }
          warpgroup_commit_batch();warpgroup_wait<0>();
          warpgroup_fence_operand(pr);warpgroup_fence_operand(out);warpgroup_fence_operand(sums);'''
new='''          warpgroup_fence_operand(pr);warpgroup_fence_operand(fused_acc);
          warpgroup_arrive();fused_mma.accumulate_=GMMA::ScaleOut::One;
          #pragma unroll
          for(int kk=0;kk<size<2>(pr);++kk) {
            auto operand=vb(_,_,kk);
            auto desc=operand.data().desc_;
            uint32_t vaddr=uint32_t(desc.bitfield.start_address_)<<4;
            uint32_t oneaddr=cast_smem_ptr_to_uint(s.ones.data());
            // N32-group displacement points at the same 32x16 ones tile for every K16.
            desc.bitfield.leading_byte_offset_=(oneaddr-vaddr)>>4;
            auto wide_v=make_tensor(GMMA::DescriptorIterator{desc},operand.layout());
            cute::gemm(fused_mma,pr(_,_,kk),wide_v,fused_acc);
          }
          warpgroup_commit_batch();warpgroup_wait<0>();
          warpgroup_fence_operand(pr);warpgroup_fence_operand(fused_acc);'''
assert old in s;s=s.replace(old,new)
s=s.replace('qkv_attention_inference','qkv_attention_fused_n40')
folder=r/a.artifact;assert not (folder/'build-ready.json').exists()
folder.mkdir(exist_ok=True);(folder/'fused.cu').write_text(s)

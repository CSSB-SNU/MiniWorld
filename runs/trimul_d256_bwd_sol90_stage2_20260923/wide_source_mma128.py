"""Pair adjacent dW N64 accumulators into WGMMA N128 without regrouping K."""
import re


def widen_dw(body):
    marker='extern "C" __global__'
    helper='''
template<int K> TMN_DEVI void dw128(float (&dw)[NC][32],uint64_t a,uint64_t b,int accumulate){
 static_for<NC/2>([&](auto cc){constexpr int c=decltype(cc)::value;
  auto& v=*reinterpret_cast<float(*)[64]>(dw[2*c]);
  mma128_off<K*32,c*16384+K*2048,0,1>(v,a,b,accumulate);
 });
 if constexpr(NC%2)mma64_off<K*32,(NC-1)*8192+K*2048,0,1>(dw[NC-1],a,b,accumulate);
}
'''
    # The helper must precede pipe_consume if the producer/consumer variant
    # is being transformed, and precede the kernel in the ordinary source.
    point='template<int WG> TMN_DEVI void pipe_consume' if 'void pipe_consume' in body else marker
    assert body.count(point)==1
    body=body.replace(point,helper+'\n'+point)
    pattern=r'static_for<NC>\(\[&\]\(auto cc\)\{constexpr int c=decltype\(cc\)::value;\s*mma64_off<k\*32,c\*8192\+k\*2048,0,1>\(dw\[c\],(smem_desc\(smem_u32\([^;]+?),it>0\|\|k>0\);\s*\}\);'
    body,count=re.subn(pattern,lambda m:'dw128<k>(dw,'+m.group(1)+',it>0||k>0);',body)
    assert count==1,count
    return body

"""Preserve the K accumulation order while issuing one N256 WGMMA."""
from pathlib import Path
import re
def widen(body):
 # N256 needs >=154 statically targeted registers in ptxas. The 256-thread
 # kernel therefore loses its two-CTA launch bound; occupancy is queried
 # from the compiled kernel by DxLN. The 384-thread row-sharing path already
 # uses one CTA and retains its bound.
 body=body.replace('__launch_bounds__(256,2)','__launch_bounds__(256,1)')
 assert body.count('float v0[64]={},v1[64]={};')==1
 body=body.replace('float v0[64]={},v1[64]={};','float v[128]={};')
 body=body.replace('fence_regs(v0);fence_regs(v1);','fence_regs(v);')
 pattern=r'mma128<([01]),1>\(v0,a,(.*?)\);mma128<\1,1>\(v1,a,.*?\);'
 body,count=re.subn(pattern,lambda m:f'mma256<{m[1]},1>(v,a,{m[2]});',body)
 assert count==2,count
 assert body.count('static_for<64>([&](auto jj)')==1
 body=body.replace('static_for<64>([&](auto jj)','static_for<128>([&](auto jj)')
 body=body.replace('cv(v0[j])','cv(v[j])')
 body,count=re.subn(r'reinterpret_cast<bf\*>\((sm|ln)\)\[pos\(r,c\+128\)\]=cv\(v1\[j\]\);','',body)
 assert count==1 and 'v0' not in body and 'v1' not in body
 return body.replace('constexpr int D=256',Path(__file__).with_name('mma256.cuh').read_text()+'\nconstexpr int D=256')

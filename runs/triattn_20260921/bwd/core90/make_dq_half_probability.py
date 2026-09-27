"""Retire current bias into half of P in FP32, without any extra allocation.

After all bias readers finish, 16 of each lane's 32 probabilities use that
tile's 8192 bytes. The remaining16 stay in registers while dP is computed.
Try the occupancy benefit separately from the serial schedule's cost.
"""
from pathlib import Path

root = Path(__file__).resolve().parent.parent / 'dq'
base = (root / 'rs_softmax_overlap/fused.cu').read_text()
a = base.index('  auto score=partition_fragment_C')
b = base.index('  auto acc_a=make_tensor(dp.data()', a)
old = base[a:b]
prob_start = old.index('  uint32_t bp=')
prob_end = old.index('  warpgroup_wait<0>();warpgroup_fence_operand(dp);')
prob = old[prob_start:prob_end]
new = '''  float kept[16];
  uint32_t pp=cast_smem_ptr_to_uint(s.bias[slot].data())+lane*4;
  {
   auto score=partition_fragment_C(smma,Shape<_64,_64>{});
   flash::gemm<true,0>(smma,qa,kb,score);
''' + prob + '''   // Every ldmatrix reader must finish before bias storage is reused.
   __syncthreads();
   #pragma unroll
   for(int x=0;x<16;++x){
    float v=score(x);uint32_t ptr=pp+x*128*4;
    asm volatile("st.shared.f32 [%0],%1;"::"r"(ptr),"f"(v):"memory");
    kept[x]=score(x+16);
   }
  }
  auto dp=partition_fragment_C(smma,Shape<_64,_64>{});
  flash::gemm<true,0>(smma,da,vb,dp);
  #pragma unroll
  for(int x=0;x<32;++x){
   float pr;
   if(x<16)pr=sloadf(pp+x*128*4);else pr=kept[x-16];
   dp(x)=pr*(dp(x)-((x%4)/2?d1:d0));
  }
'''
for ctas in (4, 5):
    s = base[:a] + new + base[b:]
    s = s.replace('__launch_bounds__(128,4)', '__launch_bounds__(128,%d)' % ctas)
    target = root / ('rs_half_p%d' % ctas)
    target.mkdir(exist_ok=True)
    (target / 'fused.cu').write_text(s)

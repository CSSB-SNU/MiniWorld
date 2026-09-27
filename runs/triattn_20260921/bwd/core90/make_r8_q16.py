"""R8 Q16: smaller score fragments permit four consumer warpgroups.

All variants retain FP32 R8 bias partials and final dK/dV output ownership.
The partial buffer has exactly the installed R8 size, with a Q16 tile layout.
"""
from pathlib import Path

root = Path(__file__).resolve().parent.parent / 'bias_fusion'
s = (root / 'rs8_producer32/grouped.cu').read_text()
s = s.replace('QL=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_32,_32>{}))',
              'QL=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_16,_32>{}))')
s = s.replace('QT=decltype(tile_to_shape(GMMA::Layout_MN_SW64_Atom<Element>{},Shape<_32,_32>{}))',
              'QT=decltype(tile_to_shape(GMMA::Layout_MN_SW64_Atom<Element>{},Shape<_32,_16>{}))')
s = s.replace('SL=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_64,_32>{}))',
              'SL=decltype(tile_to_shape(GMMA::Layout_K_SW32_Atom<Element>{},Shape<_64,_16>{}))')
s = s.replace('Shape<_32,_64>{}', 'Shape<_16,_64>{}')
s = s.replace('QL{},Shape<_32,_32>{}', 'QL{},Shape<_16,_32>{}')
s = s.replace('Shape<_64,_32,_32>>()', 'Shape<_64,_16,_32>>()')
s = s.replace('Shape<_64,_32,_32>,GMMA::Major', 'Shape<_64,_32,_16>,GMMA::Major')
s = s.replace('array_aligned<Element,1024,1024> q', 'array_aligned<Element,512,1024> q')
s = s.replace('array_aligned<Element,2048,1024> bias', 'array_aligned<Element,1024,1024> bias')
s = s.replace('array_aligned<float,32,128> lse', 'array_aligned<float,16,128> lse')
s = s.replace('nt=L/32', 'nt=L/16')
s = s.replace('s.b_full[bs].arrive_and_expect_tx(2048*sizeof(Element))',
              's.b_full[bs].arrive_and_expect_tx(1024*sizeof(Element))')
s = s.replace('2*1024*sizeof(Element)+2*32*sizeof(float)', '2*512*sizeof(Element)+2*16*sizeof(float)')
s = s.replace('Shape<_32,_32>{},make_coord(jt,0)', 'Shape<_16,_32>{},make_coord(jt,0)')
s = s.replace('jt*32', 'jt*16').replace('32*sizeof(float)', '16*sizeof(float)')
s = s.replace('st.partition_C(make_identity_tensor(Shape<_64,_32>{}))',
              'st.partition_C(make_identity_tensor(Shape<_64,_16>{}))')
s = s.replace('partition_fragment_C(smma,Shape<_64,_32>{})',
              'partition_fragment_C(smma,Shape<_64,_16>{})')

start = s.index('    { // Consumers split the bias tile')
end = s.index('    // The other dS slot', start)
s = s[:start] + '''    { // Both WG counts use the same packed-pair Q16 partial layout.
      int64_t base=(((int64_t(h)*(L/R)+group)*nt+jt)*nkt+kt)*1024;
      #pragma unroll
      for(int local=0;local<8/C::NWG;local+=2){
        int x=wg*(8/C::NWG)+local;
        int key=(lane/32)*16+(lane%32)/4+((x%4)/2)*8;
        int query=(lane%4)*2+(x/4)*8;
        int off=as_position_independent_swizzle_layout(typename C::SL{})(make_coord(key,query))*2;
        float sum0=0.f,sum1=0.f;
        #pragma unroll
        for(int w=0;w<C::NWG;++w){
          #pragma unroll
          for(int rr=0;rr<C::RP;++rr){
            uint32_t pair,ptr=cast_smem_ptr_to_uint(s.ds[w][bs][rr].data())+off;
            asm volatile("ld.shared.b32 %0,[%1];":"=r"(pair):"r"(ptr):"memory");
            sum0+=float(Element::bitcast(uint16_t(pair)));
            sum1+=float(Element::bitcast(uint16_t(pair>>16)));
          }
        }
        reinterpret_cast<float2*>(p.db+base)[(x/2)*128+lane]=make_float2(sum0,sum1);
      }
    }
''' + s[end:]
start = s.index('__global__ void reduce_bias(')
end = s.index('template<int R> std::vector<torch::Tensor> launch', start)
s = s[:start] + '''__global__ void reduce_bias(float const* part,Element* out,int L,int groups){
  int h=blockIdx.z,nt=L/16,nkt=L/64;
  float acc[4]={};
  for(int g=0;g<groups;++g){
    int64_t base=(((int64_t(h)*groups+g)*nt+blockIdx.y)*nkt+blockIdx.x)*256+threadIdx.x;
    float4 v=reinterpret_cast<float4 const*>(part)[base];
    acc[0]+=v.x;acc[1]+=v.y;acc[2]+=v.z;acc[3]+=v.w;
  }
  #pragma unroll
  for(int c=0;c<4;++c){
    int idx=threadIdx.x*4+c,lane=(idx/2)%128,x=(idx/256)*2+(idx%2);
    int k=blockIdx.x*64+(lane/32)*16+(lane%32)/4+((x%4)/2)*8;
    int q=blockIdx.y*16+(lane%4)*2+(x%2)+(x/4)*8;
    out[(h*L+q)*L+k]=Element(acc[c]);
  }
}
__global__ void reduce_bias_small(float const* part,Element* out,int L,int groups){
  int h=blockIdx.z,nt=L/16,nkt=L/64,jt=blockIdx.y/2,chunk=blockIdx.y%2;
  float a=0.f,b=0.f;
  for(int g=0;g<groups;++g){
    int64_t base=(((int64_t(h)*groups+g)*nt+jt)*nkt+blockIdx.x)*512+chunk*256+threadIdx.x;
    float2 v=reinterpret_cast<float2 const*>(part)[base];a+=v.x;b+=v.y;
  }
  #pragma unroll
  for(int c=0;c<2;++c){
    int idx=chunk*512+threadIdx.x*2+c,lane=(idx/2)%128,x=(idx/256)*2+(idx%2);
    int k=blockIdx.x*64+(lane/32)*16+(lane%32)/4+((x%4)/2)*8;
    int q=jt*16+(lane%4)*2+(x%2)+(x/4)*8;
    out[(h*L+q)*L+k]=Element(c?b:a);
  }
}

''' + s[end:]
s = s.replace('make_q(q,_32{})', 'make_q(q,_16{})').replace('make_q(dout,_32{})', 'make_q(dout,_16{})')
s = s.replace('else reduce_bias<<<dim3(L/64,L/32,4)', 'else reduce_bias<<<dim3(L/64,L/16,4)')

for name, nwg, registers in [('rs8_q16_two', 2, 232), ('rs8_q16_four', 4, 112)]:
    t = s.replace('NWG=2, RP=R/NWG', 'NWG=%d, RP=R/NWG' % nwg)
    threads = (nwg + 1) * 128
    t = t.replace('__launch_bounds__(384,1)', '__launch_bounds__(%d,1)' % threads)
    t = t.replace('warpgroup_reg_alloc<232>()', 'warpgroup_reg_alloc<%d>()' % registers)
    t = t.replace('NamedBarrier::sync(256,5)', 'NamedBarrier::sync(%d,5)' % (nwg * 128))
    t = t.replace('dim3(L/64,L/R,4),384,', 'dim3(L/64,L/R,4),%d,' % threads)
    t = t.replace('// One producer warpgroup and two consumer warpgroups; each consumer owns four outer rows.',
                  '// Q16; one producer WG and %d consumer WGs, %d rows per consumer.' % (nwg, 8 // nwg))
    target = root / name
    target.mkdir(exist_ok=True)
    (target / 'grouped.cu').write_text(t)

t = (root / 'rs8_q16_four/grouped.cu').read_text()
t = t.replace('__launch_bounds__(640,1)', '__launch_bounds__(512,1)').replace('wg=tid/128-1', 'wg=tid/128')
start = t.index('  if(tid<128) {')
end = t.index('  typename C::ScoreMMA', start)
t = t[:start] + '''  auto load_query=[&](int it){
    int jt=it/C::RP,rr=it%C::RP,bs=jt%2,bphase=(jt/2)%2,slot=it%2,phase=(it/2)%2;
    auto shape=make_shape(L,_32{},_4{},L);
    if(wg==0 && rr==0){
      s.b_empty[bs].wait(bphase^1);s.b_full[bs].arrive_and_expect_tx(1024*sizeof(Element));
      auto mb=p.bias.get_tma_tensor(make_shape(L,L,_4{}));
      auto gb=local_tile(mb(_,_,h),Shape<_16,_64>{},make_coord(jt,kt));
      tma_copy(p.bias,gb,make_tensor(make_smem_ptr(s.bias[bs].data()),typename C::BL{}),s.b_full[bs]);
    }
    int row=group*R+wg*C::RP+rr;
    s.q_empty[wg][slot].wait(phase^1);
    s.q_full[wg][slot].arrive_and_expect_tx(2*512*sizeof(Element)+2*16*sizeof(float));
    auto mq=p.q.get_tma_tensor(shape);auto md=p.dout.get_tma_tensor(shape);
    auto gq=local_tile(mq(_,_,h,row),Shape<_16,_32>{},make_coord(jt,0));
    auto gd=local_tile(md(_,_,h,row),Shape<_16,_32>{},make_coord(jt,0));
    tma_copy(p.q,gq,make_tensor(make_smem_ptr(s.q[wg][slot].data()),typename C::QL{}),s.q_full[wg][slot]);
    tma_copy(p.dout,gd,make_tensor(make_smem_ptr(s.dout[wg][slot].data()),typename C::QL{}),s.q_full[wg][slot]);
    int stat=(h*L+row)*L+jt*16;
    SM90_BULK_COPY_G2S::copy(p.lse+stat,reinterpret_cast<uint64_t*>(&s.q_full[wg][slot]),s.lse[wg][slot].data(),16*sizeof(float));
    SM90_BULK_COPY_G2S::copy(p.delta+stat,reinterpret_cast<uint64_t*>(&s.q_full[wg][slot]),s.delta[wg][slot].data(),16*sizeof(float));
  };
  if(lane==0){
    auto shape=make_shape(L,_32{},_4{},L);
    auto mk=p.k.get_tma_tensor(shape);auto mv=p.v.get_tma_tensor(shape);
    s.kv_full.arrive_and_expect_tx(C::RP*2*2048*sizeof(Element));
    for(int rr=0;rr<C::RP;++rr){
      int r=wg*C::RP+rr;
      auto gk=local_tile(mk(_,_,h,group*R+r),Shape<_64,_32>{},make_coord(kt,0));
      auto gv=local_tile(mv(_,_,h,group*R+r),Shape<_64,_32>{},make_coord(kt,0));
      tma_copy(p.k,gk,make_tensor(make_smem_ptr(s.k[r].data()),typename C::KL{}),s.kv_full);
      tma_copy(p.v,gv,make_tensor(make_smem_ptr(s.v[r].data()),typename C::KL{}),s.kv_full);
    }
    load_query(0);
  }
''' + t[end:]
needle = '      int it=jt*C::RP+rr,slot=it%2,phase=(it/2)%2;'
assert t.count(needle) == 1
t = t.replace(needle, needle + '\n      if(lane==0 && it+1<nt*C::RP)load_query(it+1);')
t = t.replace('dim3(L/64,L/R,4),640,', 'dim3(L/64,L/R,4),512,')
t = t.replace('// Q16; one producer WG and 4 consumer WGs, 2 rows per consumer.',
              '// Q16; four cooperative WGs with independent TMA prefetch, two rows each.')
target = root / 'rs8_q16_coop'
target.mkdir(exist_ok=True)
(target / 'grouped.cu').write_text(t)

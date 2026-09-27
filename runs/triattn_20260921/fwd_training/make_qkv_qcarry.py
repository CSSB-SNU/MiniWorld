"""Carry the last projected Q into the first attention iteration using shared memory."""
from pathlib import Path
R = Path(__file__).resolve().parent


def transform(s):
    s = s.replace('int tid=threadIdx.x,', 'constexpr bool CarryQ=Consumers==Projectors;\n  int tid=threadIdx.x,', 1)
    old = '''    if(lane==0)asm volatile("cp.async.bulk.wait_group 0;":::"memory");
  }
  // QKV saves are complete. The projection scratch may become bias storage.
  __syncthreads();
'''
    new = '''    if constexpr(!CarryQ) {
      if(lane==0)asm volatile("cp.async.bulk.wait_group 0;":::"memory");
    }
  }
  // Each WG snapshots its last Q before any WG reuses projection scratch.
  uint32_t carry[8];
  if constexpr(CarryQ) {
    if(c<nt) {
      auto src=reinterpret_cast<uint32_t const*>(s.scratch.proj.z[c].data());
      #pragma unroll
      for(int i=0;i<8;++i)carry[i]=src[lane+i*128];
    }
  }
  __syncthreads();
  if constexpr(CarryQ) {
    if(c<nt) {
      auto dst=reinterpret_cast<uint32_t*>(s.scratch.attn.q[c].data());
      #pragma unroll
      for(int i=0;i<8;++i)dst[lane+i*128]=carry[i];
      cutlass::arch::fence_view_async_shared();
      cutlass::arch::NamedBarrier::sync(128,c+1);
    }
  }
'''
    assert old in s
    s = s.replace(old, new, 1)
    s = s.replace('for(int qt=c;qt<nt;qt+=Consumers) {', '''for(int qi=0;qi<(nt-c+Consumers-1)/Consumers;++qi) {
    int qt=CarryQ ? c+((nt-1-c)/Consumers-qi)*Consumers : c+qi*Consumers;''', 1)
    begin = s.index('    if(lane==0) {\n      auto qg=')
    end = s.index('    auto score=', begin)
    s = s[:begin] + '''    if(lane==0) {
      if(!CarryQ || qi>0) {
        if constexpr(CarryQ) {
          // This projector owns every Q consumed by this WG. Complete its
          // saved destinations before reloading the remaining query tiles.
          if(qi==1)asm volatile("cp.async.bulk.wait_group 0;":::"memory");
        }
        auto qg=p.loadq.get_tma_tensor(make_shape(L,_32{},_4{},L));
        auto qq=local_tile(qg(_,_,h,row),Shape<_64,_32>{},make_coord(qt,0));
        s.qfull[c].arrive_and_expect_tx(2048*sizeof(Element));
        tma_load(p.loadq,qq,make_tensor(make_smem_ptr(s.scratch.attn.q[c].data()),typename C::QL{}),s.qfull[c]);
      }
      for(int b=0;b<Stages && b<nt;++b)load_bias(b);
    }
    if(!CarryQ || qi>0)s.qfull[c].wait((qi-(CarryQ?1:0))%2);
''' + s[end:]
    s = s.replace('((qt/Consumers)*(nt/Stages)+kt/Stages)%2', '(qi*(nt/Stages)+kt/Stages)%2')
    needle = '''  }
}

template<int Capacity,int Consumers,int Stages,int Projectors>
void launch'''
    assert needle in s
    s = s.replace(needle, '''  }
  if constexpr(CarryQ) {
    if(lane==0)asm volatile("cp.async.bulk.wait_group 0;":::"memory");
  }
}

template<int Capacity,int Consumers,int Stages,int Projectors>
void launch''', 1)
    return s


for source, target in [('qkv_compact_alias', 'qkv_compact_qcarry'),
                       ('qkv_compact_retire6', 'qkv_compact_qcarry_retire')]:
    d=R/target
    assert not (d/'build-ready.json').exists()
    d.mkdir(exist_ok=True)
    (d/'fused.cu').write_text(transform((R/source/'fused.cu').read_text()))

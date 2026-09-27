from pathlib import Path

root=Path(__file__).resolve().parent.parent/'bias_fusion'
s=(root/'rs_tma_store/grouped.cu').read_text()
s=s.replace('    // dBias reduction reads', '    array_aligned<float,512,128> partial[NWG];\n    // dBias reduction reads')
old='      reinterpret_cast<float4*>(p.db+base)[wg*128+lane]=make_float4(values[0],values[1],values[2],values[3]);'
new='''      reinterpret_cast<float4*>(s.partial[wg].data())[lane]=make_float4(values[0],values[1],values[2],values[3]);
      cutlass::arch::fence_view_async_shared();cutlass::arch::NamedBarrier::sync(128,wg+1);
      if(lane==0){
        SM90_BULK_COPY_S2G::copy(s.partial[wg].data(),p.db+base+wg*512,512*sizeof(float));
        tma_store_arrive();tma_store_wait<0>();
      }
      // The elected issuer has finished reading the shared buffer before its
      // warpgroup can produce the next tile (the loop's WG barrier joins it).
'''
assert old in s;s=s.replace(old,new)
d=root/'rs_partial_tma';d.mkdir(exist_ok=True);(d/'grouped.cu').write_text(s)

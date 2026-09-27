"""Generate a separately named direct STSM proof, including the K-to-V gap."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=(HERE/'projection.cu').read_text()
a=s.index(' auto id=mt.partition_C(')
b=s.index('\n}\n\n__global__',a)
s=s[:a]+''' auto converted=make_tensor<Element>(shape(acc));
 #pragma unroll
 for(int n=0;n<size(acc);n++)converted(n)=Element(acc(n));
 auto copyop=make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N,Element>{},mma);
 auto thread_copy=copyop.get_slice(tid);
 auto src=thread_copy.retile_S(converted);
 if constexpr(Q){
  auto out=local_tile(make_tensor(make_smem_ptr(s.q[r]),SO{}),Shape<_64,_32>{},make_coord(hh,0));
  copy(copyop,src,thread_copy.partition_D(out));
 }else{
  using Full=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_128,_32,Int<Stages*R*2>>{}));
  // Logical N[0:32] maps into K, N[32:64] into V at the whole K-array gap.
  auto mapping=make_layout(Shape<_128,Shape<_32,_2>>{},Stride<_1,Stride<_128,Int<Stages*R*M*D>>>{});
  auto both=make_tensor(make_smem_ptr(s.k[0][r]),composition(Full{},mapping));
  auto out=local_tile(both,Shape<_64,_64>{},make_coord(hh,0));
  copy(copyop,src,thread_copy.partition_D(out));
 }
'''+s[b:]
s=s.replace('triattn_projected_proof','triattn_projected_stsm').replace('projected_qkv_cuda','projected_stsm_qkv_cuda')
(HERE/'projection_stsm.cu').write_text(s)

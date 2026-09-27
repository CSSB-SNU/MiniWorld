#include <cute/tensor.hpp>
#include <cstdio>
using namespace cute;
using Element=cutlass::bfloat16_t;
template<class L> int check(L l,int M,int N) {
 alignas(1024) Element buf[4096];
 auto t=make_tensor(make_smem_ptr(buf),l);
 auto pos=as_position_independent_swizzle_layout(l);
 int bad=0;
 for(int m=0;m<M;m++)for(int n=0;n<N;n++) bad+=(&t(m,n)-buf)!=pos(make_coord(m,n));
 printf("%dx%d mismatches=%d\n",M,N,bad);return bad;
}
int main(){return check(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_64,_32>{}),64,32)+check(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_32,_64>{}),32,64)+check(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_64>{}),64,64);}

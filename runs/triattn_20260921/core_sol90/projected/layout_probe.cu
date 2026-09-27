#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cstdio>
using namespace cute;
int main(){
using Elt=cutlass::bfloat16_t;
auto a=tile_to_shape(GMMA::Layout_K_SW128_Atom<Elt>{},Shape<_32,_128>{});
auto b=tile_to_shape(GMMA::Layout_K_SW128_Atom<Elt>{},Shape<_64,_128>{});
print(a);puts("");print(b);puts("");
for(int n: {0,1,8,16,31,32,63})for(int k: {0,1,32,64,96,127})printf("n%d k%d a%d b%d\n",n,k,int(a(n%32,k))+(n/32)*4096,int(b(n,k)));
}

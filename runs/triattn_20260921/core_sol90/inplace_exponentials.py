"""Keep each row's scale/exponentials in the original accumulator operands."""
def transform(s):
 old='''   #pragma unroll
   for(int col=0;col<N/4;col++){
    float x=sr(row,col);
    if constexpr(Safe)sr(row,col)=x==-INFINITY?0.f:ex2(fmaf(x,c,-mx[hh][row]));
    else sr(row,col)=ex2(fmaf(x,c,-mx[hh][row]));
   }'''
 assert s.count(old)==1
 instructions=''.join('fma.rn.ftz.f32 %%%d,%%%d,%%8,%%9;\\n'%(i,i) for i in range(8))
 instructions+=''.join('ex2.approx.ftz.f32 %%%d,%%%d;\\n'%(i,i) for i in range(8))
 operands=','.join('"+f"(sr(row,%d))'%i for i in range(8))
 new='''   if constexpr(Safe){
    #pragma unroll
    for(int col=0;col<N/4;col++){
     float x=sr(row,col);sr(row,col)=x==-INFINITY?0.f:ex2(fmaf(x,c,-mx[hh][row]));
    }
   }else{
    static_assert(N==32);
    asm volatile("'''+instructions+'" : '+operands+' : "f"(c),"f"(-mx[hh][row]));\n   }'
 return s.replace(old,new)

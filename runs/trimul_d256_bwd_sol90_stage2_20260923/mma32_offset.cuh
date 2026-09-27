template<int OA,int OB,int TA,int TB> TMN_DEVI void mma32_off(float (&v)[16],uint64_t aa,uint64_t bb,int ac){
 asm volatile("{.reg .pred p;.reg .b64 ax,bx;add.u64 ax,%16,%21;add.u64 bx,%17,%22;setp.ne.b32 p,%18,0;wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},ax,bx,p,1,1,%19,%20;}" : "+f"(v[0]),"+f"(v[1]),"+f"(v[2]),"+f"(v[3]),"+f"(v[4]),"+f"(v[5]),"+f"(v[6]),"+f"(v[7]),"+f"(v[8]),"+f"(v[9]),"+f"(v[10]),"+f"(v[11]),"+f"(v[12]),"+f"(v[13]),"+f"(v[14]),"+f"(v[15]) : "l"(aa),"l"(bb),"r"(ac),"n"(TA),"n"(TB),"n"(OA>>4),"n"(OB>>4));
}

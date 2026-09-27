// Reconstruct rounded projection from the existing rounded forward AB.
// Two-bit codes select unchanged, +1 BF16 bit, -1 BF16 bit, or exact escape.
TMN_DEVI uint32_t ab_projection_estimate(uint32_t ab,float gate,uint32_t mask){
 float den=__fmul_rn(gate,__uint_as_float(mask<<16));
 float value=__fmul_rn(__uint_as_float(ab<<16),math::rcp_approx_ftz(den));
 return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}
TMN_DEVI uint32_t ab_projection_decode(uint32_t ab,float gate,uint32_t mask,uint32_t code,const bf* escape){
 if(code==3)return __bfloat16_as_ushort(*escape);
 // For a zero mask, any finite signed projection of the same sign gives
 // the identical rounded masked derivative, including its sign of zero.
 if((mask&0x7fffu)==0)return (ab^mask)&0x8000u;
 return (ab_projection_estimate(ab,gate,mask)+(code==1?1u:code==2?0xffffu:0u))&0xffffu;
}

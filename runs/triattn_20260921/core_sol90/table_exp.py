"""Isolated approximation: quarter/half logits through a shared exp2 table."""
def transform(name,s,fraction):
 if name!='triattn_m1_sm90.cuh':return s
 marker='        int bad;'
 assert s.count(marker)==1
 s=s.replace(marker,'        uint32_t exp_table[kSafe ? 1 : 1024];\n'+marker)
 marker='    if (wg_idx != 0) {\n        // the all-ones tile'
 assert s.count(marker)==1
 s=s.replace(marker,'''    if constexpr(!kSafe){
        for(int idx=tid;idx<1024;idx+=T::kNumThreads)
            shared.exp_table[idx]=__float_as_uint(ex2_approx(float(idx)*0x1p-10f));
    }
'''+marker)
 pos=s.index('template <class T>',s.index('__device__ __forceinline__ float ex2_approx'))
 helper='''__device__ __forceinline__ float ex2_table(float x,uint32_t const* table){
 uint32_t i=__float_as_uint(fmaf(x,1024.f,12582912.f));
 uint32_t bits=table[i&1023u]+((i<<13)&0xff800000u);
 float y=__uint_as_float(bits);
 return x < -126.f ? 0.f : (x >= 128.f ? INFINITY : y);
}
'''
 s=s[:pos]+helper+s[pos:]
 old='s_rc(mi, ni) = ex2_approx(fmaf(s_rc(mi, ni), c_l2, nm[hh][mi]));'
 new=f'''if constexpr(!kSafe && (false)) {{}} // replaced below
                if constexpr(!kSafe) {{
                    if(ni%4<{fraction}) s_rc(mi,ni)=ex2_table(fmaf(s_rc(mi,ni),c_l2,nm[hh][mi]),shared.exp_table);
                    else {{ {old} }}
                }} else {{ {old} }}'''
 new=new.replace('if constexpr(!kSafe && (false)) {} // replaced below\n                ','')
 assert s.count(old)==1
 return s.replace(old,new)

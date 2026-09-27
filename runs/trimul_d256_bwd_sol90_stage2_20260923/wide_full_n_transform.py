"""Use one N192/N256 dW instruction, preserving each output's K order."""
import re

def full_n_dw(body,n):
    assert n in (192,256)
    nr=n//2
    registers=','.join(f'%{i}' for i in range(nr))
    constraints=','.join(f'"+f"(v[{i}])' for i in range(nr))
    helper=f'''
template<int OA,int OB> TMN_DEVI void mma_full_n(float (&v)[{nr}],uint64_t aa,uint64_t bb,int ac){{
 asm volatile("{{.reg .pred p;.reg .b64 ax,bx;add.u64 ax,%{nr},%{nr+3};add.u64 bx,%{nr+1},%{nr+4};setp.ne.b32 p,%{nr+2},0;wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{registers}}},ax,bx,p,1,1,0,1;}}" : {constraints} : "l"(aa),"l"(bb),"r"(ac),"n"(OA>>4),"n"(OB>>4));
}}
'''
    point='template<int WG> TMN_DEVI void pipe_consume'
    assert body.count(point)==1;body=body.replace(point,helper+'\n'+point)
    pattern=r'static_for<NC>\(\[&\]\(auto cc\)\{constexpr int c=decltype\(cc\)::value;\s*mma64_off<k\*32,c\*8192\+k\*2048,0,1>\(dw\[c\],(smem_desc\(smem_u32\([^;]+?),it>0\|\|k>0\);\s*\}\);'
    body,count=re.subn(pattern,lambda m:f'mma_full_n<k*32,k*2048>(*reinterpret_cast<float(*)[{nr}]>(dw),'+m.group(1)+',it>0||k>0);',body)
    assert count==1,count
    return body

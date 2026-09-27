"""Projection operands in registers; consumers keep their shared dW inputs."""


def registers(body,d):
    begin=body.index('  float pre[32]={};')
    end=body.index('  });wgmma_commit();',begin)+len('  });')
    body=body[:begin]+'''
  uint32_t fa[D/16][4];load_frag_bf16<D/16,8192>(fa,smem_u32(xn),warp*16,lane);
  float pre[32]={};fence_regs(pre);wgmma_fence();
  uint64_t desc=smem_desc(smem_u32(sm+WEIGHT),16,1024,1);uint32_t lo=desc,hi=desc>>32;
  static_for<D/16>([&](auto kk){constexpr int k=decltype(kk)::value;
   wgmma_m64n64k16_rs_off<(k/4)*8192+(k%4)*32>(pre,fa[k],lo,hi,k>0);
  });'''+body[end:]
    producer,consumer=(160,168) if d==384 else (184,160)
    body=body.replace('setmaxnreg_dec<96>()',f'setmaxnreg_inc<{producer}>()') if producer>168 else body.replace('setmaxnreg_dec<96>()',f'setmaxnreg_dec<{producer}>()')
    # Stay within the 168-register initial allocation of 384 threads:
    # 496/504 registers across roles, at most 64512 registers per CTA.
    body=body.replace('setmaxnreg_inc<200>()',f'setmaxnreg_{"inc" if consumer>168 else "dec"}<{consumer}>()')
    body=body.replace('mw_wide_pipe_source','mw_wide_rs_source')
    return body

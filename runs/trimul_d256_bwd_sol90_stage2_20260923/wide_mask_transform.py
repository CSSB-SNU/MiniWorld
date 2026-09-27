"""Overlap source mask loads with the input/compute pipeline."""


def mask_stage(body,mode):
    assert mode in ('prefetch','bulk')
    if mode=='prefetch':
        begin=body.index('  int ra=warp*16+lane/4;')
        end=body.index('  packed_glu(',begin)
        load=body[begin:end];body=body[:begin]+body[end:]
        point='  float pre[32]={};'
        assert body.count(point)==1
        body=body.replace(point,load+point)
    else:
        point=' mbar_arrive_expect_tx(bar+slot,INPUT);'
        assert body.count(point)==1
        body=body.replace(point,point.replace('INPUT','INPUT+128')+'''
 asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],128,[%2];"::"r"(smem_u32(sm+BAR+128+slot*128)),"l"(p.mask+row),"r"(smem_u32(bar+slot)):"memory");''')
        body=body.replace('p.mask[row+ra]','reinterpret_cast<bf*>(sm+BAR+128+slot*128)[ra]')
        body=body.replace('p.mask[row+ra+8]','reinterpret_cast<bf*>(sm+BAR+128+slot*128)[ra+8]')
    return body.replace('mw_wide_pipe_source','mw_wide_mask_source')

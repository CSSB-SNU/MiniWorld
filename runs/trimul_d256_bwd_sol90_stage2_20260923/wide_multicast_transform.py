"""Reuse the engine's D128 multicast protocol for shared wide source inputs."""


def multicast(body):
    body=body.replace('#include "tmn_kernels.cuh"','#include "tmn_kernels.cuh"\n#include <cooperative_groups.h>')
    start=body.index('TMN_DEVI void load_input(')
    end=body.index('TMN_DEVI void store_gp(',start)
    body=body[:start]+'''
TMN_DEVI uint32_t remote_addr(const void* ptr,int rank){uint32_t v;asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(v):"r"(smem_u32(ptr)),"r"(rank));return v;}
TMN_DEVI void signal_peer(uint64_t* bar,int peer){asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _,[%0];"::"r"(remote_addr(bar,peer)):"memory");}
TMN_DEVI void cluster_wait(uint64_t* bar,int phase){uint32_t v;do{asm volatile("{.reg .pred P;mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P,[%1],%2;selp.u32 %0,1,0,P;}":"=r"(v):"r"(smem_u32(bar)),"r"(phase):"memory");}while(!v);}
TMN_DEVI void multicast_xn(void* dst,const CUtensorMap* map,uint64_t* bar,int c,int row){
 asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0],[%1,{%3,%4}],[%2],%5;"::"r"(smem_u32(dst)),"l"(map),"r"(smem_u32(bar)),"r"(c),"r"(row),"h"(uint16_t((1<<WIDE_SOURCE_CLUSTER)-1)):"memory");
}
TMN_DEVI void load_input(const Params& p,uint8_t* sm,uint64_t* bar,int row,int rank,int slot){
 int begin=(p.M/64)*(blockIdx.x/RANKS)/WEIGHT_SPLITS,it=row/64-begin;
 // Both local consumers have released this input slot before reaching here.
 // Announce transaction bytes on every peer before the leader multicasts.
 mbar_arrive_expect_tx(bar+slot,INPUT);
 tma_load_2d(sm+slot*INPUT+CH,rank<H/32?&p.dl:&p.dr,bar+slot,row,(rank%(H/32))*32);
 signal_peer(bar+7+slot,0);
 if(rank%WIDE_SOURCE_CLUSTER==0){
  cluster_wait(bar+7+slot,(it/2)&1);
  for(int c=0;c<D/64;++c)multicast_xn(sm+slot*INPUT+c*8192,&p.xn,bar+slot,c*64,row);
 }
}
'''+body[end:]
    body=body.replace('mw_wide_pipe_source','mw_wide_multicast_source')
    marker='for(int b=0;b<7;++b)mbar_init(bar+b,b>=5?2:1);'
    assert body.count(marker)==1
    body=body.replace(marker,'for(int b=0;b<9;++b)mbar_init(bar+b,b>=7?WIDE_SOURCE_CLUSTER:b>=5?2:1);')
    marker=' __syncthreads();\n if(threadIdx.x<128)'
    assert body.count(marker)==1
    body=body.replace(marker,' __syncthreads();cooperative_groups::this_cluster().sync();\n if(threadIdx.x<128)')
    end=body.rfind('\n}')
    body=body[:end]+'\n cooperative_groups::this_cluster().sync();'+body[end:]
    return body

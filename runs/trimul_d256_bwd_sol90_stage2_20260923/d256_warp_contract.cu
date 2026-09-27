#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
constexpr int CHANNELS=32,TM=16,TN=64,TK=16,AS=TM*TK*2,BS=TN*TK*2,INPUT=CHANNELS*(AS+BS),BAR=2*INPUT;
struct Params {CUtensorMap a[4],b[4];bf *dl,*dr;int N;};
TMN_DEVI void ld2(uint32_t (&v)[2],uint32_t addr){asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];":"=r"(v[0]),"=r"(v[1]):"r"(addr));}
TMN_DEVI void ld2t(uint32_t (&v)[2],uint32_t addr){asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];":"=r"(v[0]),"=r"(v[1]):"r"(addr));}
TMN_DEVI void mma(float (&d)[4],const uint32_t (&a)[4],const uint32_t (&b)[2]){
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
template<int MODE> TMN_DEVI void load(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi,int ni,int ki,int slot){
 constexpr bool TA=MODE==1,TB=MODE!=2;
 int t=threadIdx.x;
 if(t==0)mbar_arrive_expect_tx(bar+slot,INPUT);
 __syncthreads();
 if(t<CHANNELS){int c=ch+t;
  tma_load_2d(sm+slot*INPUT+t*AS,p.a+MODE,bar+slot,TA?mi:ki,c*p.N+(TA?ki:mi));
  tma_load_2d(sm+slot*INPUT+CHANNELS*AS+t*BS,p.b+MODE,bar+slot,TB?ni:ki,c*p.N+(TB?ki:ni));
 }
}
template<int MODE> TMN_DEVI void run(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi,int ni){
 constexpr bool TA=MODE==1,TB=MODE!=2;
 int lane=threadIdx.x%32,warp=threadIdx.x/32;float acc[4][8][4]={};
 load<MODE>(p,sm,bar,ch,mi,ni,0,0);
 for(int ki=0,it=0;ki<p.N;ki+=TK,++it){
  int slot=it&1;mbar_wait(bar+slot,(it/2)&1);__syncthreads();
  if(ki+TK<p.N)load<MODE>(p,sm,bar,ch,mi,ni,ki+TK,slot^1);
  #pragma unroll
  for(int cc=0;cc<4;++cc){int c=warp+cc*8;
   uint8_t* sa=sm+slot*INPUT+c*AS;uint8_t* sb=sm+slot*INPUT+CHANNELS*AS+c*BS;
   uint32_t a[4];
   if constexpr(TA){int r=(lane&7)+8*(lane>>4),col=8*((lane>>3)&1);ldsm_x4_t(a,smem_u32(sa+2*(r*TM+col)));}
   else {int r=(lane&7)+8*((lane>>3)&1),col=8*(lane>>4);ldsm_x4(a,smem_u32(sa+2*(r*TK+col)));}
   #pragma unroll
   for(int q=0;q<8;++q){uint32_t b[2];
    if constexpr(TB){int r=(lane&7)+8*((lane>>3)&1);ld2t(b,smem_u32(sb+2*(r*TN+q*8)));}
    else {int r=q*8+(lane&7),col=8*((lane>>3)&1);ld2(b,smem_u32(sb+2*(r*TK+col)));}
    mma(acc[cc][q],a,b);
   }
  }
  __syncthreads();
 }
 int m=lane/4,n=2*(lane%4);bf* dst=(MODE==1||MODE==3)?p.dr:p.dl;
 #pragma unroll
 for(int cc=0;cc<4;++cc){int c=ch+warp+cc*8+(MODE>=2?256:0);
  #pragma unroll
  for(int q=0;q<8;++q){size_t at=size_t(c)*p.N*p.N+size_t(mi+m)*p.N+ni+q*8+n;
   *reinterpret_cast<uint32_t*>(dst+at)=pack_bf16(acc[cc][q][0],acc[cc][q][1]);
   *reinterpret_cast<uint32_t*>(dst+at+8*p.N)=pack_bf16(acc[cc][q][2],acc[cc][q][3]);
  }
 }
}
extern "C" __global__ __launch_bounds__(256,1)
void mw_d256_warp_contract(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 int tiles=(p.N/TM)*(p.N/TN),group=blockIdx.x/tiles,mode=group/8,ch=(group%8)*32,tile=blockIdx.x%tiles;
 int mi=(tile/(p.N/TN))*TM,ni=(tile%(p.N/TN))*TN;
 if(threadIdx.x==0){mbar_init(bar,1);mbar_init(bar+1,1);fence_barrier_init();}__syncthreads();
 if(mode==0)run<0>(p,sm,bar,ch,mi,ni);
 else if(mode==1)run<1>(p,sm,bar,ch,mi,ni);
 else if(mode==2)run<2>(p,sm,bar,ch,mi,ni);
 else run<3>(p,sm,bar,ch,mi,ni);
}

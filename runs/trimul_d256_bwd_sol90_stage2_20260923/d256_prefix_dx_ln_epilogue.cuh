  // The completed dXn tile remains in shared memory until LN consumes it.
  #if EMIT_DXN
  for(int i=tid;i<64*256;i+=128){int r=i/256,c=i%256;
   p.dxn[size_t(blockIdx.x*128+WG*64+r)*256+c]=*reinterpret_cast<bf*>(out+(c/64)*8192+swz128(r,(c%64)*2));
  }
  #endif
  int ct=WG*128+tid;reinterpret_cast<float*>(sm+98304)[ct]=p.gamma[ct];
  named_bar_sync(3,256);
  uint8_t* scratch=sm+65536+WG*16384;
  auto read16=[&](uint8_t* s,int r,int c){return __bfloat162float(*reinterpret_cast<bf*>(s+(c/64)*2048+swz128(r,(c%64)*2)));};
  auto read64=[&](int r,int c){return __bfloat162float(*reinterpret_cast<bf*>(out+(c/64)*8192+swz128(r,(c%64)*2)));};
  float gg[8]={},bb[8]={};
  for(int half=0;half<4;++half){
   int row=blockIdx.x*128+WG*64+half*16;
   if(tid==0){
    mbar_arrive_expect_tx(bars+2*NSLOT+WG,16384);
    tma_load_3d(scratch,&p.x,bars+2*NSLOT+WG,0,row,0);
    tma_load_3d(scratch+8192,&p.res,bars+2*NSLOT+WG,0,row,0);
   }
   mbar_wait(bars+2*NSLOT+WG,half&1);named_bar_sync(WG+1,128);
   for(int r=warp;r<16;r+=4){
    float xv[8],s=0;
    #pragma unroll
    for(int q=0;q<8;++q){xv[q]=read16(scratch,r,lane+q*32);s+=xv[q];}
    float mu=sumwarp(s)/256;s=0;
    #pragma unroll
    for(int q=0;q<8;++q){float z=xv[q]-mu;s+=z*z;}
    float rs=rsqrtf(sumwarp(s)/256+1e-5f),s0=0,s1=0;
    #pragma unroll
    for(int q=0;q<8;++q){int c=lane+q*32;float z=(xv[q]-mu)*rs,dy=read64(half*16+r,c),v=dy*reinterpret_cast<float*>(sm+98304)[c];
     s0+=v;s1+=v*z;gg[q]+=dy*z;bb[q]+=dy;
    }
    s0=sumwarp(s0)/256;s1=sumwarp(s1)/256;
    #pragma unroll
    for(int q=0;q<8;++q){int c=lane+q*32;float z=(xv[q]-mu)*rs,v=read64(half*16+r,c)*reinterpret_cast<float*>(sm+98304)[c];
     float dx=(v-s0-z*s1)*rs;
     *reinterpret_cast<bf*>(out+(c/64)*8192+swz128(half*16+r,(c%64)*2))=__float2bfloat16_rn(dx+read16(scratch+8192,r,c));
    }
   }
   named_bar_sync(WG+1,128);
  }
  float* local=reinterpret_cast<float*>(scratch);
  #pragma unroll
  for(int q=0;q<8;++q){int c=lane+q*32;local[warp*256+c]=gg[q];local[(4+warp)*256+c]=bb[q];}
  named_bar_sync(WG+1,128);
  for(int c=tid;c<256;c+=128){float g=0,b=0;
   #pragma unroll
   for(int w=0;w<4;++w){g+=local[w*256+c];b+=local[(4+w)*256+c];}
   p.partial[(blockIdx.x*2+WG)*512+c]=g;p.partial[(blockIdx.x*2+WG)*512+256+c]=b;
  }

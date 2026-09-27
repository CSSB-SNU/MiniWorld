"""Keep two dW WGMMA groups in flight before returning input slot credit."""


def async_consumers(body):
    # Preload both inputs; publishing GP1 must precede the wait for slot0 credit.
    initial='if(threadIdx.x==0)load_input(p,sm,bar,begin*64,rank,0);'
    assert body.count(initial)==1
    body=body.replace(initial,initial+'\n if(threadIdx.x==0 && begin+1<end)load_input(p,sm,bar,(begin+1)*64,rank,1);')
    prefetch='''  if(tile+1<end){
   if(it>=1)mbar_wait(bar+5+(1-slot),((it-1)/2)&1);
   if(tid==0)load_input(p,sm,bar,(tile+1)*64,rank,1-slot);
  }
'''
    assert body.count(prefetch)==1
    body=body.replace(prefetch,'')
    marker='''  named_bar_sync(1,128);
 }
}
template<int WG>'''
    replacement='''  named_bar_sync(1,128);
  if(it>=1 && tile+1<end){
   mbar_wait(bar+5+(1-slot),((it-1)/2)&1);
   if(tid==0)load_input(p,sm,bar,(tile+1)*64,rank,1-slot);
  }
 }
}
template<int WG>'''
    assert body.count(marker)==1
    body=body.replace(marker,replacement)
    # CPU does not access the accumulators until all outstanding groups finish.
    fence='  static_for<NC>([&](auto cc){constexpr int c=decltype(cc)::value;fence_regs(dw[c]);});wgmma_fence();'
    assert body.count(fence)==1
    body=body.replace(fence,'')
    marker=' float dw[NC][32]={};'
    assert body.count(marker)==1
    body=body.replace(marker,marker+'\n'+fence)
    marker='''  });wgmma_commit();wgmma_wait<0>();
  static_for<NC>([&](auto cc){constexpr int c=decltype(cc)::value;fence_regs(dw[c]);});
  named_bar_sync(2+WG,128);if(tid==0)mbar_arrive(bar+5+slot);
 }
 static_for<NC>'''
    replacement='''  });wgmma_commit();
  if(it>0){
   wgmma_wait<1>();named_bar_sync(2+WG,128);
   if(tid==0)mbar_arrive(bar+5+(1-slot));
  }
 }
 wgmma_wait<0>();
 static_for<NC>([&](auto cc){constexpr int c=decltype(cc)::value;fence_regs(dw[c]);});
 named_bar_sync(2+WG,128);
 if(tid==0 && end>begin)mbar_arrive(bar+5+(end-begin-1)%2);
 static_for<NC>'''
    assert body.count(marker)==1
    return body.replace(marker,replacement).replace('mw_wide_mask_source','mw_wide_async_source')

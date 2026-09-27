  // Four scores, two P fragments; QK(k+2), E(k), PV(k-1), bias(k+3).
  // The first four QKs are drained so each query half can seed from 32 keys.
  clear(pp);clear(pp1);
  init_score(scores[0],0);issue_qk(scores[0],0,Int<0%H>{});
  init_score(scores[1],1);issue_qk(scores[1],1,Int<1%H>{});
  init_score(scores[2],2);issue_qk(scores[2],2,Int<2%H>{});
  init_score(scores[3],3);issue_qk(scores[3],3,Int<3%H>{});
  drain();
  auto seed_pair=[&](auto& a,auto& b,auto half) __attribute__((always_inline)) {
    constexpr int hh=decltype(half)::value;
    auto ar=make_tensor(a.data(),flash::convert_layout_acc_rowcol(a.layout()));
    auto br=make_tensor(b.data(),flash::convert_layout_acc_rowcol(b.layout()));
    #pragma unroll
    for(int row=0;row<2;++row) {
      float m=ar(row,0);
      #pragma unroll
      for(int col=1;col<N/4;++col)m=fmaxf(m,ar(row,col));
      #pragma unroll
      for(int col=0;col<N/4;++col)m=fmaxf(m,br(row,col));
      m=fmaxf(m,__shfl_xor_sync(0xffffffffu,m,1));
      m=fmaxf(m,__shfl_xor_sync(0xffffffffu,m,2));
      if(m!=-INFINITY)mx[hh][row]=m*c+64.f;else missing_seed=true;
    }
  };
  if constexpr(H==1)seed_pair(scores[0],scores[1],_0{});
  else {seed_pair(scores[0],scores[2],_0{});seed_pair(scores[1],scores[3],_1{});}
  exponentiate(scores[0],Int<0%H>{},cute::false_type{});
  pack(scores[0],pp);issue_pv(0,Int<0%H>{},pp);
  exponentiate(scores[1],Int<1%H>{},cute::false_type{});
  init_score(scores[0],4);
  auto step=[&](auto seqc) __attribute__((always_inline)) {
    constexpr int seq=decltype(seqc)::value;
    constexpr int cur=seq%4,future=(seq+2)%4,previous=(seq+3)%4;
    auto& prob=(seq-1)%2==0?pp:pp1;
    // No phantom tail QK: the final body has only three prior groups.
    // wait1 there retires PV(seq-3) before its P slot is reused.
    if constexpr(seq==768/N*H-1)warpgroup_wait<1>();
    else warpgroup_wait<2>();
    asm volatile("":::"memory");
    warpgroup_fence_operand(scores[cur]);warpgroup_fence_operand(prob);
    release_k(seq);if constexpr(seq>=3)release(seq-3);
    pack(scores[previous],prob);warpgroup_fence_operand(prob);
    if constexpr(seq+2<768/N*H)issue_qk(scores[future],seq+2,Int<(seq+2)%H>{});
    exponentiate(scores[cur],Int<seq%H>{},cute::false_type{});
    // Explicit fence is retained: compiler placement, not source order,
    // determines whether the packed P was written before QK's fence.
    issue_pv(seq-1,Int<(seq-1)%H>{},prob);
    if constexpr(seq+3<768/N*H)init_score(scores[previous],seq+3);
  };
  // STATIC_STEPS
  drain();
  pack(scores[3],pp1);issue_pv(768/N*H-1,Int<(768/N*H-1)%H>{},pp1);
  drain();release(768/N*H-1);

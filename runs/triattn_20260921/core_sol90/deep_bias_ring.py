"""Independent bias ring depth for the exact fixed-shape hot pipeline."""
def transform(s,slots=6,narrow_guard=False):
    assert slots in (4,6)
    def rep(a,b,count=1):
        nonlocal s
        assert s.count(a)==count,(a,s.count(a),count)
        s=s.replace(a,b)
    rep('kSlotsB = kBlockN / CW;', 'kSlotsB = (kFlags_ & 1073741824) ? %d : kBlockN / CW;'%slots)
    rep('static_assert(kChunksPerTile == 8 && kSlotsB == 4);', 'static_assert(kChunksPerTile == 8 && (kSlotsB == 4 || kSlotsB == 6));')
    a=s.index('            for (int j = 0; j < n_tiles; ++j) {',s.index('// ---- the bias stream:'))
    b=s.index('\n        }\n        if constexpr (!kList)',a)
    old=s[a:b]
    s=s[:a]+'''            if constexpr(kFast){
                uint32_t phase=0;
                for(int base=0;base<8*n_tiles;base+=T::kHalves,phase^=1u){
                    #pragma unroll
                    for(int mm=0;mm<T::kHalves;++mm){
                        if(base+mm<8*n_tiles){
                            pipe_b.producer_wait_empty(mm,phase);
                            pipe_b.producer_expect(mm>>1,T::kBytesHalf);
                            copy(params.tma_b.with(*pipe_b.full_barrier(mm>>1),b_mask),tBgB(_,blk0+base+mm),tBsB(mm));
                        }
                    }
                }
            }else{
''' + old + '\n            }' + s[b:]
    # Prologue chunks0..4 have the same slots for both depths. The body uses
    # a stream-relative sequence index, independent of the K/V ring's tiles.
    a=s.index('    auto issue_qk =')
    s=s[:a]+'''    auto init_bias_stream = [&](AccC& acc,int seq) __attribute__((always_inline)) {
        if(seq<8*n_w){
            int const slot=(seq/2)%T::kSlotsB,hh=seq&1;
            #pragma unroll
            for(int u=0;u<4;++u){
                float4 x=*reinterpret_cast<float4 const*>(bias_thread+slot*T::kSlotElems+(hh*4+u)*512);
                acc(4*u)=x.x;acc(4*u+1)=x.y;acc(4*u+2)=x.z;acc(4*u+3)=x.w;
            }
        }else{
            // Unused lookahead past the stream must not read never-filled
            // extra slots when a short mask leaves fewer than six columns.
            clear(acc);
        }
    };
''' + s[a:]
    a=s.index('    auto body =');b=s.index('    // drained state:',a)
    block=s[a:b]
    block=block.replace('bias_release_by(e2 & 7, leadk2, dep)', 'bias_release_by(kFast?((16*p+e2)%T::kHalves):(e2&7),leadk2,dep)')
    block=block.replace('bias_release(e2 & 7, dep)', 'bias_release(kFast?((16*p+e2)%T::kHalves):(e2&7),dep)')
    old='uint32_t const bph3 = kReplayOps ? 0u : uint32_t(t3 & 1);'
    assert block.count(old)==1
    block=block.replace(old,'''int const bslot3=kFast?((8*p+e3/2)%T::kSlotsB):c3;
        uint32_t const bph3 = kFast?uint32_t(((8*p+e3/2)/T::kSlotsB)&1):(kReplayOps ? 0u : uint32_t(t3 & 1));''')
    block=block.replace('pipe_b.test_full(c3, bph3)', 'pipe_b.test_full(bslot3,bph3)')
    block=block.replace('pipe_b.wait_full(c3, bph3)', 'pipe_b.wait_full(bslot3,bph3)')
    old='init_chunk(accC[bp], Int<c3>{}, Int<h3>{});'
    assert block.count(old)==1
    block=block.replace(old,'if constexpr(kFast)init_bias_stream(accC[bp],16*p+e3);else '+old)
    s=s[:a]+block+s[b:]
    a=s.index('    int p = 0; uint32_t pp = 0;',s.index('    auto period ='))
    b=s.index('    {   // last chunk',a)
    old=s[a:b]
    # Three compile-time periods cover L768. This makes the six-slot bias
    # phase/addresses constants while retaining every original early exit.
    s=s[:a]+'''    if constexpr(kFast){
        if(!period(cute::false_type{},0,0u)){
            if(!period(cute::false_type{},1,1u)){
                bool const ended=period(cute::false_type{},2,0u);(void)ended;
            }
        }
    }else{
''' + old + '    }\n' + s[b:]
    if narrow_guard:
        # Every physical slot is initialized once n_tiles>=2. For a one-tile
        # stream, only first-cycle half slots8..11 can be uninitialized; the
        # stream exits before any later-period lookahead. Four-slot rings need
        # no guard because even one tile fills every physical slot.
        rep('if(seq<8*n_w){', 'if(!(T::kSlotsB==6 && seq>=8 && seq<12 && n_w==1)){')
    return s

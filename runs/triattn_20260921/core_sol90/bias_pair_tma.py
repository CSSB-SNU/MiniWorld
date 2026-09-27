"""One16KB TMA and one empty notification per pair of64-query bias halves."""
def transform(name,s):
 if name=='triattn_m1_sm90.cuh':
  mark='    using ShapeB  ='
  assert mark in s
  s=s.replace(mark,'''    static constexpr bool kPairBias = !kSafe && (kFlags & 1073741824);
    using BiasLoadTile = std::conditional_t<kPairBias,Shape<_256,_8,_2>,Shape<_256,_8>>;
    using SmemLayoutBiasLoad = std::conditional_t<kPairBias,
        Layout<Shape<_256,_8,_2>,Stride<_1,_256,_2048>>,SmemLayoutBiasHalf>;
'''+mark)
  old='SmemLayoutBiasHalf{}, make_shape(_256{}, _8{}), Int<CR>{}));'
  assert s.count(old)==1
  s=s.replace(old,'SmemLayoutBiasLoad{}, BiasLoadTile{}, Int<CR>{}));')
  s=s.replace('T::PipeB::init(shared.pipe_b, T::kArrivalsB, 2);','T::PipeB::init(shared.pipe_b, T::kArrivalsB, kFast ? 1 : 2);')
  a=s.index('            // ---- the bias stream:')
  b=s.index('\n        }\n        if constexpr (!kList) { break; }',a)
  old=s[a:b]
  s=s[:a]+'''            if constexpr(kFast){
                auto gB=params.tma_b.get_tma_tensor(params.shape_b)(_,_,_,qtile,bh);
                auto sl=params.tma_b.get_slice(_0{});
                auto src=group_modes<0,3>(sl.partition_S(gB));
                for(int j=0;j<n_tiles;++j){
                    uint32_t phase=j&1;
                    #pragma unroll
                    for(int col=0;col<4;++col){
                        pipe_b.producer_wait_empty(2*col,phase);
                        pipe_b.producer_expect(col,2*T::kBytesHalf);
                        auto sb=make_tensor(make_smem_ptr(shared.smem_bias.data()+col*T::kSlotElems),typename T::SmemLayoutBiasLoad{});
                        auto dst=group_modes<0,4>(sl.partition_D(sb));
                        copy(params.tma_b.with(*pipe_b.full_barrier(col),0),src(_,kc0+4*j+col),dst);
                    }
                }
            }else{
'''+old+'''
            }'''+s[b:]
  a=s.index('        asm volatile("" ::: "memory"); __syncwarp();',s.index('auto bias_release_by ='))
  b=s.index('\n    };',a)
  old=s[a:b]
  s=s[:a]+'''        if constexpr(kFast){
            // The odd QK has consumed this warp's bias registers for both
            // halves. Every consumer warp still sends its own notification.
            if(c&1){asm volatile("" ::: "memory");__syncwarp();pipe_b.release(c-1,leader);}
            (void)dep;
        }else{
'''+old+'''
        }'''+s[b:]
 elif name=='launch_m1.cuh':
  old='typename T::SmemLayoutBiasHalf{}, make_shape(_256{}, _8{}), Int<T::CR>{});'
  assert s.count(old)==1
  s=s.replace(old,'typename T::SmemLayoutBiasLoad{}, typename T::BiasLoadTile{}, Int<T::CR>{});')
 return s

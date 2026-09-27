"""Experimental 24-bit bias transport; retain original FP32 for SAFE."""
def transform(name,s):
 if name=='m1_binding.cu':
  s=s.replace('float* out = dst + ((int64_t(bh) * nq + qt) * nkc + blockIdx.x) * 4096;', 'float* tile_base = dst + (int64_t(bh) * nq + qt) * nkc * 7168;\n    float* out = tile_base + blockIdx.x * 4096;\n    uint32_t* packed_out = reinterpret_cast<uint32_t*>(tile_base + nkc * 4096 + blockIdx.x * 3072);')
  s=s.replace('        out[idx] = val;', '''        out[idx] = val; // exact FP32 source for SAFE
        uint32_t v=__float_as_uint(val)>>8;
        uint32_t v0=__shfl_sync(0xffffffffu,v,0,4),v1=__shfl_sync(0xffffffffu,v,1,4);
        uint32_t v2=__shfl_sync(0xffffffffu,v,2,4),v3=__shfl_sync(0xffffffffu,v,3,4);
        uint32_t w=e==0?(v0|(v1<<24)):(e==1?((v1>>8)|(v2<<16)):((v2>>16)|(v3<<8)));
        int pos=3*u+e;
        if(e<3)packed_out[hh*1536+(pos/4)*512+t*4+pos%4]=w;''')
  s=s.replace('nq, W4, 4096}', 'nq, W4, 7168}')
 if name=='triattn_m1_sm90.cuh':
  s=s.replace('static constexpr int kSlotElems = 2 * 4 * 128 * 4,', 'static constexpr int kBiasRows = kSafe ? 8 : 6;\n    static constexpr int kSlotElems = 2 * 256 * kBiasRows,')
  s=s.replace('Layout<Shape<_256, _8>, Stride<_1, _256>>', 'Layout<Shape<_256, Int<kBiasRows>>, Stride<_1, _256>>')
  s=s.replace('SmemLayoutBiasHalf{}, make_shape(_256{}, _8{})', 'SmemLayoutBiasHalf{}, make_shape(_256{}, Int<kBiasRows>{})')
  a=s.index('    auto init_chunk =');b=s.index('    auto issue_qk =',a)
  old=s[a:b]
  mark='''        } else {
            #pragma unroll
            for (int u = 0; u < 4; ++u) {'''
  assert old.count(mark)==1
  new='''        } else if constexpr (!kSafe) {
            uint32_t w[12];
            #pragma unroll
            for(int j=0;j<3;j++){
                uint4 a=*reinterpret_cast<uint4 const*>(bias_thread+c*T::kSlotElems+hh*1536+j*512);
                w[4*j]=a.x;w[4*j+1]=a.y;w[4*j+2]=a.z;w[4*j+3]=a.w;
            }
            #pragma unroll
            for(int j=0;j<4;j++){
                uint32_t a=w[3*j],b=w[3*j+1],c_=w[3*j+2];
                acc(4*j)=__uint_as_float(a<<8);
                acc(4*j+1)=__uint_as_float(__byte_perm(a,b,0x5430)&0xffffff00u);
                acc(4*j+2)=__uint_as_float(__byte_perm(b,c_,0x4320)&0xffffff00u);
                acc(4*j+3)=__uint_as_float(c_&0xffffff00u);
            }
        } else {
            #pragma unroll
            for (int u = 0; u < 4; ++u) {'''
  old=old.replace(mark,new);s=s[:a]+old+s[b:]
 if name=='launch_m1.cuh':
  s=s.replace('a.bias.size(3) == T::kSlotElems', 'a.bias.size(3) == 7168')
  s=s.replace('make_shape(256, 8, 2 * n_kcol', 'make_shape(256, T::kBiasRows, 2 * n_kcol')
  s=s.replace('int64_t(T::kSlotElems) * n_kcol', 'int64_t(7168) * n_kcol')
  s=s.replace('make_gmem_ptr(a.bias.data_ptr<float>())', 'make_gmem_ptr(a.bias.data_ptr<float>() + (T::kSafe ? 0 : 4096 * n_kcol))')
  s=s.replace('typename T::SmemLayoutBiasHalf{}, make_shape(_256{}, _8{})', 'typename T::SmemLayoutBiasHalf{}, make_shape(_256{}, Int<T::kBiasRows>{})')
 return s

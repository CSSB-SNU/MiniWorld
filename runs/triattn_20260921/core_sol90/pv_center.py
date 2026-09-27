"""Experimental centered-V FP16 PV; native BF16/FP32 SAFE remains available."""
from pv_half import transform as half_transform

def transform(name,s):
    s=half_transform(name,s,True)
    def rep(a,b,count=1):
        nonlocal s
        assert s.count(a)==count,(name,a,s.count(a),count)
        s=s.replace(a,b)
    if name=='triattn_m1_sm90.cuh':
        rep('int const* vbad;', 'int const* vbad;\n        float const* vmean;')
        rep('o_rc(mi, ni) *= f;', '''if constexpr(T::kHalfPV)o_rc(mi,ni)=cutlass::half_t(float(o_rc(mi,ni))*f);
                        else o_rc(mi,ni)*=f;''',2)
        rep('''__nv_bfloat162 v2 = __floats2bfloat162_rn(o_rc(mi, ni) * inv, o_rc(mi, ni + 1) * inv);''', '''float out0=float(o_rc(mi,ni))*inv,out1=float(o_rc(mi,ni+1))*inv;
                        if constexpr(T::kHalfPV){
                            auto mean=reinterpret_cast<float2 const*>(params.vmean+((int64_t(b)*768+i)*4+h)*32+d);
                            float2 mu=*mean;out0+=mu.x;out1+=mu.y;
                        }
                        __nv_bfloat162 v2 = __floats2bfloat162_rn(out0,out1);''')
    elif name=='launch_m1.cuh':
        rep('int const* vbad = nullptr;', 'int const* vbad = nullptr;\n    float const* vmean = nullptr;')
        rep('a.rowkc0, a.rowkc1, a.vbad};', 'a.rowkc0, a.rowkc1, a.vbad, a.vmean};')
    elif name=='m1_binding.cu':
        a=s.index('__global__ void convert_pv_half(')
        b=s.index('\nstruct Entry {',a)
        s=s[:a]+'''__global__ void convert_pv_half(__nv_bfloat16 const* v,int64_t vb,int64_t vn,int64_t vh,int64_t vs,
                               __half* out,int* bad,float* means,int N,int H,int S){
    int row=blockIdx.x,h=row%H,n=(row/H)%N,b=row/(N*H);
    int d=threadIdx.x%32,g=threadIdx.x/32;
    __shared__ float part[8][32],mu[32];
    float sum=0.f;
    for(int k=g;k<S;k+=8)sum+=__bfloat162float(v[int64_t(b)*vb+int64_t(n)*vn+int64_t(h)*vh+int64_t(k)*vs+d]);
    part[g][d]=sum;__syncthreads();
    if(g==0){
        float total=0.f;
        #pragma unroll
        for(int r=0;r<8;++r)total+=part[r][d];
        mu[d]=total/float(S);means[int64_t(row)*32+d]=mu[d];
    }
    __syncthreads();
    bool failed=!isfinite(mu[d]);
    for(int k=g;k<S;k+=8){
        float f=__bfloat162float(v[int64_t(b)*vb+int64_t(n)*vn+int64_t(h)*vh+int64_t(k)*vs+d])-mu[d];
        __half y=__float2half_rn(f);out[(int64_t(row)*S+k)*32+d]=y;
        failed|=!isfinite(f)||!isfinite(__half2float(y));
    }
    int any=__syncthreads_or(failed);
    if(threadIdx.x==0)bad[row]=any;
}
''' + s[b:]
        rep('torch::Tensor vhalf,vbad;', 'torch::Tensor vhalf,vbad,vmean;')
        line='vbad=torch::empty({q.size(0)*q.size(1)*q.size(2)},q.options().dtype(torch::kInt32));'
        rep(line,line+'\n        vmean=torch::empty({vbad.numel(),32},q.options().dtype(torch::kFloat32));')
        rep('vbad.data_ptr<int>(),q.size(1),q.size(2),q.size(3));', 'vbad.data_ptr<int>(),vmean.data_ptr<float>(),q.size(1),q.size(2),q.size(3));')
        rep('a.vhalf=&vhalf;a.vbad=vbad.data_ptr<int>();', 'a.vhalf=&vhalf;a.vbad=vbad.data_ptr<int>();a.vmean=vmean.data_ptr<float>();')
    return s

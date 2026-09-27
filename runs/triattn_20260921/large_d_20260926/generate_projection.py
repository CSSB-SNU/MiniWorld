"""Keep N128/K64 WGMMA tiles; grow global C and loop over its K tiles."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parent
source=ROOT.parent.parent/'trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/triangle_attention/cuda/projection_dgrad.cu'
s=source.read_text()
def rep(a,b,n=None):
    global s
    assert s.count(a) and (n is None or s.count(a)==n),(a,s.count(a),n)
    s=s.replace(a,b)
rep('using namespace cute;', 'using namespace cute;\n#ifndef PAIR_DIM\n#define PAIR_DIM 256\n#endif\nusing Width=Int<PAIR_DIM>;')
rep('#include <cutlass/arch/barrier.h>','#include <cutlass/arch/barrier.h>\n#include <cutlass/arch/reg_reconfig.h>')
rep('make_shape(int32_t{},_128{}),make_stride(_128{},_1{})','make_shape(int32_t{},Width{}),make_stride(Width{},_1{})')
rep('Shape<_128,_128>{},Stride<_1,_128>{}', 'Shape<Width,Width>{},Stride<_1,Width>{}')
rep('it<8','it<4*(PAIR_DIM/64)')
rep('proj=it/2,kt=it%2','proj=it/(PAIR_DIM/64),kt=it%(PAIR_DIM/64)')
rep('make_shape(p.rows,_128{})','make_shape(p.rows,Width{})')
rep('get_tma_tensor(Shape<_128,_128>{})','get_tma_tensor(Shape<Width,Width>{})')
rep('make_coord(0,kt)','make_coord(blockIdx.y,kt)')
rep('col=get<1>(coord(i))','col=blockIdx.y*128+get<1>(coord(i))')
rep('p.wb[h*128+col]','p.wb[h*PAIR_DIM+col]')
rep('p.dx[row*128+col]','p.dx[row*PAIR_DIM+col]')
rep('dy[0].numel()/128','dy[0].numel()/PAIR_DIM')
rep('{rows,128}','{rows,PAIR_DIM}')
rep('make_shape(rows,_128{}),make_stride(_128{},_1{})','make_shape(rows,Width{}),make_stride(Width{},_1{})')
rep('<<<(rows+M-1)/M,256','<<<dim3((rows+M-1)/M,PAIR_DIM/128),256')
rep('j<4?128:4','j<4?PAIR_DIM:4')
rep('{j<4?PAIR_DIM:4,128}','{j<4?PAIR_DIM:4,PAIR_DIM}')
rep('projection_dgrad_tma','wide_projection_dgrad_tma')
# Producer does not need the consumer accumulator register budget.
rep('  if(tid<128) {','  if(tid<128) {\n    cutlass::arch::warpgroup_reg_dealloc<32>();',1)
rep('  typename C::MMA mma;','  cutlass::arch::warpgroup_reg_alloc<192>();\n  typename C::MMA mma;',1)
for C in (256,512):
    p=ROOT/f'projection_C{C}';p.mkdir(exist_ok=True)
    (p/'projection.cu').write_text(s)
    (p/'source.json').write_text(json.dumps(dict(origin=str(source),origin_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),sha256=hashlib.sha256(s.encode()).hexdigest()),indent=2)+'\n')

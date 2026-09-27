"""Derive isolated wide CUDA sources from qualified head32 kernels.

Dimension edits are explicit. Attention tile dimensions stay M64/N64 (FWD/dQ)
and K64/Q32 (dK/dV); only the feature axis grows. Never edit the serving tree.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent
HEADER = '''\n#ifndef HEAD_DIM
#define HEAD_DIM 64
#endif
#define WIDTH (4*HEAD_DIM)
#define ROW_GROUP (HEAD_DIM==64 ? 4 : 2)
using Head=cute::Int<HEAD_DIM>;
'''


def replace(source, old, new, count=None):
    actual = source.count(old)
    assert actual and (count is None or actual == count), (old, actual, count)
    return source.replace(old, new)


def common(source):
    source = replace(source, 'using namespace cute;', 'using namespace cute;' + HEADER, 1)
    source = replace(source, 'SCALE=0.1767766952966369f',
                     'SCALE=(HEAD_DIM==64 ? 0.125f : 0.08838834764831845f)', 1)
    for old, new in [('q.size(4)==32','q.size(4)==HEAD_DIM'),
                     ('B1 H4 D32 required','B1 H4 matching wide head required'),
                     ('{1,L,L,128}','{1,L,L,WIDTH}'), ('{1,L,L,4,32}','{1,L,L,4,HEAD_DIM}')]:
        source = replace(source, old, new)
    source = source.replace('.stride(1)==32', '.stride(1)==HEAD_DIM')
    source = source.replace('.stride(3)==128', '.stride(3)==WIDTH')
    source = source.replace('.stride(2)==L*128', '.stride(2)==L*WIDTH')
    return source


fwd_path = RUN / 'fwd_training/cooperative_head2/fused.cu'
dq_path = RUN / 'bwd/dq/reuse_qdo/fused.cu'
kv_path = RUN / 'bwd/bias_fusion/rs8_async_q4/grouped.cu'
fwd = common(fwd_path.read_text())
fwd = fwd.replace('_32', 'Head')
fwd = replace(fwd, '2048', '(64*HEAD_DIM)')
fwd = replace(fwd, 'QStride{128,_1{},Head{},int64_t(L)*128}',
              'QStride{WIDTH,_1{},Head{},int64_t(L)*WIDTH}')
fwd = replace(fwd, '__launch_bounds__(128,5)', '__launch_bounds__(128,2)')
fwd = fwd.replace('training_fwd_stream', 'wide_training_fwd')

dq = common(dq_path.read_text())
dq = dq.replace('_32', 'Head')
dq = replace(dq, '2048', '(64*HEAD_DIM)')
dq = replace(dq, 'QStride{128,_1{},Head{},int64_t(L)*128}',
             'QStride{WIDTH,_1{},Head{},int64_t(L)*WIDTH}')
dq = replace(dq, '__launch_bounds__(128,4)', '__launch_bounds__(128,2)')
dq = replace(dq, 'array_aligned<Element,(64*HEAD_DIM),1024> q,dout,k[2],v[2];\n  array_aligned<Element,4096,1024> bias[2];', '''
  // Q/dO retire into registers before the first bias TMA. Their shared storage
  // is then reused by the two bias stages, and by the final dQ store afterward.
  union ResidentBias {
    struct Resident { array_aligned<Element,(64*HEAD_DIM),1024> q,dout; } resident;
    array_aligned<Element,4096,1024> bias[2];
  } resident_bias;
  array_aligned<Element,(64*HEAD_DIM),1024> k[2],v[2];''', 1)
dq = dq.replace('s.q.data()', 's.resident_bias.resident.q.data()')
dq = dq.replace('s.dout.data()', 's.resident_bias.resident.dout.data()')
dq = dq.replace('s.bias[', 's.resident_bias.bias[')
dq = replace(dq, '   load(0);', '   // Bias aliases Q/dO: defer its first load until register copies retire.', 1)
dq = replace(dq, 'static_assert(size(qr_words)==8)', 'static_assert(size(qr_words)==HEAD_DIM/4)', 1)
dq = replace(dq, 'static_assert(size(dr_words)==8)', 'static_assert(size(dr_words)==HEAD_DIM/4)', 1)
dq = replace(dq, 'slice<2', 'slice<HEAD_DIM/16', 2)
dq = replace(dq, ' uint32_t mp=cast_smem_ptr_to_uint(s.lse.data())',
             ' __syncthreads(); // Every ldmatrix reader retires before bias overwrites Q/dO.\n'
             ' if(tid==0)load(0);\n uint32_t mp=cast_smem_ptr_to_uint(s.lse.data())', 1)
dq = dq.replace('dq_tma_reuse_qdo', 'wide_dq_resident_alias')

kv = common(kv_path.read_text())
kv = replace(kv, 'NWG=2, RP=R/NWG, QS=4', 'NWG=2, RP=R/NWG, QS=(HEAD_DIM==64 ? 4 : 2)', 1)
kv = replace(kv, 'static_assert(R==8)', 'static_assert(R==ROW_GROUP)', 1)
for old, new in [
    ('using KL=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_64,_32>{}));',
     'using KL=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_64,Head>{}));'),
    ('using QL=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_32,_32>{}));',
     'using QL=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_32,Head>{}));'),
    ('using QT=decltype(tile_to_shape(GMMA::Layout_MN_SW64_Atom<Element>{},Shape<_32,_32>{}));',
     'using QT=decltype(tile_to_shape(GMMA::Layout_MN_SW64_Atom<Element>{},Shape<Head,_32>{}));'),
    ('Shape<int32_t,_32,_4,int32_t>', 'Shape<int32_t,Head,_4,int32_t>'),
    ('Stride<int64_t,_1,_32,int64_t>', 'Stride<int64_t,_1,Head,int64_t>'),
    ('KL{},Shape<_64,_32>', 'KL{},Shape<_64,Head>'),
    ('QL{},Shape<_32,_32>', 'QL{},Shape<_32,Head>'),
    ('float,Shape<_64,_32,_32>>()', 'float,Shape<_64,_32,Head>>()'),
    ('float,Shape<_64,_32,_32>,GMMA', 'float,Shape<_64,Head,_32>,GMMA'),
    ('array_aligned<Element,2048,1024> k[R],v[R]', 'array_aligned<Element,64*HEAD_DIM,1024> k[R],v[R]'),
    ('array_aligned<Element,1024,1024> q[NWG][QS],dout[NWG][QS]', 'array_aligned<Element,32*HEAD_DIM,1024> q[NWG][QS],dout[NWG][QS]'),
    ('make_shape(L,_32{},_4{},L)', 'make_shape(L,Head{},_4{},L)'),
    ('C::RP*2*2048*sizeof(Element)', 'C::RP*2*64*HEAD_DIM*sizeof(Element)'),
    ('2*1024*sizeof(Element)+2*32*sizeof(float)', '2*32*HEAD_DIM*sizeof(Element)+2*32*sizeof(float)'),
    ('Shape<_64,_32>{},make_coord(kt,0)', 'Shape<_64,Head>{},make_coord(kt,0)'),
    ('Shape<_32,_32>{},make_coord(jt,0)', 'Shape<_32,Head>{},make_coord(jt,0)'),
    ('partition_fragment_C(gmma,Shape<_64,_32>{})', 'partition_fragment_C(gmma,Shape<_64,Head>{})'),
    ('gt.partition_C(make_identity_tensor(Shape<_64,_32>{}))', 'gt.partition_C(make_identity_tensor(Shape<_64,Head>{}))'),
    ('Shape<decltype(tile_rows),_32>', 'Shape<decltype(tile_rows),Head>'),
    ('StrideQ{x.stride(3),_1{},_32{},x.stride(2)}', 'StrideQ{x.stride(3),_1{},Head{},x.stride(2)}'),
    ('StrideQ{128,_1{},_32{},int64_t(L)*128}', 'StrideQ{WIDTH,_1{},Head{},int64_t(L)*WIDTH}'),
    ('R==4 || R==8,"legacy group4 ABI or native group8 required"', 'R==ROW_GROUP,"matching wide row group required"'),
    ('return launch<8>', 'return launch<ROW_GROUP>'),
    ('m.attr("row_group")=8', 'm.attr("row_group")=ROW_GROUP'),
    ('sizeof(Config<8>::Shared)', 'sizeof(Config<ROW_GROUP>::Shared)')]:
    kv = replace(kv, old, new)
kv = kv.replace('grouped_dkdv', 'wide_grouped_dkdv')

sources = dict(fwd=fwd, dq=dq, dkdv=kv)
origins = dict(fwd=fwd_path, dq=dq_path, dkdv=kv_path)
for hd in (64,128):
    folder = ROOT / f'native_head{hd}'
    folder.mkdir(exist_ok=True)
    for name, source in sources.items():
        (folder / f'{name}.cu').write_text(source)
    manifest = dict(head_dim=hd, row_group=4 if hd==64 else 2,
                    origin={name: dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for name,p in origins.items()},
                    source_sha256={name+'.cu':hashlib.sha256(source.encode()).hexdigest() for name,source in sources.items()})
    (folder / 'sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('GENERATED',folder)

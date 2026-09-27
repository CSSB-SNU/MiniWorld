"""Issue each row's startup Q TMA from its consumer WG before mask derivation."""
from pathlib import Path
import hashlib,importlib.util,os
HERE=Path(__file__).resolve().parent
src=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src/'triattn_broadcast.so').read_bytes()).hexdigest()=='6d52ebfc4b43ca8fcbeac7de165d4572cf30d9cd2efe6ee6f63508425dc284fe'
variant='baseqbootstrap';dst=HERE/variant;dst.mkdir(exist_ok=True)
for p in src.rglob('*'):
 if not p.is_file() or p.suffix not in ('.py','.cu','.cuh','.h'):continue
 rel=p.relative_to(src)
 if '__pycache__' in rel.parts or str(rel) in ('__init__.py','build_native.py'):continue
 if str(rel)=='build_source.py':rel=Path('triattn_m1.py')
 s=p.read_text().replace('ta_core_broadcast','ta_sol_'+variant).replace('triattn_broadcast','triattn_sol_'+variant)
 if p.name=='triattn_m1_sm90.cuh':
  assert s.count('shared.barrier_q.init(1);')==1
  s=s.replace('shared.barrier_q.init(1);','shared.barrier_q.init(kFast ? R : 1);')
  a=s.index('        auto issue_q =');b=s.index('        if constexpr (kMaskInKernelC)',a)
  block=s[a:b];old='if (warp_idx_in_wg == 1 && lane_predicate)';assert block.count(old)==1
  block=block.replace(old,'if (!kFast && warp_idx_in_wg == 1 && lane_predicate)')
  s=s[:a]+block+s[b:]
  marker='    derive(rg, b, i0, kind3, kc0, n_tiles, jb_, je_, fr_, dead);';assert s.count(marker)==1
  s=s.replace(marker,'''    if constexpr(kFast){
        if((tid&127)==0){
            int const r=wg_idx-1;
            Tensor sq=make_tensor(make_smem_ptr(shared.smem_q.data()),typename T::SmemLayoutQ{});
            Tensor mq=params.tma_q.get_tma_tensor(params.shape_qk)(_,_,h,_,b);
            Tensor gq=local_tile(mq,make_shape(Int<kBlockM>{},Int<kHeadDim>{}),make_coord(qtile,_0{},_));
            auto sl=params.tma_q.get_slice(_0{});
            Tensor src=group_modes<0,3>(sl.partition_S(gq));
            Tensor dst=group_modes<0,3>(sl.partition_D(sq));
            shared.barrier_q.arrive_and_expect_tx(T::kBytesQ);
            copy(params.tma_q.with(reinterpret_cast<uint64_t&>(shared.barrier_q),0),src(_,rg*R+r),dst(_,r));
        }
    }
'''+marker)
 target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR']=str(HERE/'build');os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
os.environ['CUTLASS_PATH']=str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2');os.environ['TRIATTN_PTXAS']='image'
spec=importlib.util.spec_from_file_location('q_bootstrap_builder',dst/'triattn_m1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m._build(verbose=True)
print('BUILT',variant,flush=True)

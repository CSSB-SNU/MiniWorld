from pathlib import Path
import hashlib,importlib.util,os
from producer_exp_helper import transform, transform_static_finish, transform_explicit_pv_fence, transform_direct_pack, transform_packed_response, transform_wg_queue
HERE=Path(__file__).resolve().parent
src=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src/'triattn_broadcast.so').read_bytes()).hexdigest()=='97634810cdba1e91fe4aecb8bdfa610dd050dd39e7504ffb8babee8c46d4077b'
variant=os.environ.get('VARIANT','baseproducerexp4v3');dst=HERE/variant;dst.mkdir(exist_ok=True)
if variant=='baseproducerexp4v4':transform=transform_static_finish
if variant=='baseproducerexp4v5':transform=transform_explicit_pv_fence
if variant=='baseproducerexp4v6':transform=transform_direct_pack
if variant=='baseproducerexp4v7':transform=transform_packed_response
if variant=='baseproducerexp4v8':transform=transform_wg_queue
if variant=='baseproducerexp4v9':transform=lambda s:transform_wg_queue(s,packed=False)
if variant in ('baseproducerexp3score','baseproducerexp3scoreru0'):
    from producer_three_score import transform
if variant in ('basenative3scoreqr','basenative3scoreqrru0','basenative3scoreqrdep'):
    from producer_three_score import transform as three_score_transform
    transform=lambda s:three_score_transform(s,helper=False,full_q=True,dependent_init=variant.endswith('dep'))
if variant=='basenative3scorelatebias':
    from late_bias_three_score import transform
assert variant in ('baseproducerexp4v3','baseproducerexp4v4','baseproducerexp4v5','baseproducerexp4v6','baseproducerexp4v7','baseproducerexp4v8','baseproducerexp4v9','baseproducerexp3score','baseproducerexp3scoreru0','basenative3scoreqr','basenative3scoreqrru0','basenative3scoreqrdep','basenative3scorelatebias')
for p in src.rglob('*'):
    if not p.is_file() or p.suffix not in ('.py','.cu','.cuh','.h'):continue
    rel=p.relative_to(src)
    if '__pycache__' in rel.parts or str(rel) in ('__init__.py','build_native.py'):continue
    if str(rel)=='build_source.py':rel=Path('triattn_m1.py')
    s=p.read_text().replace('ta_core_broadcast','ta_sol_'+variant).replace('triattn_broadcast','triattn_sol_'+variant)
    if p.name=='build_source.py' and variant.endswith('ru0'):
        marker='    csrc = os.path.join(_HERE, "csrc", "m1")'
        assert s.count(marker)==1
        s=s.replace(marker,'    flags.append("-Xptxas=--register-usage-level=0")\n'+marker)
    if p.name=='triattn_m1_sm90.cuh':s=transform(s)
    target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR']=str(HERE/'build')
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
os.environ['CUTLASS_PATH']=str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2')
os.environ['TRIATTN_PTXAS']='image'
spec=importlib.util.spec_from_file_location('producer_exp_builder',dst/'triattn_m1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m._build(verbose=True)
print('BUILT',variant,flush=True)

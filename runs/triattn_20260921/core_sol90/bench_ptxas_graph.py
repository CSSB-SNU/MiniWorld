"""Replace only a captured hot node with an offline-assembled CUDA kernel.

The original core/block graph, arguments, SAFE node, output and all other
nodes are retained. This is an isolated comparison harness, not serving code.
CUDA 12.9 cuda.h defines the v2 driver node structure and argument ownership.
"""
from pathlib import Path
import ctypes as C,hashlib,json,os,re,statistics,sys
from register_role_gate import check_register_roles
HERE=Path(__file__).resolve().parent;R=HERE.parent
sys.path.insert(0,str(R/'oc'))
ending=os.environ.get('ENDING')=='1'
source=(R/'sweep_pro.py').read_text().split('with torch.no_grad():\n    y = call()')[0]
source=source.replace('starting=True',f'starting={not ending}').replace('ending=False',f'ending={ending}').replace('e.name[:40]','e.name')
exec(compile(source,str(R/'sweep_pro.py'),'exec'))
assert a.length==768

driver=C.CDLL('libcuda.so.1')
P=C.c_void_p;U=C.c_uint;Z=C.c_size_t
class NodeParams(C.Structure):
 _fields_=[('func',P)]+[(s,U) for s in ('gridDimX','gridDimY','gridDimZ','blockDimX','blockDimY','blockDimZ','sharedMemBytes')]+[('kernelParams',C.POINTER(P)),('extra',C.POINTER(P)),('kern',P),('ctx',P)]

def api(name,args):
 fn=getattr(driver,name);fn.argtypes=args;fn.restype=C.c_int
 def run(*values):
  code=fn(*values)
  if code:raise RuntimeError((name,code))
 return run

get_nodes=api('cuGraphGetNodes',[P,C.POINTER(P),C.POINTER(Z)])
get_type=api('cuGraphNodeGetType',[P,C.POINTER(C.c_int)])
get_params=api('cuGraphKernelNodeGetParams_v2',[P,C.POINTER(NodeParams)])
set_params=api('cuGraphKernelNodeSetParams_v2',[P,C.POINTER(NodeParams)])
get_name=api('cuFuncGetName',[C.POINTER(C.c_char_p),P])
get_arg=api('cuFuncGetParamInfo',[P,Z,C.POINTER(Z),C.POINTER(Z)])
load_module=api('cuModuleLoad',[C.POINTER(P),C.c_char_p])
get_function=api('cuModuleGetFunction',[C.POINTER(P),P,C.c_char_p])
set_attribute=api('cuFuncSetAttribute',[P,C.c_int,C.c_int])
def symbol(fn):
 s=C.c_char_p();get_name(C.byref(s),fn);return s.value.decode()

installed=R/'oc/opt_core/kernels/triattn_core_broadcast/triattn_broadcast.so'
def stage(scope,name,where):
 if os.environ.get('CUBIN_DEBUG_STAGE')=='1':print('STAGE',scope,name,where,flush=True)
assert hashlib.sha256(installed.read_bytes()).hexdigest()=='9365a4f35e018afed4adba21590ae2c1734bbccf98d1e53904ebfe2537b3e225'
case_file=os.environ.get('CUBIN_CASES')
cases=json.loads(Path(case_file).read_text()) if case_file else [dict(name=n,screen=str(HERE/'ptxas_current_screen'),config=n) for n in ('ru5','ru0','ru6')]
variant_names=['serving']+[case['name'] for case in cases]
assert len(set(variant_names))==len(variant_names) and cases
functions={};modules=[];metadata={}
for case in cases:
 name=case['name'];screen=Path(case['screen']);config=case['config']
 screens=json.loads((screen/'results.json').read_text())['variants']
 record=next(r for r in screens if r['name']==config)
 assert record['resources']==[0,0,0,128] and not record['warnings']
 path=screen/(config+'.cubin')
 sasstext=(path.with_suffix('.sass')).read_text()
 # A clean ptxas summary does not prove that dynamic role allocation fits.
 # Check actual emitted values; mixed-role callers must declare multiplicity.
 roles=case.get('register_roles',dict(producer=32,consumers=[160,160,160]))
 role_check=check_register_roles(sasstext,roles,record['resources'][3])
 module=P();load_module(C.byref(module),str(path).encode());modules.append(module)
 sym=sasstext.splitlines()[0].strip();assert 'TraitsILi1073741824E' in sym
 fn=P();get_function(C.byref(fn),module,sym.encode());functions[name]=fn
 metadata[name]=dict(binary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),symbol=sym,**role_check)

with torch.no_grad():
 q,k,v,g,b=pf.prologue(x,W,impl='fpf',ending=ending)
 def core_call():return pf.core_attention(q,k,v,b,m5,core=KW['core'])
 graphs={};snapshots={};output_refs={};replacements={}
 for scope,fn in [('core',core_call),('block',call)]:
  graphs[scope]={};snapshots[scope]={};output_refs[scope]={};replacements[scope]={}
  for name in variant_names:
   stage(scope,name,'warmup')
   for _ in range(5):fn()
   torch.cuda.synchronize()
   stage(scope,name,'capture')
   graph=torch.cuda.CUDAGraph(keep_graph=True)
   with torch.cuda.graph(graph):result=fn()
   raw=P(graph.raw_cuda_graph());count=Z();get_nodes(raw,None,C.byref(count))
   nodes=(P*count.value)();get_nodes(raw,nodes,C.byref(count));hits=[]
   for node in nodes:
    kind=C.c_int();get_type(node,C.byref(kind))
    if kind.value!=0:continue
    params=NodeParams();get_params(node,C.byref(params));sym=symbol(params.func)
    if 'ta_core_broadcast17triattn_m1_kernel' in sym and 'TraitsILi1073741824E' in sym:
     hits.append(sym)
     assert (params.blockDimX,params.blockDimY,params.blockDimZ,params.sharedMemBytes)==(512,1,1,191488)
     if name!='serving':
      off,size=Z(),Z();get_arg(params.func,0,C.byref(off),C.byref(size))
      newoff,newsize=Z(),Z();get_arg(functions[name],0,C.byref(newoff),C.byref(newsize))
      assert off.value==newoff.value and size.value==newsize.value and size.value>0
      assert params.kernelParams and not params.extra
      arg=C.create_string_buffer(C.string_at(params.kernelParams[0],size.value))
      args=(P*1)(C.addressof(arg))
      params.func=functions[name].value;params.kernelParams=args
      params.kern=None;params.ctx=None
      set_attribute(params.func,8,params.sharedMemBytes)  # MAX_DYNAMIC_SHARED_SIZE_BYTES
      set_params(node,C.byref(params))
      observed=NodeParams();get_params(node,C.byref(observed));assert observed.func==functions[name].value
      replacements[scope][name]=dict(node_count=count.value,argument_size=size.value,original=sym,replacement=symbol(observed.func),grid=[params.gridDimX,params.gridDimY,params.gridDimZ])
   assert len(hits)==1,(scope,name,hits)
   stage(scope,name,'instantiate');graph.instantiate()
   stage(scope,name,'replay');graph.replay()
   stage(scope,name,'clone');snapshots[scope][name]=result.clone()
   stage(scope,name,'synchronize');torch.cuda.synchronize()
   assert torch.equal(snapshots[scope][name],snapshots[scope]['serving']),(scope,name,'not bitwise')
   graphs[scope][name]=graph;output_refs[scope][name]=result
   print('BITWISE',scope,name,replacements[scope].get(name),flush=True)
 data={scope:{name:[] for name in gg} for scope,gg in graphs.items()}
 for scope,gg in graphs.items():
  for _ in range(30):
   for graph in gg.values():graph.replay()
  torch.cuda.synchronize()
  for rd in range(24):
   order=list(gg);order=order[rd%len(order):]+order[:rd%len(order)]
   if (rd//len(order))%2:order.reverse()
   for name in order:
    for _ in range(3):gg[name].replay()
    st,en=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(20):gg[name].replay()
    en.record();en.synchronize();data[scope][name].append(st.elapsed_time(en)*1000/20)
  # Recheck every output after timing; each clone is distinct.
  after={}
  for name,graph in gg.items():graph.replay();after[name]=output_refs[scope][name].clone()
  torch.cuda.synchronize()
  assert all(torch.equal(v,after['serving']) for v in after.values()),scope
 summary={scope:{name:dict(median_us=statistics.median(times),paired_ratio=statistics.median([v/s for v,s in zip(times,d['serving'])])) for name,times in d.items()} for scope,d in data.items()}
 result=dict(length=a.length,ending=ending,scope='Captured core/block graphs; only hot kernel replaced; full bitwise checks before and after timing',summary=summary,rounds=data,metadata=metadata,replacements=replacements)
 Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print('RESULT',json.dumps(summary),flush=True)

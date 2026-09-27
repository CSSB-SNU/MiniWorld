from pathlib import Path
import json,os
import torch
from lt_contract import LtBmm
from lt_config_search import ConfigSearch
D=512;M=768*768
a=torch.empty((1,M,9*D),device='cuda',dtype=torch.bfloat16).transpose(1,2).contiguous().transpose(1,2)
b=torch.empty((1,9*D,D),device='cuda',dtype=torch.bfloat16)
out=torch.empty((1,M,D),device='cuda',dtype=torch.bfloat16)
workspace=torch.empty(64*1024*1024,device='cuda',dtype=torch.uint8)
op=LtBmm(a,b,out,workspace);search=ConfigSearch(op)
record={'ids':search.ids(),'heuristics':[],'caps':[]}
for index in op.indices:
    algo=op.heuristics[index].algo
    record['heuristics'].append({'index':index,'config':{k:search.get(algo,k,True) for k in range(9)}})
for ident in record['ids']:
    algo=search.init(ident)
    record['caps'].append({'id':ident,'tiles':search.get(algo,6),'stages':search.get(algo,13),
        'custom':search.get(algo,7),'swizzle':search.get(algo,2),'split':search.get(algo,0)})
Path(__file__).with_name(f'lt-config-caps-{os.environ["SLURM_JOB_ID"]}.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
header=Path('/usr/local/cuda-12.9/include/cublasLt.h').read_text().splitlines()
for i,line in enumerate(header):
    if 'CUBLASLT_CLUSTER_SHAPE_' in line or 'CUBLASLT_MATMUL_INNER_SHAPE_' in line or 2440<=i+1<=2455:
        print(i+1,line,flush=True)

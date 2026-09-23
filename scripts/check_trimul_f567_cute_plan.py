"""Verify registry-based build units and the isolated native compile contract."""
import argparse,json
from pathlib import Path
import torch
from miniworld_engine.autotune import native,native_compile
from miniworld_engine.autotune.builder import op_units
from miniworld_engine.autotune.cute_config import f567_candidates
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
op='trimul_output_f567_sm90_cute';units=op_units(only={op});assert units
L=384;M=L*L;kw=dict(dtype=torch.bfloat16)
metas=[torch.empty(*s,**kw) for s in [(M,256),(M,128),(128,256),(128,128),(M,128),(L,128),(128,)]]
limit=torch.cuda.get_device_properties(0).shared_memory_per_block_optin
bucket=native.tensor_key(*metas,extra=(L,limit));grid=f567_candidates(256,128,limit)
assert native.candidates_for(op,bucket)==grid
contracts=[native_compile.task_for(op,grid[i],bucket) for i in (0,len(grid)//2,len(grid)-1)]
assert all(contracts)
a.output.write_text(json.dumps(dict(registry_units=len(units),grid=len(grid),contracts=contracts),indent=2))
print('Registry build units and CPU compile contracts verified; full compile results in tuning reports')

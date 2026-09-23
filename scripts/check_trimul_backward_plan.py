"""Ensure module-derived and registry builds both reach the backward fusions."""
import argparse
import json
import os
from pathlib import Path
os.environ['MINIWORLD_COMPILE_WRAP']='disable'
from miniworld_engine.autotune import derive
from miniworld_engine.autotune.builder import cases,op_units

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
derive.require_environment();derive.install_no_calibration();derive.install_module_apply();derive.install_native_recorders()
catalog={c.name:c for c in cases()};records=[]
for length in (128,384,768):
    unit=derive.DeriveUnit('triangle_multiplication_bidirectional','token_pair',length,
        (('d_pair',128),('d_hidden',128)),'triton','bfloat16','','train',None)
    launches,error=derive.record(unit,catalog)
    records.append(dict(length=length,error=error,launches=launches))
    assert error is None,(length,error)
    for op in ('trimul_input_dual_bwd_triton','trimul_input_ln_residual_bwd_triton'):
        assert op in json.dumps(launches),launches
units=op_units(only={'trimul_input_dual_bwd_triton','trimul_input_ln_residual_bwd_triton'})
assert units,'new registry op has no build-all driver units'
a.output.write_text(json.dumps({'module_units':records,'registry_units':len(units)},indent=2,default=str))
print('Backward fusions found in module-derived plans and registry build units')

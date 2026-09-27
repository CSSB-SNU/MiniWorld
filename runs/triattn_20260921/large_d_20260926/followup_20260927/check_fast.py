import os
import runpy
from pathlib import Path
import torch
import accumulate
from build import extension
fast=extension('accumulate_fast',512);old=extension('accumulate',512)
torch.manual_seed(92791)
for C in (256,512):
    for rows in (64,257,147456):
        x=torch.randn(rows,C,device='cuda',dtype=torch.bfloat16)
        b=torch.randn(rows,4,device='cuda',dtype=torch.bfloat16)
        w=torch.randn(4,C,device='cuda',dtype=torch.bfloat16)
        a=old.accumulate(x.clone(),b,w);c=fast.accumulate(x.clone(),b,w)
        assert torch.equal(a,c)
print('FAST_EPILOGUE_BITWISE_PASS',flush=True)
original=accumulate.dgrad
def dgrad(gs,ws):return original(gs,ws,'accumulate_fast')
accumulate.dgrad=dgrad
os.environ['CHECK_TOOL']='fast-'+os.getenv('CHECK_TOOL','plain')
runpy.run_path(str(Path(__file__).with_name('check_accumulate.py')),run_name='__main__')

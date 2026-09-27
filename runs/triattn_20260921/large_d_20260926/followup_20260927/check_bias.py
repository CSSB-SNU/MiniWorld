"""Reuse frozen independent FP64/mask/graph fixtures with compact dBias output."""
from pathlib import Path
import runpy
import sys
from build import extension
BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE))
import candidate
def backward(q,k,v,b,m,out,dy,mode='full'):
    D=q.shape[-1];dy=candidate.projection_layout(dy)
    delta=candidate.aux_extension().delta(out,dy)
    ext=extension('bias16',D*4)
    dk,dv,db=ext.backward(q,k,v,b,m,delta,dy,ext.row_group)
    dq=candidate.extension(D,'dq').backward(q,k,v,b,m,delta,dy)
    return dq,dk,dv,db
candidate.backward=backward
runpy.run_path(str(BASE/'check_native.py'),run_name='__main__')

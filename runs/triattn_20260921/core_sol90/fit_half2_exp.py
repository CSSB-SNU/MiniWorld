"""Fit exp2 on [-.5,.5] with actual FP16-rounded Horner stages.

This coefficient search is independent of attention benchmark tensors.
The CUDA probe must confirm real HFMA2 rounding and end-to-end accuracy.
"""
from pathlib import Path
import itertools,json
import numpy as np

HERE=Path(__file__).resolve().parent
grid=np.linspace(-.5,.5,16385)
target=np.exp2(grid)
fraction=grid.astype(np.float16).astype(np.float32)
results={}
for degree in (3,4):
 exact=np.linalg.lstsq(np.polynomial.polynomial.polyvander(grid,degree)/target[:,None],np.ones_like(grid),rcond=None)[0]
 center=np.array(exact,dtype=np.float16)
 neighborhoods=[[np.nextafter(c,np.float16(-np.inf)),c,np.nextafter(c,np.float16(np.inf))] for c in center]
 best=None
 for choice in itertools.product(*neighborhoods):
  y=np.full_like(fraction,float(choice[-1]))
  for c in reversed(choice[:-1]):
   y=(y*fraction+float(c)).astype(np.float16).astype(np.float32)
  rel=y/target-1
  score=float(np.mean((rel-rel.mean())**2))
  if best is None or score<best[0]:best=(score,choice,float(rel.mean()),float(np.max(np.abs(rel))))
 variance,choice,mean,maximum=best
 bits=np.array(choice,dtype=np.float16).view(np.uint16)
 results[str(degree)]=dict(coefficients=[float(x) for x in choice],packed_bits=[int(x)|(int(x)<<16) for x in bits],
  grid_points=len(grid),fraction_rounded_to_half=True,stages_rounded_to_half=True,
  relative_error_variance=variance,relative_error_mean=mean,max_relative_error=maximum)
 print(degree,results[str(degree)],flush=True)
(HERE/'half2-exp-coefficients.json').write_text(json.dumps(results,indent=2)+'\n')

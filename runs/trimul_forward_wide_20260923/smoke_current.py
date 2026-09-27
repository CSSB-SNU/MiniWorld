"""Check the final device-context wrapper with CUDA graph replay on all shapes."""
from validate_engine import *
import gc

checks={}
for D in (256,384,512):
    for N in (384,768):
        with torch.no_grad():
            leaves,dy,mask,ds,*_=setup(D,N)
            model=W.Forward(leaves,mask,ds)
            graph,y=capture(model)
            leaves[0].mul_(.99);leaves[1].add_(.001);ds.mul_(.9)
            expected=model().clone();graph.replay();torch.cuda.synchronize()
            assert torch.equal(y,expected)
            checks[f'{D}-{N}']=True
            print('GRAPH_OK',D,N,flush=True)
            del graph,y,expected,model,leaves,dy,mask,ds
        gc.collect()
(R/'final-context-graph.json').write_text(json.dumps(checks,indent=2)+'\n')

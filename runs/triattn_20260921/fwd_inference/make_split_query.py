"""Static query groups with resident KV; explicit recompute-versus-occupancy controls."""
from pathlib import Path

r = Path(__file__).resolve().parent
base = (r/'resident6/fused.cu').read_text()
for consumers in (2, 4):
    name = 'splitq%d' % consumers
    s = base.replace('for(int batch=0;batch<nt;batch+=Consumers) {',
                     '{\n    int batch=int(blockIdx.z)*Consumers;')
    s = s.replace('s.wfull.wait((1+batch/Consumers)%2)', 's.wfull.wait(1)')
    s = s.replace('((qt/Consumers)*(nt/Stages)+kt/Stages)%2', '(kt/Stages)%2')
    s = s.replace('<<<dim3(4,L),', '<<<dim3(4,L,(L/64+Consumers-1)/Consumers),')
    for cap, oldc, stages in ((384,6,2),(768,6,1),(1024,6,1)):
        s = s.replace('launch<%d,%d,%d,%d>'%(cap,oldc,stages,oldc),
                      'launch<%d,%d,2,%d>'%(cap,consumers,consumers))
    s = s.replace('Config<1024,6,1,6>::Shared', 'Config<1024,%d,2,%d>::Shared'%(consumers,consumers))
    path = r/name
    assert not (path/'build-ready.json').exists()
    path.mkdir(exist_ok=True)
    (path/'fused.cu').write_text(s)

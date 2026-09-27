"""Prepare fresh final artifacts; do not rewrite an already measured candidate."""
import argparse
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('--bias',required=True)
ap.add_argument('--dq',required=True)
ap.add_argument('--only',choices=('bias','dq','both'),default='both')
a=ap.parse_args()
root=Path(__file__).resolve().parent.parent
for kind,folder,source,target,filename in (
    ('bias','bias_fusion',a.bias,'rs8_tma','grouped.cu'),
    ('dq','dq',a.dq,'rs_tma_ldmatrix','fused.cu')):
    if a.only!='both' and a.only!=kind:continue
    path=root/folder/target
    assert not path.exists(),('release artifact already exists',path)
    s=(root/folder/source/filename).read_text()
    if kind=='bias':
        s=s.replace('// One TMA producer, four consumer warpgroups; each WG owns one outer row.',
                    '// One producer warpgroup and two consumer warpgroups; each consumer owns four outer rows.')
        s=s.replace('// dBias reduction reads the four BF16 dS tiles directly.',
                    '// dBias reduction reads the eight BF16 dS tiles directly.')
        s=s.replace('// All four consumers reduce one quarter of the bias tile each.',
                    '// Consumers split the bias tile into disjoint FP32 partial outputs.')
        s=s.replace('"ABI4 or actual group8 required"','"legacy group4 ABI or native group8 required"')
        s=s.replace('m.def("backward",&backward);','m.attr("row_group")=8;m.def("backward",&backward);')
    path.mkdir();(path/filename).write_text(s)
    (path/'origin.txt').write_text(source+'\n')
    print(kind,source,'->',target)

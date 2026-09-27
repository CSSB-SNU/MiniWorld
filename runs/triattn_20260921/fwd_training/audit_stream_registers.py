"""Reject register redistribution exceeding the actual per-CTA initial pool."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ap=argparse.ArgumentParser();ap.add_argument('--artifact',required=True);a=ap.parse_args()
r=Path(__file__).resolve().parent/a.artifact
resources=(r/'resources.txt').read_text();sass=(r/'sass.txt').read_text()
registers={m.group(1):int(m.group(2)) for m in re.finditer(r'Function\s+(\S+?)\s*:\s*REG:(\d+)',resources)}
assert registers,resources[:2000]
records=[]
role_path=r/'register-roles.json'
roles=json.loads(role_path.read_text()) if role_path.exists() else {}
for section in re.split(r'Function\s*:\s*',sass)[1:]:
    name=section.split()[0]
    shape=re.search(r'ILi(\d+)ELi(\d+)ELi(\d+)E',name)
    if not shape:continue
    capacity,consumers,stages=map(int,shape.groups())
    assert name in registers,name
    initial=registers[name]
    warpgroups=consumers if 'qkv_attention_compact' in name else consumers+1
    # Hopper allocates registers in 256-register warp units (8 per thread).
    allocated=(initial+7)//8*8
    pool=allocated*warpgroups*128
    inc=set(int(x,16) for x in re.findall(r'USETMAXREG\.TRY_ALLOC\.CTAPOOL\s+UP\d+,\s*0x([0-9a-f]+)',section))
    dec=set(int(x,16) for x in re.findall(r'USETMAXREG\.DEALLOC\.CTAPOOL\s+0x([0-9a-f]+)',section))
    if inc or dec:
        assert len(inc)==len(dec)==1,(name,inc,dec)
        role=roles.get(str(capacity),dict(consumers=consumers,producers=1))
        assert role['consumers']+role['producers']==warpgroups,(name,role,warpgroups)
        need=(next(iter(inc))*role['consumers']+next(iter(dec))*role['producers'])*128
    else:need=pool
    record=dict(kernel=name,capacity=capacity,consumers=consumers,stages=stages,
                initial_registers=initial,allocated_registers=allocated,warpgroups=warpgroups,
                pool=pool,required=need,valid=need<=pool and pool<=65536)
    if inc or dec:
        record['redistribution']=dict(consumer_warpgroups=role['consumers'],
            producer_warpgroups=role['producers'],consumer_registers=next(iter(inc)),
            producer_registers=next(iter(dec)))
    records.append(record)
assert records
report=dict(artifact=a.artifact,records=records,complete=all(x['valid'] for x in records),
            build=json.loads((r/'build-ready.json').read_text()),
            evidence_sha256={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ('sass.txt','resources.txt')})
if role_path.exists():report['evidence_sha256']['register-roles.json']=hashlib.sha256(role_path.read_bytes()).hexdigest()
(r/'register-audit.json').write_text(json.dumps(report,indent=2)+'\n')
for row in records:print('REGISTER_POOL',row)
assert report['complete'],'register pool too small; do not launch'

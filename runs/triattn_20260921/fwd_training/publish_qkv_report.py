"""Publish only a completed, verified full-QKV promotion; preserve historical reports."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ap=argparse.ArgumentParser();ap.add_argument('--promotion',type=int,required=True);a=ap.parse_args()
r=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(r/'summarize_qkv_alias.py'),'--promotion',str(a.promotion)],check=True)
data=json.loads((r/'qkv-alias-results.json').read_text())
assert data['promotion']['state']=='complete'
for old,new in [('README.md','IMPROVEMENT_18006.md'),('installed-results.json','installed-results-18006.json')]:
    if not (r/new).exists():shutil.copy2(r/old,r/new)
(r/'README.md').write_text((r/'QKV_ALIAS_RESULTS.md').read_text())
installed=dict(data,baseline_promotion=18006,node='node02',qos='normal_h100',
               previous_installed_results='installed-results-18006.json')
installed['measurements']=[x for x in data['full_module'] if x['label']=='installed%s'%a.promotion]
installed['installed_full_module']=installed['measurements']
installed['installed_checks']=json.loads((r/('qkv-installed-verify-%s.json'%a.promotion)).read_text())
(r/'installed-results.json').write_text(json.dumps(installed,indent=2)+'\n')

for filename,note in [
    ('QKV_STREAM_RESULTS.md','Historical full-QKV follow-up through18162. Later L1024 full fusion is installed; see [current results](QKV_ALIAS_RESULTS.md).'),
    ('QG_FUSION_RESULTS.md','Historical promotion18006. L384/L768 retain this QG implementation; L1024 now uses [qualified full-QKV fusion](QKV_ALIAS_RESULTS.md).'),
]:
    p=r/filename;s=p.read_text();line='> '+note+'\n\n'
    if line not in s:
        title,rest=s.split('\n',1);p.write_text(title+'\n\n'+line+rest.lstrip('\n'))

rows=installed['measurements']
f=[x['time_reduction_pct'] for x in rows if x['kind']=='forward']
b=[x['time_reduction_pct'] for x in rows if x['kind']=='forward_backward']
note=('> **Full Q/K/V/gate fusion installed at L1024, 2026-09-26:** promotion%s, '
      '`qkv_compact_retire6`. [Current results](fwd_training/QKV_ALIAS_RESULTS.md) compare '
      'actual installed full FWD/BWD/F+B with frozen18006. Both directions: FWD%.2f–%.2f%% '
      'and F+B%.2f–%.2f%% time reduction, independently repeated with64x40 paired graphs. '
      'Projection Z/weights share storage with attention Q/bias, enabling four projection '
      'warpgroups at1024. L384/L768 retain18006 QG. All outputs/gradients bitwise; FP64, '
      'cold compile, graph, AMP, three sanitizers includingL1024 and fresh-installed '
      'default/opt-out/fallback checks pass. Existing backward binaries unchanged. '
      'Ten new full-QKV controls are recorded; SOL90 remains unmet.\n\n')%(
          a.promotion,min(f),max(f),min(b),max(b))
p=r.parent/'HANDOFF.md';s=p.read_text()
if note not in s:
    title,rest=s.split('\n',1);p.write_text(title+'\n\n'+note+rest.lstrip('\n'))
print('Published installed results and handoff for',a.promotion)

"""Collect initial screens; these results are not full qualification."""
from pathlib import Path
import json,re,hashlib

HERE=Path(__file__).resolve().parent
cases=[
 (14708,'basesharedp3partialv2',False),
 (14709,'basesharedp3fullv2',False),
 (14716,'basesharedp3stsmv2',False),
 (14727,'basesharedp3mixedv3',False),
 (14757,'coop4pairedp6qrv4',False),
 (14764,'coop2pairedp6qr2ctav4',True),
 (14770,'m64coop2s4p3qr',True),
 (14774,'m64coop2s4p3qrkv4b2',True),
 (14777,'m64coop2s4p2sskv4',True),
]
rows=[]
for job,variant,two_cta in cases:
 path=HERE/(variant+'-L768.json')
 result=json.loads(path.read_text())['results']
 baseline=result['serving'];candidate=result[variant]
 def hot(d,pattern):
  found=[value for name,value in d['kernels'].items() if pattern in name]
  assert len(found)==1,(variant,pattern,found)
  return found[0]
 base_hot=hot(baseline,'Traits<1073741824>')
 candidate_hot=hot(candidate,'Traits<1073741824>' if variant.startswith('base') else 'attention<false>')
 log=(HERE/('build-'+variant+'.log')).read_text()
 assert not re.search(r'C751[12457]|C7520|wgmma\.mma_async instructions are serialized',log)
 assert candidate['core_equal'] and candidate['block_delta_rms']==0
 assert candidate['core_rms_fp64']==baseline['core_rms_fp64']
 assert candidate_hot>base_hot
 rows.append(dict(job=job,variant=variant,baseline_hot_us=base_hot,hot_us=candidate_hot,
  hot_ratio=candidate_hot/base_hot,baseline_core_us=baseline['core_us'],core_us=candidate['core_us'],
  baseline_block_us=baseline['block_us'],block_us=candidate['block_us'],
  initial_full_core_and_block_bitwise=True,initial_fp64_pair_row0_identical=True,
  runtime_two_cta_assertion_passed=two_cta,full_qualification=False,decision='rejected_slower',
  result_file=path.name))
installed=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast/triattn_broadcast.so'
sha=hashlib.sha256(installed.read_bytes()).hexdigest()
assert sha=='97634810cdba1e91fe4aecb8bdfa610dd050dd39e7504ffb8babee8c46d4077b'
out=dict(installed_sha256=sha,installed_changed=False,sol90_reached=False,
 limitations='Initial three-round graph and CUPTI screens, not balanced gain proof or sanitizer qualification.',
 results=rows)
(HERE/'shared-probability-results.json').write_text(json.dumps(out,indent=2)+'\n')
for row in rows:
 print(row['job'],row['variant'],'hot %.3f vs %.3f us'%(row['hot_us'],row['baseline_hot_us']),row['decision'])

"""Narrow diagnostic timing exception for the known three-score spill build.

This does not weaken the ordinary no-spill screen or qualify installation.
The purpose is to measure whether this architecture merits further tuning.
"""
from pathlib import Path
import json, re
HERE = Path(__file__).resolve().parent
s = (HERE / 'build-baseproducerexp3score.log').read_text()
assert 'BUILT baseproducerexp3score' in s
assert not re.search(r'C751[12457]|C7520|wgmma\.mma_async instructions are serialized', s)
m = re.search(r'Function properties for (\S*1073741824\S*)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers', s)
assert m and list(map(int, m.groups()[1:])) == [32, 36, 108, 128]
d = json.loads((HERE / 'producerexp3score-spill-attribution.json').read_text())
assert d['variant'] == 'baseproducerexp3score' and d['static_spills'] == 36
assert d['reachable_spills_by_role_register_budget'] == {'160': 36, '32': 0}
print('DIAGNOSTIC ONLY: known consumer spills explicitly permitted for initial timing; not qualified', flush=True)

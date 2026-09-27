"""Reject spilled/serialized fused kernels before performance testing."""
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()
assert 'BUILT triattn_projected_p' in s,'missing completed build'
assert not re.search(r'C751[245]',s),'compiler serialization/analysis warning'
matches=re.findall(r'Function properties for (\S*attention\S*)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers',s)
assert len(matches)==2,matches
for name,stack,stores,loads,regs in matches:
 assert int(stack)==int(stores)==int(loads)==0,(name,stack,stores,loads)
 assert int(regs)*384>=63488,(name,regs,'insufficient initial register pool')
print('PASS hot/SAFE: zero spills, sufficient initial pool, no C7512/14/15',flush=True)

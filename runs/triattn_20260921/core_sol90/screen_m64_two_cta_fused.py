import re, sys
from pathlib import Path
s=Path(sys.argv[1]).read_text()
assert not re.search(r'C751[12457]|C7520|wgmma\.mma_async instructions are serialized',s),'compiler serialized MMA'
m=re.search(r'Function properties for .*attentionILb0.*?\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\n.*?Used (\d+) registers',s,re.S)
assert m,'missing hot compiler report'
stack,stores,loads,regs=map(int,m.groups())
assert (stack,stores,loads)==(0,0,0),(stack,stores,loads)
assert 80<=regs<=80,regs
print('SCREEN PASS hot80, no spills or WGMMA serialization',flush=True)

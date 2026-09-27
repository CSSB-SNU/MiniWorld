from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()
m=re.search(r'BUILT coop([45])s2p1',s);assert m
limit=128 if m[1]=='4' else 96
assert not re.search(r'C751[245]|C7520',s),'serialized/unsupported MMA'
rows=re.findall(r'Function properties for (\S*attention\S*)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers',s)
assert len(rows)==2,rows
for name,stack,stores,loads,regs in rows:
 assert int(stack)==int(stores)==int(loads)==0,(name,stack,stores,loads)
 assert int(regs)<=limit,(name,regs)
print('PASS one-P static register cap',limit,'zero spills, no serialization',flush=True)

from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()
assert 'BUILT resident4m64' in s
assert not re.search(r'C751[12457]|C7520|wgmma\.mma_async instructions are serialized',s),'serialized/unsupported MMA'
rows=re.findall(r'Function properties for (\S*attention\S*)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers',s)
assert len(rows)==2,rows
for name,stack,stores,loads,regs in rows:
 assert int(stack)==int(stores)==int(loads)==0,(name,stack,stores,loads)
 assert int(regs)<=96,(name,regs)
print('PASS four-consumer resident resources',rows,flush=True)

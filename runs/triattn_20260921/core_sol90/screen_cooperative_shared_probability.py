from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()
assert re.search(r'BUILT (?:coop4(?:sharedp3qr|pairedp6qr)|coop2pairedp6qr2cta|m64coop2s4p3qr|m64coop2s4p2sskv4)',s)
assert not re.search(r'C751[12457]|C7520|wgmma\.mma_async instructions are serialized',s),'serialized/unsupported MMA'
rows=re.findall(r'Function properties for (\S*attention\S*)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers',s)
assert len(rows)==2,rows
for name,stack,stores,loads,regs in rows:
 assert int(stack)==int(stores)==int(loads)==0,(name,stack,stores,loads)
 assert int(regs)<=128,(name,regs)
print('PASS cooperative shared P zero spills, no serialization, regs<=128',rows,flush=True)

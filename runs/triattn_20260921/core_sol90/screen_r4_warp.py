from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()
assert 'BUILT m128r4' in s
assert not re.search(r'C751[245]|C7520',s),'serialized/unsupported MMA'
cooperative='BUILT m128r4coop2' in s
rows=re.findall(r'Function properties for (\S*attention\S*)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers',s)
assert len(rows)==2,rows
for name,stack,stores,loads,regs in rows:
 assert int(stack)==int(stores)==int(loads)==0,(name,stack,stores,loads)
 assert int(regs)<=(128 if cooperative else 96),(name,regs)
print('PASS R4 static registers, zero spills, no compiler serialization',flush=True)

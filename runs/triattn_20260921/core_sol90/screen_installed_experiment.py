from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()
assert 'BUILT base' in s
assert not re.search(r'C751[12457]|C7520|wgmma\.mma_async instructions are serialized',s),'compiler serialized MMA'
m=re.search(r'Function properties for (\S*1073741824\S*)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers',s)
assert m and [int(x) for x in m.groups()[1:]]==[0,0,0,128],m.groups() if m else None
print('PASS installed-family hot register resource screen',flush=True)

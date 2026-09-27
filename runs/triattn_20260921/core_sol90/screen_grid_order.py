from pathlib import Path
import re
here=Path(__file__).resolve().parent
s=(here/'build-basegrid.log').read_text()
assert 'BUILT basegrid' in s
assert not re.search(r'C751[245]|C7520',s),'compiler serialized MMA'
m=re.search(r'Function properties for (\S*1073741824\S*)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers',s)
assert m and [int(x) for x in m.groups()[1:]]==[0,0,0,128],m.groups() if m else None
print('PASS hot register resource screen',flush=True)

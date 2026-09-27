"""Skip numerical timing when the hot compilation spills or serializes."""
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()
marker=next((line for line in s.splitlines() if 'Compiling entry function' in line and '1073741824' in line),None)
if marker is None:print('NO HOT COMPILATION RECORD');sys.exit(2)
body=next(chunk for chunk in re.split(r'\n\[\d+/\d+\]',s) if marker in chunk)
m=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',body)
print('HOT SCREEN',m.group(0) if m else 'missing resources',flush=True)
bad=m is None or int(m.group(2)) or int(m.group(3)) or any(x in body for x in ('C7512','C7514','C7515'))
sys.exit(2 if bad else 0)

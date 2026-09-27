"""Bounded experiment exception: prove mixed168 spills belong only to producer.

This does not qualify performance or serving installation. Inspect the actual
selected hot SASS CFG, not just the aggregate ptxas spill count.
"""
from pathlib import Path
import re,json
HERE=Path(__file__).resolve().parent
s=(HERE/'build-baseqsmallmixed168.log').read_text()
assert 'BUILT baseqsmallmixed168' in s
assert not re.search(r'C751[245]|C7520',s)
m=re.search(r'Function properties for (\S*1073741824\S*)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers',s)
assert m and list(map(int,m.groups()[1:]))==[16,76,76,128]
code={}
for line in (HERE/'baseqsmallmixed168-hot.sass').read_text().splitlines():
    m=re.match(r'\s*/\*([0-9a-f]+)\*/\s*(.*?)\s*;',line)
    if m:code[int(m[1],16)]=m[2]
assert code
def opcode(instr):return next(x for x in instr.split() if not x.startswith('@'))
assert not any(opcode(v).startswith(('BRX','JMX','JMP','CALL','RET')) for v in code.values())
def reachable(start):
    visited=set();todo=[start]
    while todo:
        pc=todo.pop()
        if pc in visited:continue
        assert pc in code,hex(pc)
        visited.add(pc);line=code[pc];op=opcode(line)
        pred=line.startswith('@')
        if op.startswith('BRA'):
            target=re.search(r'\b0x([0-9a-f]+)\s*$',line)
            assert target,line
            todo.append(int(target[1],16))
            if pred:todo.append(pc+16)
        elif op=='EXIT':
            if pred:todo.append(pc+16)
        else:todo.append(pc+16)
    return visited
alloc={}
for pc,line in code.items():
    if 'USETMAXREG.' in line:
        budget=int(re.search(r'0x([0-9a-f]+)\s*$',line)[1],16)
        assert budget not in alloc
        alloc[budget]=pc
assert set(alloc)=={24,160,168}
spills={pc for pc,line in code.items() if opcode(line).startswith(('STL','LDL'))}
producer=reachable(alloc[24]);consumer=reachable(alloc[160])|reachable(alloc[168])
assert spills and not (spills & consumer)
assert spills <= producer
out={'initial_registers':128,'producer_budget':24,'consumer_budgets':[168,160,160],
     'static_spill_instructions':len(spills),'consumer_reachable_spills':0,
     'producer_reachable_spills':len(spills),'stack_bytes':16,
     'performance_qualified':False,'installation_qualified':False}
(HERE/'mixed168-producer-spill-screen.json').write_text(json.dumps(out,indent=2))
print('PASS bounded producer-only spill screen',out,flush=True)

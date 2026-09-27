"""Offline admission for standalone N16 CTA families, including emitted roles."""
from pathlib import Path
import hashlib,json,re,subprocess,sys
from register_role_gate import emitted_register_roles

HERE=Path(__file__).resolve().parent
name=sys.argv[1]
shape=re.fullmatch(r'm(64|128)n16r([345])s[24]p2qrf40(?:pf|fd(?:q[23])?)?',name)
assert shape,name
rows=int(shape[2]);threads=128*(rows+1);regs={3:160,4:112,5:88}[rows]
log=(HERE/('build-'+name+'.log')).read_text()
assert 'BUILT '+name in log,'Incomplete build'
binary=HERE/('build_'+name)/('triattn_sol_'+name+'.so')
source=(HERE/(name+'.cu')).read_text()
assert 'Rows=%d,Stages=2'%rows in source
assert 'warpgroup_reg_alloc<%d>'%regs in source
sass=subprocess.check_output(['/usr/local/cuda-12.9/bin/cuobjdump','-sass',str(binary)]).decode()
warnings=sorted(set(re.findall(r'C751[12457]|C7520',log)))
failures=['compiler '+x for x in warnings]
resources=[]
matches=re.findall(r'Function properties for (\S*attentionILb[01]\S*)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers',log)
assert len(matches)==2
for symbol,stack,stores,loads,initial in matches:
    safe='ILb1' in symbol
    body=next(x for x in sass.split('Function : ') if x.splitlines()[0]==symbol)
    (HERE/(name+('-safe.sass' if safe else '-hot.sass'))).write_text(body)
    emitted=emitted_register_roles(body)
    pool=128*(32+rows*regs);allocated=threads*int(initial)
    ops={op:len(re.findall(r'\b'+re.escape(op)+r'\b',body)) for op in ('HGMMA','WARPGROUP.ARRIVE','WARPGROUP.DEPBAR','MUFU.EX2','STL','LDL')}
    if emitted!={'dealloc_values':[32],'alloc_values':[regs]}:failures.append(('emitted roles',safe,emitted))
    if not pool<=allocated<=65536:failures.append(('pool',safe,pool,allocated))
    if any(int(v) for v in (stack,stores,loads)) or ops['STL'] or ops['LDL']:failures.append(('spill',safe))
    resources.append(dict(safe=safe,initial_registers=int(initial),stack=int(stack),spill_stores=int(stores),spill_loads=int(loads),threads=threads,producer_registers=32,consumer_registers=[regs]*rows,emitted=emitted,required_pool=pool,allocated_pool=allocated,counts=ops))
out=dict(name=name,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),resources=resources,compiler_warnings=warnings,failures=failures,admitted=not failures,scope='Compiler and actual register-role gate only; not runtime or numerical qualification.')
(HERE/(name+'-codegen.json')).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
sys.exit(bool(failures))

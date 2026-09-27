from pathlib import Path
import hashlib,json,re,subprocess,sys
HERE=Path(__file__).resolve().parent
name=sys.argv[1] if len(sys.argv)>1 else 'm64n64r2s3p2qrf40'
threads,consumers,consumer_regs={'m64n64r2s3p2qrf40':(384,2,224),'m64n64r3s2p2qrf40':(512,3,160)}[name]
log=(HERE/('build-'+name+'.log')).read_text()
assert 'BUILT '+name in log
assert not re.search(r'C751[12457]|C7520|wgmma\.mma_async instructions are serialized',log)
matches=re.findall(r'Function properties for (\S*attentionILb[01]\S*)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers',log)
assert len(matches)==2
resources=[]
for symbol,stack,stores,loads,registers in matches:
 assert (int(stack),int(stores),int(loads))==(0,0,0)
 assert threads*int(registers)>=128*(32+consumers*consumer_regs) and threads*int(registers)<=65536
 resources.append(dict(safe='ILb1' in symbol,initial_registers=int(registers),stack=0,stores=0,loads=0))
binary=HERE/('build_'+name)/('triattn_sol_'+name+'.so')
sass=subprocess.check_output(['/usr/local/cuda-12.9/bin/cuobjdump','-sass',str(binary)]).decode()
hot=next(x for x in sass.split('Function : ') if 'attentionILb0' in x.splitlines()[0])
(HERE/(name+'-hot.sass')).write_text(hot)
ops={op:len(re.findall(r'\b'+re.escape(op)+r'\b',hot)) for op in ('HGMMA','WARPGROUP.ARRIVE','WARPGROUP.DEPBAR','MUFU.EX2','STL','LDL')}
assert ops['STL']==ops['LDL']==0
out=dict(name=name,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),resources=resources,counts=ops,scope='Compiler gate only; not numerical or runtime qualification')
(HERE/(name+'-codegen.json')).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

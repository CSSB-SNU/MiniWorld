"""Bounded offline codegen comparison; never installs or launches a kernel."""
from pathlib import Path
import hashlib,json,re,subprocess,sys
from register_role_gate import emitted_register_roles

HERE=Path(__file__).resolve().parent
custom=len(sys.argv)>1
src=Path(sys.argv[1]) if custom else HERE/'build/triattn_sol_baseqthree1024/m1_1073741824.ptx'
out=Path(sys.argv[2]) if custom else HERE/'ptxas_current_screen';out.mkdir(exist_ok=True)
baseline=(HERE/'installed9365-hot.sass').read_text()
words=lambda s:re.findall(r'/\* (0x[0-9a-f]{16}) \*/',s)
basewords=words(baseline);assert len(basewords)==6928
rows=[]
configs=[('ru5',['--register-usage-level=5'])]
configs += [('ru'+str(i),['--register-usage-level='+str(i)]) for i in range(11) if i!=5]
configs += [('noexp',['--allow-expensive-optimizations=false'])]
for name,flags in configs:
 cubin=out/(name+'.cubin')
 cmd=['/usr/local/cuda-12.9/bin/ptxas','-arch=sm_90a','-O3','-v']+flags+[str(src),'-o',str(cubin)]
 run=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 log=run.stdout.decode();(out/(name+'.log')).write_text(log)
 assert run.returncode==0,(name,log)
 sass=subprocess.check_output(['/usr/local/cuda-12.9/bin/cuobjdump','-sass',str(cubin)]).decode()
 hot=next(p for p in sass.split('Function : ') if 'TraitsILi1073741824E' in p.splitlines()[0])
 (out/(name+'.sass')).write_text(hot)
 w=words(hot)
 if name=='ru5' and not custom:assert w==basewords,'default reassembly must reproduce installed machine words'
 props=re.search(r'Function properties for \S*1073741824\S*\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers',log)
 assert props,name
 row=dict(name=name,options=flags,machine_words=len(w),same_as_installed=w==basewords,resources=list(map(int,props.groups())),warnings=sorted(set(re.findall(r'C751[12457]|C7520',log))),counts={op:len(re.findall(r'\b'+re.escape(op)+r'\b',hot)) for op in ('HGMMA','WARPGROUP.ARRIVE','WARPGROUP.DEPBAR','MUFU.EX2','F2FP','LOP3.LUT')})
 row['register_reconfiguration']=emitted_register_roles(hot)
 rows.append(row);print(json.dumps(row),flush=True)
(out/'results.json').write_text(json.dumps(dict(ptx_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),scope='Offline compiler comparison, no runtime result',variants=rows),indent=2)+'\n')

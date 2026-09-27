from pathlib import Path
import shlex,subprocess,sys
r=Path(__file__).resolve().parent
name=sys.argv[1]
ninja=(r/('build_'+name)/'build.ninja').read_text()
vals=dict(line.split(' = ',1) for line in ninja.splitlines() if ' = ' in line and not line.startswith(' '))
flags=shlex.split(vals['cuda_cflags'])
flags=[x for x in flags if not x.startswith('-gencode=')]
out=r/(name+'.ptx')
cmd=[vals['nvcc']]+flags+['--ptx','-arch=compute_90a',str(r/(name+'.cu')),'-o',str(out)]
print('Generating',out,flush=True)
subprocess.run(cmd,check=True)
print('Saved',out.stat().st_size,'bytes',flush=True)

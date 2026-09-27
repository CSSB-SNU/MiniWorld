from pathlib import Path
import os,re,subprocess,json
R=Path(__file__).resolve().parent
log=(R/'job-19601.log').read_text()
source=Path(re.search(r'RuntimeError: (.*?\.cu):',log)[1])
ptx=R/'d256-register-minimum.ptx'
args=[os.environ['CUDA_HOME']+'/bin/nvcc','-std=c++17','-O3','-arch=sm_90a','--ptx','-lineinfo','-I'+str(R.parent.parent/'.engine-release-2.0.0/src/miniworld_engine/kernels/trimul_inproj/cuda/h100_sources/upstream/csrc'),'-DWEIGHT_SPLITS=8','-DPRODUCER_REGS=64','-DCONSUMER_REGS=88',str(source),'-o',str(ptx)]
subprocess.run(args,check=True,capture_output=True,text=True)
a=subprocess.run([os.environ['CUDA_HOME']+'/bin/ptxas','-arch=sm_90a','-v',str(ptx),'-o',str(R/'d256-register-minimum.cubin')],capture_output=True,text=True)
print(a.stderr,flush=True)
lines=ptx.read_text().splitlines()
for n in sorted(set(int(x) for x in re.findall(r'line (\d+)',a.stderr))):
 for i in range(max(0,n-10),min(len(lines),n+7)):print(str(i+1)+': '+lines[i])

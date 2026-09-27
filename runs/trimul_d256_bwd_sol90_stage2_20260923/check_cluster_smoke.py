"""Exact data/graph validation of the prospective B7 DSM handshake only."""
from pathlib import Path
import sys,os,ctypes,json
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent.parent/'.engine-release-2.0.0/src'))
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
rounds=16;clusters=2;words=2048
record=dict(job=os.environ.get('SLURM_JOB_ID'),complete=False,checks=[])
with torch.no_grad(),T.native_context(torch.device('cuda:0')):
 full_sync=int(os.environ.get('CLUSTER_FULL_SYNC','0'))
 direct=int(os.environ.get('CLUSTER_DIRECT_ARRIVE','0'))
 cubin=T.compile_text((R/'cluster_smoke.cu').read_text(),['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',f'-DCLUSTER_FULL_SYNC={full_sync}',f'-DCLUSTER_DIRECT_ARRIVE={direct}'])
 print('CUBIN',cubin,flush=True)
 k=T.load_unit(str(cubin),'mw_d256_cluster_smoke').kernel('mw_d256_cluster_smoke')
 smem=2*words*4+128;k.set_max_dynamic_smem(smem)
 out=torch.empty((rounds,clusters,4,8,words),device='cuda',dtype=torch.int32)
 seed=torch.tensor([7],device='cuda',dtype=torch.int32)
 L=T._launch_module();params=L.Struct([out,seed,rounds]);args=L._Packed([params]);drv=k.unit.drv;d=drv.d
 attr=d.CUlaunchAttribute();attr.id=d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
 attr.value.clusterDim.x=8;attr.value.clusterDim.y=1;attr.value.clusterDim.z=1
 cfg=d.CUlaunchConfig();cfg.gridDimX=clusters*8;cfg.gridDimY=1;cfg.gridDimZ=1
 cfg.blockDimX=256;cfg.blockDimY=1;cfg.blockDimZ=1;cfg.sharedMemBytes=smem
 cfg.attrs=[attr];cfg.numAttrs=1
 def run():
  cfg.hStream=d.CUstream(int(torch.cuda.current_stream().cuda_stream))
  drv._unwrap('cuLaunchKernelEx',d.cuLaunchKernelEx(cfg,d.CUfunction(int(k.handle)),ctypes.addressof(args.array),0))
 expected=(torch.arange(rounds,device='cuda',dtype=torch.int32)[:,None,None,None,None]*10000+
           torch.arange(clusters,device='cuda',dtype=torch.int32)[None,:,None,None,None]*8000000+
           torch.arange(8,device='cuda',dtype=torch.int32)[None,None,None,:,None]*1000000+
           torch.arange(words,device='cuda',dtype=torch.int32)[None,None,None,None,:]).expand_as(out)
 def check(name,value,fn):
  seed.fill_(value);out.fill_(-1);fn();torch.cuda.synchronize()
  exact=torch.equal(out,expected+value);record['checks'].append(dict(name=name,seed=value,exact=exact))
  print(record['checks'][-1],flush=True);assert exact,name
 check('ordinary',7,run);check('changed_seed',291,run)
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):run()
 for i in range(4):check('graph_'+str(i),719+i*937,g.replay)
record.update(complete=True,cubin=str(cubin),rounds=rounds,clusters=clusters,full_sync=full_sync,direct=direct)
(R/f'result-cluster-smoke-{record["job"]}.json').write_text(json.dumps(record,indent=2))

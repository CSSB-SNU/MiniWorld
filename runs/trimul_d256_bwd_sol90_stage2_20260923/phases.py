from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923';sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
N=384;leaves,dy,mask,ds,*_=setup(256,N)
with torch.no_grad(),T.native_context(leaves[0].device):
 f=F.Forward(leaves,mask,ds);f();p=W.Training(*leaves,mask,ds,dy,saved=(f.front.ab,f.tri,f.front.xn));b1=B1(p)
 flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DWIDTH=256','-DGROUPS=4','-DKCHUNK=1','-DTMN_SIGMOID_TANH=1','-DMW_MINB=1','-DMW_K1_STREAM=0']
 out=T.compile(THIS/'prepare_phases.cu',flags);k=T.load_unit(str(out),'mw_d256_prepare_phases').kernel('mw_d256_prepare_phases');k.set_max_dynamic_smem(b1.smem)
 clocks=torch.empty((b1.sms,3),device=p.x.device,dtype=torch.int64)
 maps=[F.tm(p.tri,[64,64],[p.M,512],[p.M*2]),*p.maps[:3]]
 params=T._launch_module().Struct([*maps,p.x,p.ds,p.y,p.dy,p.tensors[7],p.dg,p.tensors[6],p.floats[5],p.floats[6],p.floats[2],p.floats[3],p.M,p.n,clocks])
 for _ in range(10):k.launch((b1.sms,1,1),(512,1,1),[params],b1.smem)
 torch.cuda.synchronize();v=clocks.float().mean(0);print('LN, norm store, products fraction',(v/v.sum()).tolist(),flush=True)
 (THIS/'prepare-phases.json').write_text(json.dumps(dict(cycles=clocks.tolist(),fractions=(v/v.sum()).tolist()),indent=2))

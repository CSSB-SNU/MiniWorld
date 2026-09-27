from plan import *
from check import setup,bench,error
import json,argparse
p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);a=p.parse_args();D=a.width
leaves,dy,mask,ds,*_=setup(D,384)
with torch.no_grad():
    w=leaves[0].new_empty((8*D,D));pack_into(w,*leaves[1:5])
    cfg=[1,64,8,2,1] if D==512 else [2,64,2,6,1]
    base=Front(leaves[0],w,mask,leaves[7],leaves[8],cfg,D!=512);base();torch.cuda.synchronize();ab=base.ab.clone();xn=base.xn.clone()
    rows=[]
    for bi in (1,2):
        for sk in (1,2,4):
            for slots in (1,2,3,4):
                for mb in ([2,1] if bi==1 else [1]):
                    c=[bi,64,slots,sk,mb,2]
                    try:k1_smem(D,c)
                    except ValueError:continue
                    for norm in ([False,True] if D==512 else [True]):
                        print('TRY',c,norm,flush=True)
                        try:
                            f=Front(leaves[0],w,mask,leaves[7],leaves[8],c,norm);f();torch.cuda.synchronize()
                            er=error(f.ab,ab);xer=error(f.xn,xn);assert er<.005 and xer<.005
                            t=bench({'base':base,'new':f},10);row=dict(cfg=c,normalize=norm,times=t,error=er,xn_error=xer)
                        except Exception as e:row=dict(cfg=c,normalize=norm,failed=str(e))
                        rows.append(row);(R/f'stream-D{D}.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)

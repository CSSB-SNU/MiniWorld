from pathlib import Path
source=(Path(__file__).resolve().parent/'bench.py').read_text().split('\nwith torch.no_grad():')[0]
exec(compile(source,str(Path(__file__).resolve().parent/'bench.py'),'exec'))
with torch.no_grad():
 q,k,v,g,b=pf.prologue(x,W,impl='fpf',ending=False)
 for case in range(8):
  qq=q.clone();kk=k.clone();vv=v.clone();bb=b.clone()
  if case<3:qq.zero_();kk.zero_()
  if case==0:bb.zero_()
  if case==1:vv.fill_(1)
  if case==3:bb.zero_()
  if case>=5:
   qq.zero_();kk.zero_();bb.zero_()
   if case==5:vv.copy_(torch.arange(32,device=q.device,dtype=q.dtype).expand_as(vv))
   if case==6:vv.reshape(-1,a.length,32).copy_((torch.arange(a.length,device=q.device,dtype=torch.float32)%32).to(q.dtype)[None,:,None])
   if case==7:vv.reshape(-1,a.length,32).copy_((torch.arange(a.length,device=q.device,dtype=torch.float32)//32).to(q.dtype)[None,:,None])
  y=candidate(qq,kk,vv,bb,m5,32**-.5).reshape(1,a.length,4,a.length,32)
  def row(x):return x.reshape(1,a.length,4,a.length,32)[:,:8].double()
  log=row(qq)@row(kk).transpose(-1,-2)*32**-.5+bb.reshape(1,1,4,a.length,a.length).double()
  log.masked_fill_(~m5[:,:8],-torch.inf);ref=log.softmax(-1)@row(vv)
  print('CASE',case,'rms',rms(y[:,:8],ref),'rowRMS',(y[:,:8].double()-ref).square().mean((0,2,3,4)).sqrt().tolist(),'y',y[0,0,0,0,:8].tolist(),'ref',ref[0,0,0,0,:8].tolist(),flush=True)

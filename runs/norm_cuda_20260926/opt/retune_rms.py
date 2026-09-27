import torch,json,statistics
from pathlib import Path
from miniworld_engine.kernels.norm_cuda import cuda_rmsnorm,extension
extension();torch.set_num_threads(4);torch.manual_seed(622)
x=torch.randn(1,147456,128,device='cuda',dtype=torch.bfloat16,requires_grad=True);w=torch.randn(128,device='cuda',requires_grad=True);dy=torch.randn_like(x)
out=[]
for t in (128,256):
 for r in (1,2,4,8,12,16,24,32):
  def run():return torch.autograd.grad(cuda_rmsnorm(x,w,threads=t,rows=r),(x,w),dy)
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(3):run()
  torch.cuda.current_stream().wait_stream(stream);g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g,stream=stream):
   for _ in range(20):run()
  times=[]
  for _ in range(9):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();times.append(a.elapsed_time(b)*50)
  row={'threads':t,'rows':r,'train_us':statistics.median(times)};out.append(row);print(row,flush=True)
Path(__file__).with_name('retune-rms.json').write_text(json.dumps(out,indent=2))

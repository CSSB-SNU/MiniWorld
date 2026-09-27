"""Projection GEMM with exact MiniWorld gate gradients in its epilogue."""
from typing import NamedTuple
import os, torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, BFloat16
from cutlass.cute.runtime import from_dlpack
from quack.gemm_sm90 import GemmSm90
from quack.gemm_act import GemmActMixin
from quack.epi_ops import TileLoad, TileStore
from quack.cute_dsl_utils import mlir_namedtuple, get_device_capacity, get_max_active_clusters
from quack.gemm_tvm_ffi_utils import compile_gemm_kernel, make_scheduler_args, make_varlen_args, make_fake_scheduler_args
from quack.activation import sigmoid_tanh
from quack.rounding import RoundingMode
from blas_prepare import BlasPrepare

class GateGradMixin(GemmActMixin):
 _epi_ops=(TileStore('mAuxOut'),TileLoad('mDy'),TileLoad('mDs'))
 _extra_param_fields=()
 @mlir_namedtuple
 class EpilogueArguments(NamedTuple):
  mAuxOut:cute.Tensor
  mDy:cute.Tensor
  mDs:cute.Tensor
 def epi_to_underlying_arguments(self,args,*,loc=None,ip=None):
  self.rounding_mode=RoundingMode.RN
  self.aux_out_dtype=args.mAuxOut.element_type
  self.aux_out_layout=cutlass.utils.LayoutEnum.from_tensor(args.mAuxOut)
  self.cta_tile_shape_aux_out_mn=self.cta_tile_shape_mnk[:2]
  return self.EpilogueParams(**self._epi_ops_to_params_dict(args))
 @cute.jit
 def epi_visit_subtile(self,params,epi_loop_tensors,tRS_rD,tRS_rC=None):
  aux=cute.make_rmem_tensor(tRS_rD.layout.shape,Float32)
  dy=epi_loop_tensors['mDy'];ds=epi_loop_tensors['mDs']
  for i in cutlass.range(cute.size(aux),unroll_full=True):
   v=tRS_rD[i].to(BFloat16).to(Float32)
   a=(dy[i].to(Float32)*ds[i].to(Float32)).to(BFloat16).to(Float32)
   g=sigmoid_tanh(tRS_rC[i].to(Float32))
   tRS_rD[i]=a*g
   aux[i]=((a*v)*g)*(1.0-g)
  return aux
class GateGradSm90(GateGradMixin,GemmSm90):pass

class CutePrepare(BlasPrepare):
 def __init__(self,p,wp,wg):
  super().__init__(p,wp,wg)
  n=p.n
  batch=lambda x,c:x.reshape(n,n,c).permute(1,2,0)
  self.args=(batch(p.tensors[6],512),wp.unsqueeze(-1).expand(256,512,n),batch(p.tensors[7],256),batch(self.gate,256))
  self.epiargs=GateGradSm90.EpilogueArguments(batch(p.dg,256),batch(p.dy,256),p.ds.reshape(n,256).unsqueeze(-1).expand(n,256,n))
  dc=get_device_capacity(p.x.device);cm=int(os.environ.get('CUTE_CM','1'))
  self.sched=make_scheduler_args(get_max_active_clusters(cm,device_capacity=dc),8,None)
  self.var=make_varlen_args(None,None,None)
  cv=lambda x:from_dlpack(x.detach(),assumed_align=16)
  tile=(int(os.environ.get('CUTE_TM','128')),int(os.environ.get('CUTE_TN','128')))
  self.fn=compile_gemm_kernel(GateGradSm90,BFloat16,tile,(cm,1,1),os.environ.get('CUTE_PING','1')=='1',True,False,False,dc,*map(cv,self.args),GateGradSm90.EpilogueArguments(*map(cv,self.epiargs)),make_fake_scheduler_args(False,False,n),self.var)
 def projection(self):self.fn(*self.args,self.epiargs,self.sched,self.var,None)
 def __call__(self):
  self.normalize();torch.mm(self.p.xn.reshape(self.p.M,256),self.wg,out=self.gate);self.projection()

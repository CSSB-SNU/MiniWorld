"""Exercise every expanded config on representative tail shapes and fused math."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from miniworld_engine import settings
from miniworld_engine.autotune import cute_config as policy, native, native_compile
from miniworld_engine.kernels.layernorm_linear.cute.gemm_layernorm_linear import layernorm_linear_cute
from miniworld_engine.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import layernorm_linear_cute_fused
from miniworld_engine.kernels.layernorm_linear.cute.dgrad_lnbwd import dgrad_lnbwd_cute
from miniworld_engine.kernels.transition.cute.dab_lnbwd import transition_dab_lnbwd_cute
from miniworld_engine.kernels.transition.cute.gemm_transition_swiglu import transition_expand_swiglu_cute
from miniworld_engine.kernels.transition.cute.backward_gatebwd import transition_expand_gatebwd_cute
from miniworld_engine.kernels.transition.cute.squeeze_residual import squeeze_residual

parser = argparse.ArgumentParser()
parser.add_argument('--family', choices=('linear','transition','backward'), required=True)
parser.add_argument('--width', type=int, default=128)
args = parser.parse_args()
settings.configure(run_autotune=False, compile_jobs=12)
torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(721)
path = Path('runs/native_tuning_dev/qualification') / (args.family + (f'_D{args.width}' if args.width != 128 else '') + '.json')
path.parent.mkdir(parents=True, exist_ok=True)
results = {}
rand = lambda *shape: torch.randn(*shape, device='cuda', dtype=torch.bfloat16)
m,k,n = 264,args.width,256
x, gamma, beta, w, bias = rand(m,k), rand(k), rand(k)*.1, rand(n,k)*.08, rand(n)*.1
yln = F.layer_norm(x.float(), (k,), gamma.float(), beta.float(), .03)
ref_linear = yln @ w.float().T + bias.float()
gate = rand(m,n)
wa,wb = rand(n,k)*.08,rand(n,k)*.08
a,b = yln@wa.float().T, yln@wb.float().T
grad = rand(m,n)
mean = x.float().mean(-1)
rstd = torch.rsqrt(x.float().var(-1,unbiased=False)+.03)
xhat = ((x.float()-mean[:,None])*rstd[:,None]).to(x.dtype)
dxnorm = (grad.float()@w.float())*gamma.float()
def lnbwd_ref(xh):
    return (dxnorm-dxnorm.mean(-1,keepdim=True)-xh*(dxnorm*xh).mean(-1,keepdim=True))*rstd[:,None]

def qualify(op, call, references):
    captured = {}
    original = policy.resolve_config
    def discover(name, configs, **kw):
        captured.update(op=name, configs=configs, bucket=kw['bucket'])
        return kw.get('default') or configs[0]
    policy.resolve_config = discover
    try:
        call(None)
    finally:
        policy.resolve_config = original
    assert captured['op'] == op
    configs = captured['configs']
    declared = {json.dumps(c,sort_keys=True) for c in native.candidates_for(op,captured['bucket'])}
    assert declared == {json.dumps(policy.config_to_kwargs(c),sort_keys=True) for c in configs}
    compiled = native_compile.precompile(op,[policy.config_to_kwargs(c) for c in configs],captured['bucket'])
    rows = []
    for i,c in enumerate(configs):
        if i in compiled and compiled[i]['status'] != 'ok':
            rows.append({'config':policy.config_to_kwargs(c),'compile':compiled[i]})
            continue
        actual = call(c)
        actual = actual if isinstance(actual,tuple) else (actual,)
        errors=[]
        for output,ref in zip(actual,references):
            error=float((output.float()-ref).norm()/ref.norm().clamp_min(1e-8))
            errors.append(error)
        if any(not e < .025 for e in errors):
            print('NUMERIC REJECT',op,c,errors,flush=True)
            rows.append({'config':policy.config_to_kwargs(c),'numerical_failure':errors})
            continue
        rows.append({'config':policy.config_to_kwargs(c),'relative_l2':errors})
    results[op] = {'candidates':len(configs),'correct':sum('relative_l2' in r for r in rows),'rows':rows}
    path.write_text(json.dumps(results,indent=2))
    print(op,{k:v for k,v in results[op].items() if k!='rows'},flush=True)

if args.family == 'linear':
    qualify('layernorm_linear_fwd_foldstats_sm90_cute',
            lambda c:layernorm_linear_cute(x,gamma,beta,w,bias,.03,config=c),[ref_linear])
    qualify('layernorm_linear_fwd_sm90_cute',
            lambda c:layernorm_linear_cute_fused(x,gamma,beta,w,bias,.03,config=c,gate=gate),[ref_linear*gate.float()])
    expand,ws,residual = rand(m,2048), rand(512,2048)*.02, rand(m,512)
    def squeeze(c):
        if c is None:
            return squeeze_residual(expand,ws,residual)
        original=policy.resolve_config
        policy.resolve_config=lambda *a,**kw:c
        try:
            return squeeze_residual(expand,ws,residual)
        finally:
            policy.resolve_config=original
    qualify('transition_squeeze_residual_sm90_cute',squeeze,[expand.float()@ws.float().T+residual.float()])
elif args.family == 'transition':
    qualify('transition_swiglu_fwd_sm90_cute',
            lambda c:transition_expand_swiglu_cute(x,gamma,beta,wa,wb,.03,config=c),[F.silu(a)*b])
    a,b=x.float()@wa.float().T,x.float()@wb.float().T
    da=grad.float()*b*a.sigmoid()*(1+a*(1-a.sigmoid()))
    db=grad.float()*F.silu(a)
    dab=torch.stack((da,db),-1).flatten(-2)
    qualify('transition_gate_bwd_sm90_cute',
            lambda c:transition_expand_gatebwd_cute(x,grad,wa,wb,config=c),[F.silu(a)*b,dab])
else:
    qualify('layernorm_linear_bwd_dx_sm90_cute',
            lambda c:dgrad_lnbwd_cute(grad,w,xhat,gamma,rstd,config=c),[lnbwd_ref(xhat.float())])
    qualify('transition_bwd_dx_sm90_cute',
            lambda c:transition_dab_lnbwd_cute(grad,w,x,gamma,rstd,mean*rstd,config=c),
            [lnbwd_ref((x.float()-mean[:,None])*rstd[:,None])])
print('ALL DONE',args.family,flush=True)
assert all(r['correct']==r['candidates'] for r in results.values()), 'some candidates failed qualification'

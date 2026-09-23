"""SM90 F567 configuration axes, TMA tails, saves, bias and graph semantics."""
import pytest
import torch
from miniworld_engine.autotune.cute_config import f567_candidates
from miniworld_engine.kernels.trimul_inproj.cute.output_f567 import output_f567_impl,output_f567_sm90


def inputs(m=263,n=40,kp=200,kg=72,l=17,bias=True):
    torch.manual_seed(315);kw=dict(device='cuda',dtype=torch.bfloat16)
    a=torch.randn(m,kp,**kw);x=torch.randn(m,kg,**kw)
    wp=torch.randn(n,kp,**kw)/kp**.5;wg=torch.randn(kg,n,**kw)/kg**.5
    res=torch.randn(m,n,**kw);ds=(torch.rand(l,n,device='cuda')>.25).to(torch.bfloat16)/.75
    b=torch.randn(n,**kw)*.1 if bias else None
    return a,x,wp,wg,res,ds,l,b


def reference(args):
    a,x,wp,wg,r,ds,l,b=args
    p=(a.float()@wp.float().t()+(b.float() if b is not None else 0)).bfloat16().float()
    g=torch.sigmoid((x.float()@wg.float()).bfloat16().float())
    return r.float()+p*g*ds.float()[torch.arange(a.shape[0],device=a.device)%l],p,g


def compare(values,refs):
    errors=[]
    for v,r in zip(values,refs):
        assert torch.isfinite(v).all()
        e=((v.float()-r.float()).norm()/r.float().norm().clamp_min(1e-8)).item()
        errors.append(e);assert e<.004,errors
    return errors


CONFIGS=[dict(tile_m=tm,tile_n=tn,tile_k=tk,group_m=gm) for tm,tn,tk,gm in [
 (64,32,32,1),(64,64,64,2),(64,128,128,4),(128,32,128,8),
 (128,128,32,2),(192,32,64,4),(256,32,32,8),
 (64,128,32,8),(192,64,32,1),(128,64,64,4)]]


@pytest.mark.parametrize('config',CONFIGS)
@pytest.mark.parametrize('bias',[False,True])
def test_tails(config,bias):
    args=inputs(bias=bias);compare(output_f567_impl(*args,config=config),reference(args))


def test_noncontiguous_gate_weight():
    args=list(inputs());args[3]=args[3].t().contiguous().t()
    compare(output_f567_impl(*args,config=CONFIGS[0]),reference(args))


def test_unsupported_alignment():
    args=inputs(kp=201)
    with pytest.raises(ValueError,match='aligned'):output_f567_impl(*args,config=CONFIGS[0])


def test_grid_resource_and_axes():
    grid=f567_candidates(256,128,232448)
    assert grid and len(grid)==len({tuple(c.items()) for c in grid})
    for field in ('tile_m','tile_n','tile_k','group_m'):assert len({c[field] for c in grid})>1
    assert not f567_candidates(4096,4096,232448)


def test_compiled_graph_replay():
    args=list(inputs());fn=torch.compile(output_f567_sm90,fullgraph=True,dynamic=False)
    compare(fn(*args),reference(args))
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):out=fn(*args)
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):out=fn(*args)
    graph.replay();torch.cuda.synchronize()
    compare(out,reference(args))
    args[5].zero_();graph.replay();torch.cuda.synchronize()
    assert torch.equal(out[0],args[4])

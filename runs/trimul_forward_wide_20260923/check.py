from plan import *
import argparse,json,statistics,os,time
sys.path.insert(0,str(R.parent/'trimul_cuda_widths_opt_20260923'))
from fixture import setup
import width_plan

def error(y,ref):
    return float((y.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-20))

def capture(fn):
    s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=s): y=fn()
    g.replay();torch.cuda.synchronize()
    return g,y

def bench(fns,iters=30):
    graphs={k:capture(v)[0] for k,v in fns.items()}
    ev={k:[] for k in graphs}
    for r in range(3):
        for g in graphs.values():
            for _ in range(5):g.replay()
        for i in range(iters):
            for k in (list(graphs) if (i+r)%2 else list(graphs)[::-1]):
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();graphs[k].replay();b.record();ev[k].append((a,b))
    torch.cuda.synchronize()
    return {k:statistics.median(a.elapsed_time(b)*1000 for a,b in pairs) for k,pairs in ev.items()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);p.add_argument('--length',type=int,default=384);p.add_argument('--tune',action='store_true');p.add_argument('--sanitize',action='store_true');a=p.parse_args()
    D,N=a.width,a.length
    result=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),candidates=[],complete=False)
    path=R/f'result-D{D}-L{N}.json'
    def save():
        text=json.dumps(result,indent=2);path.write_text(text)
        (R/f'job-{result["job"]}-result.json').write_text(text)
    leaves,dy,mask,ds,ref,triton,names=setup(D,N)
    with torch.no_grad():
        print('BASELINE',D,N,flush=True)
        old=width_plan.Training(*leaves,mask,ds,dy);yold=old.forward().clone();torch.cuda.synchronize()
        cfg=list(old.front.cfg);normalize=not old.separate
        out_candidates=[]
        output_configs=([(3,1),(3,2),(2,2)] if D==384 else [(4,1),(4,2),(2,2)] if D==256 else [(4,1),(2,2),(1,2)])
        for g,kc in output_configs:
            print('OUTPUT',g,flush=True)
            out=Output(old.tri,old.xn,leaves[0],leaves[6],leaves[5],leaves[9],leaves[10],ds,g,kchunk=kc)
            y=out();torch.cuda.synchronize();er=error(y,yold)
            print('OUTPUT_ERROR',g,er,flush=True);assert er<.005 and bool(y.isfinite().all())
            times=bench({'previous_output':lambda:width_plan.launch(old.ks['forward'],old.params,old.grid,D=D),'new_output':out})
            out_candidates.append(dict(groups=g,kchunk=kc,error=er,times=times));print('OUTPUT_TIME',g,kc,times,flush=True)
        result['output_candidates']=out_candidates
        out_config=min(out_candidates,key=lambda z:z['times']['new_output']);groups=out_config['groups'];kc=out_config['kchunk'];save()
        configs=[(cfg,normalize)]
        if a.tune:
            for bi in (1,2,3):
                for slots,sk in ((1,D//64),(2,D//64),(4,D//64),(8,2),(4,2),(2,4)):
                    for mb in ([1,2] if bi==1 else [1]):
                        c=[bi,64,slots,sk,mb]
                        try:k1_smem(D,c)
                        except ValueError:continue
                        for norm in ([True,False] if D==512 else [True]):
                            if (c,norm) not in configs:configs.append((c,norm))
            configs += [(c+[1],norm) for c,norm in list(configs)]
        best=None
        for c,norm in configs:
            print('FRONT',c,norm,flush=True)
            try:
                model=Forward(leaves,mask,ds,c,groups,norm,kchunk=kc);y=model();torch.cuda.synchronize()
                er=error(y,yold);xnerr=error(model.front.xn,old.xn);aberr=error(model.front.ab,old.front.ab)
                assert er<.005 and xnerr<.005 and aberr<.005 and bool(y.isfinite().all()),(er,xnerr,aberr)
                times=bench({'previous_front':lambda:(normalize_into(old.xn,leaves[0],leaves[7],leaves[8]) if old.separate else None,old.front()),'new_front':model.front},15)
                row=dict(cfg=c,normalize=norm,error=er,xn_error=xnerr,ab_error=aberr,times=times)
                if best is None or times['new_front']<best[0]:best=(times['new_front'],model,c,norm)
            except Exception as e:
                row=dict(cfg=c,normalize=norm,failed=str(e));print('FAILED',str(e)[-800:],flush=True)
            result['candidates'].append(row);save();print('FRONT_RESULT',row,flush=True)
        assert best is not None
        _,model,c,norm=best
        result['selected']=dict(k1=c,normalize=norm,groups=groups,kchunk=kc)
        result['times']=bench({'previous':old.forward,'new':model})
        print('TOTAL',result['times'],flush=True)
        # Independent framework reference, including saved intermediate semantics.
        compiled_ref=torch.compile(ref,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
        yr=compiled_ref(*leaves,mask,ds);yn=model().clone();torch.cuda.synchronize()
        result['reference_error']=error(yn,yr);assert result['reference_error']<.005
        graph,gy=capture(model)
        leaves[0].mul_(.97);leaves[1].add_(.003);leaves[9][0]=0;leaves[10].add_(.017)
        expected=model().clone();graph.replay();torch.cuda.synchronize()
        result['graph_bitexact']=bool(torch.equal(gy,expected));assert result['graph_bitexact']
        yr=compiled_ref(*leaves,mask,ds);result['mutated_reference_error']=error(expected,yr);assert result['mutated_reference_error']<.005
        result['cubins']=[dict(path=k.unit.cubin_path,sha256=hashlib.sha256(Path(k.unit.cubin_path).read_bytes()).hexdigest()) for k in (model.front.k,model.output.k)]
        result['complete']=True;save();print('DONE',result['selected'],flush=True)

if __name__=='__main__':main()

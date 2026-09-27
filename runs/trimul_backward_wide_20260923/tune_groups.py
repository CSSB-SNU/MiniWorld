exec((__import__('pathlib').Path(__file__).parent/'check.py').read_text().split('a = argparse.ArgumentParser()')[0])
D = int(sys.argv[1])
leaves,dy,mask,ds,*_ = setup(D,384)
with torch.no_grad(), T.native_context(leaves[0].device):
    f=F.Forward(leaves,mask,ds);f()
    p=W.Training(*leaves,mask,ds,dy,saved=(f.front.ab,f.tri,f.front.xn))
    plans={g:B1(p,groups=g) for g in (2,3,4,5)}
    def prep(b): return lambda: b.prepare.launch((b.sms,1,1),(128*b.groups,1,1),[b.params],b.smem)
    times=paired({str(g):prep(b) for g,b in plans.items()})
    print({k:v['median_us'] for k,v in times.items()},flush=True)
    (R/f'groups-D{D}.json').write_text(json.dumps(times,indent=2))

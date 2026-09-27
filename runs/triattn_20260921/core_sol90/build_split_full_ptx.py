"""Independent K-ready and V-ready barriers, using existing PipeK storage.

Only fast code changes. K-ready is awaited before QK, V-ready before the
first PV of each stage. Empty-barrier owners and retirement frontiers stay
unchanged. Generic/SAFE use the original combined transaction barrier.
"""
from pathlib import Path
import hashlib,json,shlex,subprocess
HERE=Path(__file__).resolve().parent
variant='splitfullcurrent'
src=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src/'triattn_broadcast.so').read_bytes()).hexdigest()=='9365a4f35e018afed4adba21590ae2c1734bbccf98d1e53904ebfe2537b3e225'
dst=HERE/variant;dst.mkdir(exist_ok=True)
for p in (src/'csrc').rglob('*'):
 if not p.is_file():continue
 s=p.read_text()
 if p.name=='triattn_m1_sm90.cuh':
  old='T::PipeKV::init(shared.pipe_kv, T::kArrivalsKV, 2);'
  assert s.count(old)==1
  s=s.replace(old,'T::PipeKV::init(shared.pipe_kv, T::kArrivalsKV, kFast ? 1 : 2);')
  old='    typename T::PipeB pipe_b(shared.pipe_b);'
  assert s.count(old)==1
  s=s.replace(old,old+'\n    auto& pipe_ready_qk = kFast ? pipe_k : pipe_kv;')
  # Existing full waits/probes all guard QK readiness, including bootstrap.
  s=s.replace('pipe_kv.wait_full(','pipe_ready_qk.wait_full(').replace('pipe_kv.test_full(','pipe_ready_qk.test_full(')
  old='''                    pipe_kv.producer_expect(st, T::kBytesK);
                    copy(params.tma_k.with(*pipe_kv.full_barrier(st), 0), tKgK(_, j, i), tKsK(_, st));'''
  assert s.count(old)==1
  s=s.replace(old,'''                    pipe_ready_qk.producer_expect(st, T::kBytesK);
                    copy(params.tma_k.with(*pipe_ready_qk.full_barrier(st), 0), tKgK(_, j, i), tKsK(_, st));''')
  # Both bootstrap schedules issue PV0 through this exact line; fast uses
  # only the ordinary four-score schedule, but guard both source branches.
  old='    issue_pv(PCb[0], tV0, _0{}, _0{}, _0{});'
  assert s.count(old)==2
  s=s.replace(old,'    if constexpr(kFast){pipe_kv.wait_full(kv0,0);}\n'+old)
  old='        if constexpr (kFast) { issue_pv_prefenced(PCb[pb], tV, Int<hp>{}, Int<cp>{}, Int<(tp & 1) * R>{}); }'
  assert s.count(old)==1
  s=s.replace(old,'''        if constexpr (kFast) {
            if constexpr((ep&7)==0){
                pipe_kv.wait_full(kv0+(tp&1)*R,(tp==2)?(pp^1u):pp);
            }
            issue_pv_prefenced(PCb[pb], tV, Int<hp>{}, Int<cp>{}, Int<(tp & 1) * R>{});
        }''')
 s=s.replace('ta_core_broadcast','ta_sol_'+variant)
 q=dst/p.relative_to(src);q.parent.mkdir(parents=True,exist_ok=True);q.write_text(s)
inst=dst/'inst_m1';inst.mkdir(exist_ok=True)
cu=inst/'m1_1073741824.cu'
cu.write_text('#include "launch_m1.cuh"\nnamespace ta_sol_'+variant+' {\nvoid run_hot(Args const& a) { launch_m1<Traits<1073741824>>(a); }\n}\n')
ninja=(HERE/'build/triattn_sol_baseqthree1024/build.ninja').read_text()
line=next(l.split(' = ',1)[1] for l in ninja.splitlines() if l.startswith('cuda_cflags = '))
flags=[]
for f in shlex.split(line):
 if f=='--keep' or f.startswith('--keep-dir='):continue
 f=f.replace(str(HERE/'baseqthree1024'),str(dst)).replace(str(HERE/'build/triattn_sol_baseqthree1024/inst_m1'),str(inst))
 flags.append(f)
ptx=dst/'m1_1073741824.ptx'
command=['/usr/local/cuda-12.9/bin/nvcc']+flags+['--ptx',str(cu),'-o',str(ptx)]
(dst/'command.json').write_text(json.dumps(command,indent=2)+'\n')
subprocess.run(command,check=True)
print('BUILT',variant,ptx,flush=True)

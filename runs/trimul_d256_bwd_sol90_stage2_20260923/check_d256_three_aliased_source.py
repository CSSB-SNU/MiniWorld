"""Three compute groups with aliased slots and BF16-packed register retention."""
from pathlib import Path
body=Path(__file__).with_name('check_d256_pending_dw.py').read_text()
body=body.replace('from d256_pending_dw_source import PendingDwSource','from d256_three_aliased_source import ThreeAliasedSource')
body=body.replace('d256-pending-dw','d256-three-aliased-source').replace('for readwait in (False,True):','for readwait in (64,80):')
body=body.replace('op=PendingDwSource(plan,readwait)', '''try:op=ThreeAliasedSource(plan,readwait)
        except RuntimeError as exc:
            record['candidates'].append(dict(producer_regs=readwait,compile_failure=str(exc)))
            print('COMPILE_FAILURE',str(exc),flush=True)
            continue''')
body=body.replace('local_bytes=op.local_bytes)','local_bytes=op.local_bytes,occupancy=op.occupancy)')
exec(compile(body,__file__,'exec'))

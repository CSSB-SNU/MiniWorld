"""Compare N256 dW without changing the original loader or CTA occupancy."""
from pathlib import Path
body=Path(__file__).with_name('check_d256_pending_dw.py').read_text()
body=body.replace('from d256_pending_dw_source import PendingDwSource','from d256_full_n_dw_source import FullNDwSource')
body=body.replace('d256-pending-dw','d256-full-n-dw-source')
body=body.replace('op=PendingDwSource(plan,readwait)', '''try:op=FullNDwSource(plan,readwait)
        except RuntimeError as exc:
            record['candidates'].append(dict(offset=readwait,compile_failure=str(exc)))
            print('COMPILE_FAILURE',str(exc),flush=True)
            continue''')
exec(compile(body,__file__,'exec'))

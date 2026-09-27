"""Measure the arithmetic cost of three-CTA half-column input dW."""
from pathlib import Path
body=Path(__file__).with_name('check_d256_pending_dw.py').read_text()
body=body.replace('from d256_pending_dw_source import PendingDwSource','from d256_half_columns_source import HalfColumnsSource')
body=body.replace('d256-pending-dw','d256-half-columns-source')
body=body.replace('op=PendingDwSource(plan,readwait)', '''try:op=HalfColumnsSource(plan,readwait)
        except RuntimeError as exc:
            record['candidates'].append(dict(packed=readwait,compile_failure=str(exc)))
            print('COMPILE_FAILURE',str(exc),flush=True)
            continue''')
body=body.replace('local_bytes=op.local_bytes)','local_bytes=op.local_bytes,occupancy=op.occupancy)')
exec(compile(body,__file__,'exec'))

"""Pilot barrier placement; winning candidates still require all sanitizer gates."""
from pathlib import Path
body=Path(__file__).with_name('check_d256_pending_dw.py').read_text()
body=body.replace('from d256_pending_dw_source import PendingDwSource','from d256_barrier_source import BarrierSource')
body=body.replace('d256-pending-dw','d256-barrier-source')
body=body.replace('for readwait in (False,True):',"for readwait in ('input','proxy','both'):")
body=body.replace('op=PendingDwSource(plan,readwait)','op=BarrierSource(plan,readwait)')
exec(compile(body,__file__,'exec'))

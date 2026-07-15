"""Step-by-step LR comparison: MPFULL (inverse_sqrt, max_lr=3e-3) vs
SWAFIXINIT (step-decay, max_lr=1e-4 default). Uses the repo's own schedulers.

Resume semantics (verified from code): the LR scheduler is created fresh and
its state is NOT restored on --no-ckpt-strict resume (scheduler_state skipped),
so scheduler step counts from 0 at resume. global_step is restored (->42000,
hence "epoch 421" labels) but does NOT drive the LR lambda. ~100 optimizer
steps per epoch (global_step +100/epoch), so scheduler_step ~= (epoch-420)*100.
"""
import torch
from miniworld.utils.utils import (
    get_inverse_sqrt_scheduler_with_warmup,
    get_step_decay_scheduler_with_warmup,
)

WARMUP = 5000
# MPFULL config: max_lr 3e-3, inverse_sqrt, lr_decay_ref_steps (t_ref) = 5000
MP_MAXLR, MP_TREF = 3.0e-3, 5000
# SWAFIXINIT: no lr settings -> defaults: max_lr 1e-4, schedule "env"; neither
# launch set EDM2_INV_SQRT_LR=1 -> resolves to "step" decay (decay_steps=5e4,
# decay_factor=0.95 defaults).
SF_MAXLR, SF_DECAY_STEPS, SF_FACTOR = 1.0e-4, int(5e4), 0.95


def build(max_lr, kind):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=max_lr)  # base_lr = max_lr; lambda multiplies it
    if kind == "inverse_sqrt":
        sch = get_inverse_sqrt_scheduler_with_warmup(opt, warmup_steps=WARMUP, decay_ref_steps=MP_TREF)
    else:
        sch = get_step_decay_scheduler_with_warmup(opt, warmup_steps=WARMUP, decay_steps=SF_DECAY_STEPS, decay_factor=SF_FACTOR)
    return opt, sch


def lr_at(steps, max_lr, kind):
    opt, sch = build(max_lr, kind)
    out = {}
    want = set(steps)
    for s in range(max(steps) + 1):
        if s in want:
            out[s] = opt.param_groups[0]["lr"]
        opt.step(); sch.step()
    return out


STEPS = [0, 100, 300, 600, 1000, 1400, 2000, 3000, 5000, 7000, 9400, 14000, 18800]
mp = lr_at(STEPS, MP_MAXLR, "inverse_sqrt")
sf = lr_at(STEPS, SF_MAXLR, "step")

print(f"{'sch_step':>8} {'~epoch':>7} {'MPFULL lr':>12} {'SWAFIX lr':>12} {'ratio':>7}  phase")
print("-" * 66)
for s in STEPS:
    ep = 420 + s / 100
    ratio = mp[s] / sf[s] if sf[s] else float("nan")
    phase = "warmup" if s < WARMUP else "decay"
    note = "  <-- MPFULL run ended (~ep434)" if s == 1400 else ("  <-- SWAFIX ~final" if s == 18800 else "")
    print(f"{s:>8} {ep:>7.0f} {mp[s]:>12.3e} {sf[s]:>12.3e} {ratio:>6.1f}x  {phase}{note}")

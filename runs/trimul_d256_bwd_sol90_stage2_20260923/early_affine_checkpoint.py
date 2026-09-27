"""Shared attachment for the qualified-candidate early affine initialization."""
from early_affine_init import IndependentOutputLN
from early_both_affine_init import EarlyBothAffineGate, IndependentInputLN


def attach(plan, ordered=False, narrow=False):
    p=plan.p
    assert p.n==384
    owner=plan.b1 if p.D==256 else plan
    original_ln=owner.ln
    output_scale={256:2,384:1,512:4}[p.D]
    if ordered:
        from wide_ordered_pair_ln import OrderedPairLN
        assert p.D==512
        original_ln=OrderedPairLN(p,16,3,narrow)
        output_scale=1
    gate=EarlyBothAffineGate(plan)
    ln=IndependentOutputLN(p,original_ln,output_scale)
    input_ln=IndependentInputLN(plan,1,p.D!=384)
    plan.prefix_gate=gate
    if p.D==256:plan.b1.prepare=gate
    else:plan.b1.epi=gate
    owner.ln=ln
    plan.dx.reduce_only=input_ln
    plan.artifacts.extend((gate.cubin,ln.cubin,input_ln.cubin))
    plan.early_affine_config=dict(enabled=True,initialization='preceding_gate',output_grid=ln.grid,input_grid=input_ln.grid,packed_weights=p.D!=384,ordered_pair=ordered,pair_local_barriers=narrow)


def configuration(plan):
    return getattr(plan,'early_affine_config',dict(enabled=False))

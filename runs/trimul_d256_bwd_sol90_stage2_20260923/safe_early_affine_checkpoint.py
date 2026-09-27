"""Safe ordered-pair attachment with disjoint prefix/final reduction storage."""
from early_affine_checkpoint import attach as attach_early, configuration
from early_affine_init import IndependentOutputLN
from wide_ordered_pair_ln_safe import SafeOrderedPairLN


def attach(plan):
    attach_early(plan)
    if plan.p.D==512:
        base=SafeOrderedPairLN(plan.p,16,3,False)
        plan.ln=IndependentOutputLN(plan.p,base,1)
        plan.artifacts[-2]=plan.ln.cubin
        plan.early_affine_config.update(ordered_pair=True,output_grid=plan.ln.grid,pair_sum_storage='disjoint_prefix_and_final')

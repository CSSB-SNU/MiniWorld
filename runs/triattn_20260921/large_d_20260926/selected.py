"""Qualified experimental wide entry; does not alter installed engine dispatch.

Pair/total-QKV width256 uses native attention plus projection-dgrad fusion.
Width512 uses native attention and keeps the original projection implementation.
The original module still owns LayerNorm, gate, parameter gradients and residual.
"""
import candidate
import front_candidate


def attach(model):
    C=model.to_query.weight.shape[0]
    assert C in (256,512) and model.n_head==4
    assert model.use_self_attention and not model.use_qk_norm
    for weight in (model.to_query.weight,model.to_key.weight,model.to_value.weight,
                   model.to_gate.weight,model.to_out.weight):
        assert tuple(weight.shape)==(C,C)
    assert tuple(model.to_bias.weight.shape)==(4,C)
    return front_candidate.attach(model) if C==256 else candidate.attach(model)

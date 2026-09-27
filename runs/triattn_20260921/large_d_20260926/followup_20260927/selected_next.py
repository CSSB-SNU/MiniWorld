"""Explicit follow-up: compact bias partials, plus width512 GEMM accumulation."""
import experiments
def attach(model):
    C=model.to_query.weight.shape[0]
    return experiments.attach(model,kind='prior' if C==256 else 'accumulate_fast',compact=True)

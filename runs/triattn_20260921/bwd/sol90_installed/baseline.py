import baseline_api
from miniworld_engine.modules import TriangleAttention
from miniworld_engine.kernels.triangle_attention.cuda import bias_backward

def verify():
    assert baseline_api._extension() is not bias_backward._extension(), 'extension alias'

def core(model,q,k,v,b,backend):
    if baseline_api.can_use(q,k,v,b):
        return baseline_api.attention(q,k,v,b)
    return TriangleAttention._kernel_triangle_attention(model,q,k,v,b,backend)

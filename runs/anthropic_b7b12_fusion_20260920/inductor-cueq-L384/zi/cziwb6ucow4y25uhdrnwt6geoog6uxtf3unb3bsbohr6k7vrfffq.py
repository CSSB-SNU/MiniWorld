# AOT ID: ['0_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/inductor-cueq-L384/fx/cfxxtk47qhmwioy3m35vf6zvqmwljmhd7p4jrswl6oyxo6umzrk7.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %primals_4 : Tensor "bf16[256, 128][128, 1]cuda:0" = PlaceHolder[target=primals_4]
#   %primals_5 : Tensor "bf16[256, 128][128, 1]cuda:0" = PlaceHolder[target=primals_5]
#   %cat : Tensor "bf16[512, 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%primals_4, %primals_5],), kwargs = {})
#   return %cat
triton_poi_fused_cat_0 = async_compile.triton('triton_poi_fused_cat_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 393216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = xindex // 128
    x0 = (xindex % 128)
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 128*(x1)), tmp4, other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 512, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 128*((-256) + x1)), tmp6, other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/inductor-cueq-L384/7v/c7v5qczzm46phjq63otojviicpplwl5wcggv76oldiz7pxnxpgc5.py
# Topologically Sorted Source Nodes: [outgoing, incoming, tri], Original ATen: [aten.view, aten.permute, aten.cat]
# Source node to ATen node mapping:
#   incoming => permute_9, view_10, view_11
#   outgoing => permute_4, view_6, view_7
#   tri => cat_2
# Graph fragment:
#   %bmm : Tensor "bf16[128, 384, 384][147456, 384, 1]cuda:0" = PlaceHolder[target=bmm]
#   %bmm_1 : Tensor "bf16[128, 384, 384][147456, 384, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %view_6 : Tensor "bf16[128, 384, 1, 1, 384][147456, 384, 384, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [128, 384, 1, 1, 384]), kwargs = {})
#   %permute_4 : Tensor "bf16[128, 1, 384, 384, 1][147456, 384, 384, 1, 384]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [0, 3, 1, 4, 2]), kwargs = {})
#   %view_7 : Tensor "bf16[128, 1, 384, 384][147456, 384, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_4, [128, 1, 384, 384]), kwargs = {})
#   %view_10 : Tensor "bf16[128, 384, 1, 1, 384][147456, 384, 384, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_1, [128, 384, 1, 1, 384]), kwargs = {})
#   %permute_9 : Tensor "bf16[128, 1, 384, 384, 1][147456, 384, 384, 1, 384]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_10, [0, 3, 1, 4, 2]), kwargs = {})
#   %view_11 : Tensor "bf16[128, 1, 384, 384][147456, 384, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_9, [128, 1, 384, 384]), kwargs = {})
#   %cat_2 : Tensor "bf16[256, 1, 384, 384][147456, 147456, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_7, %view_11],), kwargs = {})
#   return %cat_2
triton_poi_fused_cat_permute_view_1 = async_compile.triton('triton_poi_fused_cat_permute_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_permute_view_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 226492416}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_permute_view_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 37748736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = xindex // 147456
    x0 = (xindex % 147456)
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 147456*(x1)), tmp4, other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 256, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 147456*((-128) + x1)), tmp6, other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/inductor-cueq-L384/jg/cjgea5lsclbzbzicntdxomvy4augh5uaprfexxiwyimp5a3b2jve.py
# Topologically Sorted Source Nodes: [linear, sigmoid, linear_1, update, mul_1, add], Original ATen: [aten._unsafe_view, aten.sigmoid, aten.mul, aten.add]
# Source node to ATen node mapping:
#   add => add
#   linear => view_15
#   linear_1 => view_17
#   mul_1 => mul_1
#   sigmoid => sigmoid
#   update => mul
# Graph fragment:
#   %mm : Tensor "bf16[147456, 128][128, 1]cuda:0" = PlaceHolder[target=mm]
#   %mm_1 : Tensor "bf16[147456, 128][128, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %primals_13 : Tensor "bf16[1, 1, 384, 128][49152, 49152, 128, 1]cuda:0" = PlaceHolder[target=primals_13]
#   %primals_1 : Tensor "bf16[1, 384, 384, 128][18874368, 49152, 128, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %view_15 : Tensor "bf16[1, 384, 384, 128][18874368, 49152, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [1, 384, 384, 128]), kwargs = {})
#   %sigmoid : Tensor "bf16[1, 384, 384, 128][18874368, 49152, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_15,), kwargs = {})
#   %view_17 : Tensor "bf16[1, 384, 384, 128][18874368, 49152, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [1, 384, 384, 128]), kwargs = {})
#   %mul : Tensor "bf16[1, 384, 384, 128][18874368, 49152, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid, %view_17), kwargs = {})
#   %mul_1 : Tensor "bf16[1, 384, 384, 128][18874368, 49152, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_13), kwargs = {})
#   %add : Tensor "bf16[1, 384, 384, 128][18874368, 49152, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_1), kwargs = {})
#   return %add
triton_poi_fused__unsafe_view_add_mul_sigmoid_2 = async_compile.triton('triton_poi_fused__unsafe_view_add_mul_sigmoid_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_mul_sigmoid_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 188841984}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_mul_sigmoid_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18874368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 49152)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp1 = tl.sigmoid(tmp0)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 + tmp6
    tl.store(out_ptr0 + (x2), tmp7, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13 = args
        args.clear()
        assert_size_stride(primals_1, (1, 384, 384, 128), (18874368, 49152, 128, 1))
        assert_size_stride(primals_2, (128, ), (1, ))
        assert_size_stride(primals_3, (128, ), (1, ))
        assert_size_stride(primals_4, (256, 128), (128, 1))
        assert_size_stride(primals_5, (256, 128), (128, 1))
        assert_size_stride(primals_6, (256, 128), (128, 1))
        assert_size_stride(primals_7, (256, 128), (128, 1))
        assert_size_stride(primals_8, (1, 384, 384), (147456, 384, 1))
        assert_size_stride(primals_9, (256, ), (1, ))
        assert_size_stride(primals_10, (256, ), (1, ))
        assert_size_stride(primals_11, (128, 128), (128, 1))
        assert_size_stride(primals_12, (128, 256), (256, 1))
        assert_size_stride(primals_13, (1, 1, 384, 128), (49152, 49152, 128, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            # Topologically Sorted Source Nodes: [x, layer_norm_transpose], Original ATen: [aten.view, cuequivariance.layer_norm_transpose]
            buf0 = torch.ops.cuequivariance.layer_norm_transpose.default(reinterpret_tensor(primals_1, (1, 147456, 128), (18874368, 128, 1), 0), primals_2, primals_3, 1e-05, True, 0)
            del primals_3
            buf1 = buf0[0]
            assert_size_stride(buf1, (1, 147456, 128), (18874368, 128, 1), 'torch.ops.cuequivariance.layer_norm_transpose.default')
            assert_alignment(buf1, 16, 'torch.ops.cuequivariance.layer_norm_transpose.default')
            buf2 = buf0[1]
            assert_size_stride(buf2, (1, 147456), (147456, 1), 'torch.ops.cuequivariance.layer_norm_transpose.default')
            assert_alignment(buf2, 16, 'torch.ops.cuequivariance.layer_norm_transpose.default')
            buf3 = buf0[2]
            assert_size_stride(buf3, (1, 147456), (147456, 1), 'torch.ops.cuequivariance.layer_norm_transpose.default')
            assert_alignment(buf3, 16, 'torch.ops.cuequivariance.layer_norm_transpose.default')
            del buf0
            buf4 = empty_strided_cuda((512, 128), (128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_0.run(primals_4, primals_5, buf4, 65536, stream=stream0)
            del primals_4
            del primals_5
            buf5 = empty_strided_cuda((512, 128), (128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_0.run(primals_6, primals_7, buf5, 65536, stream=stream0)
            del primals_6
            del primals_7
            # Topologically Sorted Source Nodes: [xn, x1, out_1], Original ATen: [aten.view, cuequivariance.fused_gated_dual_gemm]
            buf6 = torch.ops.cuequivariance.fused_gated_dual_gemm.default(reinterpret_tensor(buf1, (147456, 128), (128, 1), 0), None, buf4, buf5, None, None, primals_8, True, precision=0)
            buf7 = buf6
            assert_size_stride(buf7, (512, 147456), (147456, 1), 'torch.ops.cuequivariance.fused_gated_dual_gemm.default')
            assert_alignment(buf7, 16, 'torch.ops.cuequivariance.fused_gated_dual_gemm.default')
            del buf6
            buf8 = empty_strided_cuda((128, 384, 384), (147456, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [ab, chunk, getitem_5, getitem_6, outgoing], Original ATen: [aten.view, aten.split, aten.slice, aten.unsqueeze, aten.permute, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf7, (128, 384, 384), (147456, 384, 1), 0), reinterpret_tensor(buf7, (128, 384, 384), (147456, 1, 384), 37748736), out=buf8)
            buf9 = empty_strided_cuda((128, 384, 384), (147456, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [ab, chunk, getitem_7, getitem_8, incoming], Original ATen: [aten.view, aten.split, aten.slice, aten.unsqueeze, aten.permute, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf7, (128, 384, 384), (147456, 1, 384), 18874368), reinterpret_tensor(buf7, (128, 384, 384), (147456, 384, 1), 56623104), out=buf9)
            buf10 = empty_strided_cuda((256, 1, 384, 384), (147456, 1, 384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [outgoing, incoming, tri], Original ATen: [aten.view, aten.permute, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_permute_view_1.run(buf8, buf9, buf10, 37748736, stream=stream0)
            del buf8
            del buf9
            # Topologically Sorted Source Nodes: [outgoing, incoming, tri, x_1, layer_norm_transpose_1], Original ATen: [aten.view, aten.permute, aten.cat, cuequivariance.layer_norm_transpose]
            buf11 = torch.ops.cuequivariance.layer_norm_transpose.default(reinterpret_tensor(buf10, (256, 1, 147456), (147456, 147456, 1), 0), primals_9, primals_10, 1e-05, True, 3)
            del primals_10
            buf12 = buf11[0]
            assert_size_stride(buf12, (1, 147456, 256), (37748736, 256, 1), 'torch.ops.cuequivariance.layer_norm_transpose.default')
            assert_alignment(buf12, 16, 'torch.ops.cuequivariance.layer_norm_transpose.default')
            buf13 = buf11[1]
            assert_size_stride(buf13, (1, 147456), (147456, 1), 'torch.ops.cuequivariance.layer_norm_transpose.default')
            assert_alignment(buf13, 16, 'torch.ops.cuequivariance.layer_norm_transpose.default')
            buf14 = buf11[2]
            assert_size_stride(buf14, (1, 147456), (147456, 1), 'torch.ops.cuequivariance.layer_norm_transpose.default')
            assert_alignment(buf14, 16, 'torch.ops.cuequivariance.layer_norm_transpose.default')
            del buf11
            buf15 = empty_strided_cuda((147456, 128), (128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [xn, linear], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1, (147456, 128), (128, 1), 0), reinterpret_tensor(primals_11, (128, 128), (1, 128), 0), out=buf15)
            buf16 = empty_strided_cuda((147456, 128), (128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [norm, linear_1], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf12, (147456, 256), (256, 1), 0), reinterpret_tensor(primals_12, (256, 128), (1, 256), 0), out=buf16)
            buf17 = empty_strided_cuda((1, 384, 384, 128), (18874368, 49152, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear, sigmoid, linear_1, update, mul_1, add], Original ATen: [aten._unsafe_view, aten.sigmoid, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_sigmoid_2.run(buf15, buf16, primals_13, primals_1, buf17, 18874368, stream=stream0)
        return (buf17, primals_2, primals_8, primals_9, primals_13, reinterpret_tensor(primals_1, (1, 147456, 128), (18874368, 128, 1), 0), buf2, buf3, buf4, buf5, reinterpret_tensor(buf1, (147456, 128), (128, 1), 0), reinterpret_tensor(buf10, (256, 1, 147456), (147456, 147456, 1), 0), buf13, buf14, reinterpret_tensor(buf1, (147456, 128), (128, 1), 0), buf15, reinterpret_tensor(buf12, (147456, 256), (256, 1), 0), buf16, primals_12, primals_11, reinterpret_tensor(buf7, (128, 384, 384), (147456, 384, 1), 18874368), reinterpret_tensor(buf7, (128, 384, 384), (147456, 1, 384), 56623104), reinterpret_tensor(buf7, (128, 384, 384), (147456, 1, 384), 0), reinterpret_tensor(buf7, (128, 384, 384), (147456, 384, 1), 37748736), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 384, 384, 128), (18874368, 49152, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_5 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_6 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_7 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_8 = rand_strided((1, 384, 384), (147456, 384, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_9 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_12 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_13 = rand_strided((1, 1, 384, 128), (49152, 49152, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

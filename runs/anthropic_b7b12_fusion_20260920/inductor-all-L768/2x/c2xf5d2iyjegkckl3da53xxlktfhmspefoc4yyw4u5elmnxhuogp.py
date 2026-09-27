# AOT ID: ['2_backward']
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


# kernel path: /home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/inductor-all-L768/fs/cfs4byg6ozgxeyroarhhiqsf237m3nycxragae7vmpo5eeyu7dc2.py
# Topologically Sorted Source Nodes: [mul_2, linear, sigmoid, mul_3, linear_1, mul_4, convert_element_type_12, convert_element_type_13, sub, mul_5, mul_6, convert_element_type_14], Original ATen: [aten.mul, aten._unsafe_view, aten.sigmoid, aten.sigmoid_backward]
# Source node to ATen node mapping:
#   convert_element_type_12 => convert_element_type_12
#   convert_element_type_13 => convert_element_type_13
#   convert_element_type_14 => convert_element_type_14
#   linear => view_15
#   linear_1 => view_17
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   sigmoid => sigmoid
#   sub => sub
# Graph fragment:
#   %tangents_1 : Tensor "bf16[1, 768, 768, 128][75497472, 98304, 128, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %primals_13 : Tensor "bf16[1, 1, 768, 128][98304, 98304, 128, 1]cuda:0" = PlaceHolder[target=primals_13]
#   %mm : Tensor "bf16[589824, 128][128, 1]cuda:0" = PlaceHolder[target=mm]
#   %mm_1 : Tensor "bf16[589824, 128][128, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %mul_2 : Tensor "bf16[1, 768, 768, 128][75497472, 98304, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %primals_13), kwargs = {})
#   %view_15 : Tensor "bf16[1, 768, 768, 128][75497472, 98304, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [1, 768, 768, 128]), kwargs = {})
#   %sigmoid : Tensor "bf16[1, 768, 768, 128][75497472, 98304, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_15,), kwargs = {})
#   %mul_3 : Tensor "bf16[1, 768, 768, 128][75497472, 98304, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %sigmoid), kwargs = {})
#   %view_17 : Tensor "bf16[1, 768, 768, 128][75497472, 98304, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [1, 768, 768, 128]), kwargs = {})
#   %mul_4 : Tensor "bf16[1, 768, 768, 128][75497472, 98304, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %view_17), kwargs = {})
#   %convert_element_type_12 : Tensor "f32[1, 768, 768, 128][75497472, 98304, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4, torch.float32), kwargs = {})
#   %convert_element_type_13 : Tensor "f32[1, 768, 768, 128][75497472, 98304, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sigmoid, torch.float32), kwargs = {})
#   %sub : Tensor "f32[1, 768, 768, 128][75497472, 98304, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type_13), kwargs = {})
#   %mul_5 : Tensor "f32[1, 768, 768, 128][75497472, 98304, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_13, %sub), kwargs = {})
#   %mul_6 : Tensor "f32[1, 768, 768, 128][75497472, 98304, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_12, %mul_5), kwargs = {})
#   %convert_element_type_14 : Tensor "bf16[1, 768, 768, 128][75497472, 98304, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_6, torch.bfloat16), kwargs = {})
#   return %mul_3,%convert_element_type_14
triton_poi_fused__unsafe_view_mul_sigmoid_sigmoid_backward_0 = async_compile.triton('triton_poi_fused__unsafe_view_mul_sigmoid_sigmoid_backward_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_mul_sigmoid_sigmoid_backward_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1057161216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_mul_sigmoid_sigmoid_backward_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75497472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 98304)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tmp7 = tmp2 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp4.to(tl.float32)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp9 * tmp11
    tmp13 = tmp8 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(in_out_ptr0 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/inductor-all-L768/fu/cfugdcjeqradptnrgculs5cdjjtth2rgzcrf4q5so5q77e3surjw.py
# Topologically Sorted Source Nodes: [view_23, sum_1], Original ATen: [aten.view, aten.sum]
# Source node to ATen node mapping:
#   sum_1 => sum_1
#   view_23 => view_23
# Graph fragment:
#   %getitem_9 : Tensor "f32[1, 9216, 256][2359296, 256, 1]cuda:0" = PlaceHolder[target=getitem_9]
#   %view_23 : Tensor "f32[9216, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_9, [-1, 256]), kwargs = {})
#   %sum_1 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_23, [0]), kwargs = {})
#   return %buf10
triton_red_fused_sum_view_1 = async_compile.triton('triton_red_fused_sum_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 9584640, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_view_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 18432
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 32768*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/inductor-all-L768/6y/c6ylupqji5o5drbp3rahuxevqathmbbnv7aeugdfeuefuhm67xf3.py
# Topologically Sorted Source Nodes: [view_23, sum_1], Original ATen: [aten.view, aten.sum]
# Source node to ATen node mapping:
#   sum_1 => sum_1
#   view_23 => view_23
# Graph fragment:
#   %buf10 : Tensor "f32[256, 72][1, 256]cuda:0" = PlaceHolder[target=buf10]
#   %view_23 : Tensor "f32[9216, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_9, [-1, 256]), kwargs = {})
#   %sum_1 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_23, [0]), kwargs = {})
#   return %sum_1
triton_red_fused_sum_view_2 = async_compile.triton('triton_red_fused_sum_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 75776, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_view_2(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 72
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/inductor-all-L768/xe/cxek7ra4bfagehryaf2kf3kisykcf3orpcwrt2e2rsdtsciju2ym.py
# Topologically Sorted Source Nodes: [view_28, permute_23, view_29, permute_24, permute_25, squeeze, permute_26, squeeze_1, full_default, view_32, permute_30, view_33, permute_31, permute_32, squeeze_2, permute_33, squeeze_3, add_1, add_2, cat_3], Original ATen: [aten.view, aten.permute, aten.squeeze, aten.slice_backward, aten.add, aten.cat]
# Source node to ATen node mapping:
#   add_1 => add_1
#   add_2 => add_2
#   cat_3 => cat_3
#   full_default => full_default
#   permute_23 => permute_23
#   permute_24 => permute_24
#   permute_25 => permute_25
#   permute_26 => permute_26
#   permute_30 => permute_30
#   permute_31 => permute_31
#   permute_32 => permute_32
#   permute_33 => permute_33
#   squeeze => squeeze
#   squeeze_1 => squeeze_1
#   squeeze_2 => squeeze_2
#   squeeze_3 => squeeze_3
#   view_28 => view_28
#   view_29 => view_29
#   view_32 => view_32
#   view_33 => view_33
# Graph fragment:
#   %bmm_3 : Tensor "bf16[128, 768, 768][589824, 768, 1]cuda:0" = PlaceHolder[target=bmm_3]
#   %bmm_5 : Tensor "bf16[128, 768, 768][589824, 768, 1]cuda:0" = PlaceHolder[target=bmm_5]
#   %bmm_2 : Tensor "bf16[128, 768, 768][589824, 768, 1]cuda:0" = PlaceHolder[target=bmm_2]
#   %bmm_4 : Tensor "bf16[128, 768, 768][589824, 768, 1]cuda:0" = PlaceHolder[target=bmm_4]
#   %view_28 : Tensor "bf16[128, 768, 1, 768, 1][589824, 768, 768, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_2, [128, 768, 1, 768, 1]), kwargs = {})
#   %permute_23 : Tensor "bf16[128, 1, 1, 768, 768][589824, 768, 1, 1, 768]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 4, 3, 1]), kwargs = {})
#   %view_29 : Tensor "bf16[128, 768, 768, 1, 1][589824, 768, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_3, [128, 768, 768, 1, 1]), kwargs = {})
#   %permute_24 : Tensor "bf16[128, 1, 768, 1, 768][589824, 1, 768, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_29, [0, 3, 1, 4, 2]), kwargs = {})
#   %permute_25 : Tensor "bf16[128, 1, 768, 768, 1][589824, 768, 768, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_23, [0, 1, 4, 3, 2]), kwargs = {})
#   %squeeze : Tensor "bf16[128, 1, 768, 768][589824, 768, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_25, 4), kwargs = {})
#   %permute_26 : Tensor "bf16[128, 1, 768, 768, 1][589824, 1, 1, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_24, [0, 1, 4, 2, 3]), kwargs = {})
#   %squeeze_1 : Tensor "bf16[128, 1, 768, 768][589824, 1, 1, 768]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_26, 4), kwargs = {})
#   %full_default : Tensor "bf16[256, 1, 768, 768][589824, 589824, 768, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([256, 1, 768, 768], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default : Tensor "bf16[256, 1, 768, 768][589824, 589824, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default, %squeeze, 0, 128, 9223372036854775807), kwargs = {})
#   %slice_scatter_default_1 : Tensor "bf16[256, 1, 768, 768][589824, 589824, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default, %squeeze_1, 0, 128, 9223372036854775807), kwargs = {})
#   %view_32 : Tensor "bf16[128, 768, 1, 768, 1][589824, 768, 768, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_4, [128, 768, 1, 768, 1]), kwargs = {})
#   %permute_30 : Tensor "bf16[128, 1, 1, 768, 768][589824, 768, 1, 1, 768]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_32, [0, 2, 4, 3, 1]), kwargs = {})
#   %view_33 : Tensor "bf16[128, 768, 768, 1, 1][589824, 768, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_5, [128, 768, 768, 1, 1]), kwargs = {})
#   %permute_31 : Tensor "bf16[128, 1, 768, 1, 768][589824, 1, 768, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_33, [0, 3, 1, 4, 2]), kwargs = {})
#   %permute_32 : Tensor "bf16[128, 1, 768, 768, 1][589824, 768, 1, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_30, [0, 1, 3, 4, 2]), kwargs = {})
#   %squeeze_2 : Tensor "bf16[128, 1, 768, 768][589824, 768, 1, 768]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_32, 4), kwargs = {})
#   %permute_33 : Tensor "bf16[128, 1, 768, 768, 1][589824, 1, 768, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_31, [0, 1, 2, 4, 3]), kwargs = {})
#   %squeeze_3 : Tensor "bf16[128, 1, 768, 768][589824, 1, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_33, 4), kwargs = {})
#   %slice_scatter_default_2 : Tensor "bf16[256, 1, 768, 768][589824, 589824, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default, %squeeze_2, 0, 0, 128), kwargs = {})
#   %add_1 : Tensor "bf16[256, 1, 768, 768][589824, 589824, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default, %slice_scatter_default_2), kwargs = {})
#   %slice_scatter_default_3 : Tensor "bf16[256, 1, 768, 768][589824, 589824, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default, %squeeze_3, 0, 0, 128), kwargs = {})
#   %add_2 : Tensor "bf16[256, 1, 768, 768][589824, 589824, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_1, %slice_scatter_default_3), kwargs = {})
#   %cat_3 : Tensor "bf16[512, 1, 768, 768][589824, 589824, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_2, %add_1],), kwargs = {})
#   return %cat_3
triton_poi_fused_add_cat_permute_slice_backward_squeeze_view_3 = async_compile.triton('triton_poi_fused_add_cat_permute_slice_backward_squeeze_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_permute_slice_backward_squeeze_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 301989888, 'x': 1509949440}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_permute_slice_backward_squeeze_view_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 393216
    xnumel = 768
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y1 = yindex // 768
    x2 = xindex
    y0 = (yindex % 768)
    y3 = yindex
    tmp0 = y1
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.broadcast_to(y1, [YBLOCK, XBLOCK])
    tmp6 = tl.full([1, 1], 128, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tmp7 & tmp4
    tmp9 = tl.load(in_ptr0 + ((-75497472) + y0 + 768*x2 + 589824*(y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = 0.0
    tmp11 = tl.where(tmp7, tmp9, tmp10)
    tmp12 = tmp5 < tmp6
    tmp13 = tmp12 & tmp4
    tmp14 = tl.load(in_ptr1 + (x2 + 768*y0 + 589824*(y1)), tmp13 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tl.where(tmp12, tmp14, tmp10)
    tmp16 = tmp11 + tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp4, tmp16, tmp17)
    tmp19 = tmp0 >= tmp3
    tmp20 = tl.full([1, 1], 512, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tl.broadcast_to((-256) + y1, [YBLOCK, XBLOCK])
    tmp23 = tl.full([1, 1], 128, tl.int64)
    tmp24 = tmp22 >= tmp23
    tmp25 = tmp24 & tmp19
    tmp26 = tl.load(in_ptr2 + ((-75497472) + x2 + 768*y0 + 589824*((-256) + y1)), tmp25 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp27 = 0.0
    tmp28 = tl.where(tmp24, tmp26, tmp27)
    tmp29 = tmp22 < tmp23
    tmp30 = tmp29 & tmp19
    tmp31 = tl.load(in_ptr3 + (y0 + 768*x2 + 589824*((-256) + y1)), tmp30 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tl.where(tmp29, tmp31, tmp27)
    tmp33 = tmp28 + tmp32
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp19, tmp33, tmp34)
    tmp36 = tl.where(tmp4, tmp18, tmp35)
    tl.store(out_ptr0 + (x2 + 768*y3), tmp36, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/inductor-all-L768/js/cjsnqwvomrj4acg6czvxu733m3ihyuqa235rz4cq3cdjkxaxaewx.py
# Topologically Sorted Source Nodes: [view_21, add_3, view_36, add_4, view_37, layer_norm_transpose_bwd_1], Original ATen: [aten.view, aten.add, cuequivariance.layer_norm_transpose_bwd]
# Source node to ATen node mapping:
#   add_3 => add_3
#   add_4 => add_4
#   layer_norm_transpose_bwd_1 => layer_norm_transpose_bwd_1
#   view_21 => view_21
#   view_36 => view_36
#   view_37 => view_37
# Graph fragment:
#   %mm_5 : Tensor "bf16[589824, 128][128, 1]cuda:0" = PlaceHolder[target=mm_5]
#   %mm_8 : Tensor "bf16[589824, 128][128, 1]cuda:0" = PlaceHolder[target=mm_8]
#   %mm_9 : Tensor "bf16[589824, 128][128, 1]cuda:0" = PlaceHolder[target=mm_9]
#   %view_21 : Tensor "bf16[1, 768, 768, 128][75497472, 98304, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_5, [1, 768, 768, 128]), kwargs = {})
#   %add_3 : Tensor "bf16[589824, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_8, %mm_9), kwargs = {})
#   %view_36 : Tensor "bf16[1, 768, 768, 128][75497472, 98304, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_3, [1, 768, 768, 128]), kwargs = {})
#   %add_4 : Tensor "bf16[1, 768, 768, 128][75497472, 98304, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_21, %view_36), kwargs = {})
#   %view_37 : Tensor "bf16[1, 589824, 128][75497472, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_4, [1, 589824, 128]), kwargs = {})
#   %layer_norm_transpose_bwd_1 : [num_users=3] = call_function[target=torch.ops.cuequivariance.layer_norm_transpose_bwd.default](args = (%view_37, %view, %primals_2, %getitem_1, %getitem_2, True, 0), kwargs = {})
#   return %buf27
triton_poi_fused_add_layer_norm_transpose_bwd_view_4 = async_compile.triton('triton_poi_fused_add_layer_norm_transpose_bwd_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_layer_norm_transpose_bwd_view_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 754974720}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_layer_norm_transpose_bwd_view_4(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75497472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/inductor-all-L768/kg/ckgeabamdgvqdnjsy7t3otdftxnr5cn6t2s4ymuutbpitdtnmvby.py
# Topologically Sorted Source Nodes: [view_38, sum_3], Original ATen: [aten.view, aten.sum]
# Source node to ATen node mapping:
#   sum_3 => sum_3
#   view_38 => view_38
# Graph fragment:
#   %getitem_15 : Tensor "f32[1, 9216, 128][1179648, 128, 1]cuda:0" = PlaceHolder[target=getitem_15]
#   %view_38 : Tensor "f32[9216, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_15, [-1, 128]), kwargs = {})
#   %sum_3 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_38, [0]), kwargs = {})
#   return %buf32
triton_red_fused_sum_view_5 = async_compile.triton('triton_red_fused_sum_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4792320, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_view_5(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 9216
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 128)
    x1 = xindex // 128
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 16384*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/inductor-all-L768/6x/c6x5tuanyjqfkukbpc67bagwr73rdkkg5josh5cj4wuq3ns3nxyw.py
# Topologically Sorted Source Nodes: [view_38, sum_3], Original ATen: [aten.view, aten.sum]
# Source node to ATen node mapping:
#   sum_3 => sum_3
#   view_38 => view_38
# Graph fragment:
#   %buf32 : Tensor "f32[128, 72][1, 128]cuda:0" = PlaceHolder[target=buf32]
#   %view_38 : Tensor "f32[9216, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_15, [-1, 128]), kwargs = {})
#   %sum_3 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_38, [0]), kwargs = {})
#   return %sum_3
triton_red_fused_sum_view_6 = async_compile.triton('triton_red_fused_sum_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 37888, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_view_6(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 72
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/inductor-all-L768/e7/ce7lsxx5piwqdby62fkzafj4kvk76bunbeu52chi4y54pbxmbuoz.py
# Topologically Sorted Source Nodes: [view_40, add_5], Original ATen: [aten.view, aten.add]
# Source node to ATen node mapping:
#   add_5 => add_5
#   view_40 => view_40
# Graph fragment:
#   %tangents_1 : Tensor "bf16[1, 768, 768, 128][75497472, 98304, 128, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %getitem_14 : Tensor "bf16[1, 589824, 128][75497472, 128, 1]cuda:0" = PlaceHolder[target=getitem_14]
#   %view_40 : Tensor "bf16[1, 768, 768, 128][75497472, 98304, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_14, [1, 768, 768, 128]), kwargs = {})
#   %add_5 : Tensor "bf16[1, 768, 768, 128][75497472, 98304, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_1, %view_40), kwargs = {})
#   return %add_5
triton_poi_fused_add_view_7 = async_compile.triton('triton_poi_fused_add_view_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_view_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 603979776}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_view_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75497472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
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
        primals_2, primals_8, primals_9, primals_13, view, getitem_1, getitem_2, cat, cat_1, view_2, view_12, getitem_6, getitem_7, view_14, mm, view_16, mm_1, permute_14, permute_18, permute_21, permute_22, permute_28, permute_29, tangents_1 = args
        args.clear()
        assert_size_stride(primals_2, (128, ), (1, ))
        assert_size_stride(primals_8, (1, 768, 768), (589824, 768, 1))
        assert_size_stride(primals_9, (256, ), (1, ))
        assert_size_stride(primals_13, (1, 1, 768, 128), (98304, 98304, 128, 1))
        assert_size_stride(view, (1, 589824, 128), (75497472, 128, 1))
        assert_size_stride(getitem_1, (1, 589824), (589824, 1))
        assert_size_stride(getitem_2, (1, 589824), (589824, 1))
        assert_size_stride(cat, (512, 128), (128, 1))
        assert_size_stride(cat_1, (512, 128), (128, 1))
        assert_size_stride(view_2, (589824, 128), (128, 1))
        assert_size_stride(view_12, (256, 1, 589824), (589824, 589824, 1))
        assert_size_stride(getitem_6, (1, 589824), (589824, 1))
        assert_size_stride(getitem_7, (1, 589824), (589824, 1))
        assert_size_stride(view_14, (589824, 128), (128, 1))
        assert_size_stride(mm, (589824, 128), (128, 1))
        assert_size_stride(view_16, (589824, 256), (256, 1))
        assert_size_stride(mm_1, (589824, 128), (128, 1))
        assert_size_stride(permute_14, (128, 256), (256, 1))
        assert_size_stride(permute_18, (128, 128), (128, 1))
        assert_size_stride(permute_21, (128, 768, 768), (589824, 768, 1))
        assert_size_stride(permute_22, (128, 768, 768), (589824, 1, 768))
        assert_size_stride(permute_28, (128, 768, 768), (589824, 1, 768))
        assert_size_stride(permute_29, (128, 768, 768), (589824, 768, 1))
        assert_size_stride(tangents_1, (1, 768, 768, 128), (75497472, 98304, 128, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((1, 768, 768, 128), (75497472, 98304, 128, 1), torch.bfloat16)
            buf3 = reinterpret_tensor(mm_1, (1, 768, 768, 128), (75497472, 98304, 128, 1), 0); del mm_1  # reuse
            # Topologically Sorted Source Nodes: [mul_2, linear, sigmoid, mul_3, linear_1, mul_4, convert_element_type_12, convert_element_type_13, sub, mul_5, mul_6, convert_element_type_14], Original ATen: [aten.mul, aten._unsafe_view, aten.sigmoid, aten.sigmoid_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_sigmoid_sigmoid_backward_0.run(buf3, tangents_1, primals_13, mm, buf0, 75497472, stream=stream0)
            del mm
            del primals_13
            buf1 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_2, linear, sigmoid, mul_3, view_18, permute_12, mm_2], Original ATen: [aten.mul, aten._unsafe_view, aten.sigmoid, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf0, (128, 589824), (1, 128), 0), view_16, out=buf1)
            del view_16
            buf2 = empty_strided_cuda((589824, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_2, linear, sigmoid, mul_3, view_18, mm_3], Original ATen: [aten.mul, aten._unsafe_view, aten.sigmoid, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf0, (589824, 128), (128, 1), 0), permute_14, out=buf2)
            del permute_14
            buf4 = empty_strided_cuda((128, 128), (128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_2, linear, sigmoid, linear_1, mul_4, convert_element_type_12, convert_element_type_13, sub, mul_5, mul_6, convert_element_type_14, view_20, permute_16, mm_4], Original ATen: [aten.mul, aten._unsafe_view, aten.sigmoid, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf3, (128, 589824), (1, 128), 0), view_14, out=buf4)
            del view_14
            buf5 = reinterpret_tensor(buf0, (589824, 128), (128, 1), 0); del buf0  # reuse
            # Topologically Sorted Source Nodes: [mul_2, linear, sigmoid, linear_1, mul_4, convert_element_type_12, convert_element_type_13, sub, mul_5, mul_6, convert_element_type_14, view_20, mm_5], Original ATen: [aten.mul, aten._unsafe_view, aten.sigmoid, aten.sigmoid_backward, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf3, (589824, 128), (128, 1), 0), permute_18, out=buf5)
            del permute_18
            # Topologically Sorted Source Nodes: [view_19, view_22, layer_norm_transpose_bwd], Original ATen: [aten.view, cuequivariance.layer_norm_transpose_bwd]
            buf6 = torch.ops.cuequivariance.layer_norm_transpose_bwd.default(reinterpret_tensor(buf2, (1, 589824, 256), (150994944, 256, 1), 0), view_12, primals_9, getitem_6, getitem_7, True, 3)
            del buf2
            del getitem_6
            del getitem_7
            del primals_9
            del view_12
            buf7 = buf6[0]
            assert_size_stride(buf7, (256, 1, 589824), (589824, 589824, 1), 'torch.ops.cuequivariance.layer_norm_transpose_bwd.default')
            assert_alignment(buf7, 16, 'torch.ops.cuequivariance.layer_norm_transpose_bwd.default')
            buf8 = buf6[1]
            assert_size_stride(buf8, (1, 9216, 256), (2359296, 256, 1), 'torch.ops.cuequivariance.layer_norm_transpose_bwd.default')
            assert_alignment(buf8, 16, 'torch.ops.cuequivariance.layer_norm_transpose_bwd.default')
            buf9 = buf6[2]
            assert_size_stride(buf9, (1, 9216, 256), (2359296, 256, 1), 'torch.ops.cuequivariance.layer_norm_transpose_bwd.default')
            assert_alignment(buf9, 16, 'torch.ops.cuequivariance.layer_norm_transpose_bwd.default')
            del buf6
            buf10 = empty_strided_cuda((256, 72), (1, 256), torch.float32)
            # Topologically Sorted Source Nodes: [view_23, sum_1], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_view_1.run(buf8, buf10, 18432, 128, stream=stream0)
            del buf8
            buf11 = empty_strided_cuda((256, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_23, sum_1], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_view_2.run(buf10, buf11, 256, 72, stream=stream0)
            buf12 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [view_24, sum_2], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_view_1.run(buf9, buf12, 18432, 128, stream=stream0)
            del buf9
            buf13 = empty_strided_cuda((256, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_24, sum_2], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_view_2.run(buf12, buf13, 256, 72, stream=stream0)
            del buf12
            buf14 = reinterpret_tensor(buf3, (128, 768, 768), (589824, 768, 1), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [view_25, slice_6, view_26, permute_20, view_27, bmm_2], Original ATen: [aten.view, aten.slice, aten.permute, aten.bmm]
            extern_kernels.bmm(permute_21, reinterpret_tensor(buf7, (128, 768, 768), (589824, 768, 1), 75497472), out=buf14)
            del permute_21
            buf15 = empty_strided_cuda((128, 768, 768), (589824, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_25, slice_6, view_26, permute_20, view_27, bmm_3], Original ATen: [aten.view, aten.slice, aten.permute, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf7, (128, 768, 768), (589824, 768, 1), 75497472), permute_22, out=buf15)
            del permute_22
            buf16 = empty_strided_cuda((128, 768, 768), (589824, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_25, slice_5, view_30, permute_27, view_31, bmm_4], Original ATen: [aten.view, aten.slice, aten.permute, aten.bmm]
            extern_kernels.bmm(permute_28, reinterpret_tensor(buf7, (128, 768, 768), (589824, 768, 1), 0), out=buf16)
            del permute_28
            buf17 = empty_strided_cuda((128, 768, 768), (589824, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_25, slice_5, view_30, permute_27, view_31, bmm_5], Original ATen: [aten.view, aten.slice, aten.permute, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf7, (128, 768, 768), (589824, 768, 1), 0), permute_29, out=buf17)
            del buf7
            del permute_29
            buf18 = empty_strided_cuda((512, 1, 768, 768), (589824, 589824, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_28, permute_23, view_29, permute_24, permute_25, squeeze, permute_26, squeeze_1, full_default, view_32, permute_30, view_33, permute_31, permute_32, squeeze_2, permute_33, squeeze_3, add_1, add_2, cat_3], Original ATen: [aten.view, aten.permute, aten.squeeze, aten.slice_backward, aten.add, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_permute_slice_backward_squeeze_view_3.run(buf15, buf17, buf14, buf16, buf18, 393216, 768, stream=stream0)
            del buf14
            del buf15
            del buf16
            del buf17
            # Topologically Sorted Source Nodes: [view_34, view_35, fused_gated_dual_gemm_backward], Original ATen: [aten.view, cuequivariance.fused_gated_dual_gemm_backward]
            buf19 = torch.ops.cuequivariance.fused_gated_dual_gemm_backward.default(reinterpret_tensor(buf18, (589824, 512), (512, 1), 0), view_2, None, cat, cat_1, None, None, primals_8, True, 0)
            del buf18
            del primals_8
            buf20 = buf19[0]
            assert_size_stride(buf20, (589824, 512), (512, 1), 'torch.ops.cuequivariance.fused_gated_dual_gemm_backward.default')
            assert_alignment(buf20, 16, 'torch.ops.cuequivariance.fused_gated_dual_gemm_backward.default')
            buf21 = buf19[1]
            assert_size_stride(buf21, (589824, 512), (512, 1), 'torch.ops.cuequivariance.fused_gated_dual_gemm_backward.default')
            assert_alignment(buf21, 16, 'torch.ops.cuequivariance.fused_gated_dual_gemm_backward.default')
            del buf19
            buf23 = empty_strided_cuda((512, 128), (128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_34, mm_6], Original ATen: [aten.permute, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf20, (512, 589824), (1, 512), 0), view_2, out=buf23)
            buf24 = empty_strided_cuda((512, 128), (128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_35, mm_7], Original ATen: [aten.permute, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf21, (512, 589824), (1, 512), 0), view_2, out=buf24)
            del view_2
            buf25 = empty_strided_cuda((589824, 128), (128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mm_8], Original ATen: [aten.mm]
            extern_kernels.mm(buf20, cat, out=buf25)
            del buf20
            del cat
            buf26 = empty_strided_cuda((589824, 128), (128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mm_9], Original ATen: [aten.mm]
            extern_kernels.mm(buf21, cat_1, out=buf26)
            del buf21
            del cat_1
            buf27 = reinterpret_tensor(buf5, (1, 589824, 128), (75497472, 128, 1), 0); del buf5  # reuse
            # Topologically Sorted Source Nodes: [view_21, add_3, view_36, add_4, view_37, layer_norm_transpose_bwd_1], Original ATen: [aten.view, aten.add, cuequivariance.layer_norm_transpose_bwd]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_layer_norm_transpose_bwd_view_4.run(buf27, buf25, buf26, 75497472, stream=stream0)
            del buf25
            del buf26
            # Topologically Sorted Source Nodes: [view_21, add_3, view_36, add_4, view_37, layer_norm_transpose_bwd_1], Original ATen: [aten.view, aten.add, cuequivariance.layer_norm_transpose_bwd]
            buf28 = torch.ops.cuequivariance.layer_norm_transpose_bwd.default(buf27, view, primals_2, getitem_1, getitem_2, True, 0)
            del buf27
            del getitem_1
            del getitem_2
            del primals_2
            del view
            buf29 = buf28[0]
            assert_size_stride(buf29, (1, 589824, 128), (75497472, 128, 1), 'torch.ops.cuequivariance.layer_norm_transpose_bwd.default')
            assert_alignment(buf29, 16, 'torch.ops.cuequivariance.layer_norm_transpose_bwd.default')
            buf30 = buf28[1]
            assert_size_stride(buf30, (1, 9216, 128), (1179648, 128, 1), 'torch.ops.cuequivariance.layer_norm_transpose_bwd.default')
            assert_alignment(buf30, 16, 'torch.ops.cuequivariance.layer_norm_transpose_bwd.default')
            buf31 = buf28[2]
            assert_size_stride(buf31, (1, 9216, 128), (1179648, 128, 1), 'torch.ops.cuequivariance.layer_norm_transpose_bwd.default')
            assert_alignment(buf31, 16, 'torch.ops.cuequivariance.layer_norm_transpose_bwd.default')
            del buf28
            buf32 = empty_strided_cuda((128, 72), (1, 128), torch.float32)
            # Topologically Sorted Source Nodes: [view_38, sum_3], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_view_5.run(buf30, buf32, 9216, 128, stream=stream0)
            del buf30
            buf33 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_38, sum_3], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_view_6.run(buf32, buf33, 128, 72, stream=stream0)
            buf34 = buf32; del buf32  # reuse
            # Topologically Sorted Source Nodes: [view_39, sum_4], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_view_5.run(buf31, buf34, 9216, 128, stream=stream0)
            del buf31
            buf35 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_39, sum_4], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_view_6.run(buf34, buf35, 128, 72, stream=stream0)
            del buf34
            buf36 = reinterpret_tensor(buf29, (1, 768, 768, 128), (75497472, 98304, 128, 1), 0); del buf29  # reuse
            # Topologically Sorted Source Nodes: [view_40, add_5], Original ATen: [aten.view, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_view_7.run(buf36, tangents_1, 75497472, stream=stream0)
            del tangents_1
        return (buf36, buf33, buf35, reinterpret_tensor(buf23, (256, 128), (128, 1), 0), reinterpret_tensor(buf23, (256, 128), (128, 1), 32768), reinterpret_tensor(buf24, (256, 128), (128, 1), 0), reinterpret_tensor(buf24, (256, 128), (128, 1), 32768), None, buf11, buf13, buf4, buf1, None, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((1, 768, 768), (589824, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_9 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((1, 1, 768, 128), (98304, 98304, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    view = rand_strided((1, 589824, 128), (75497472, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_1 = rand_strided((1, 589824), (589824, 1), device='cuda:0', dtype=torch.float32)
    getitem_2 = rand_strided((1, 589824), (589824, 1), device='cuda:0', dtype=torch.float32)
    cat = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    view_2 = rand_strided((589824, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    view_12 = rand_strided((256, 1, 589824), (589824, 589824, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_6 = rand_strided((1, 589824), (589824, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((1, 589824), (589824, 1), device='cuda:0', dtype=torch.float32)
    view_14 = rand_strided((589824, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    mm = rand_strided((589824, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    view_16 = rand_strided((589824, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_1 = rand_strided((589824, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_14 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_18 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_21 = rand_strided((128, 768, 768), (589824, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_22 = rand_strided((128, 768, 768), (589824, 1, 768), device='cuda:0', dtype=torch.bfloat16)
    permute_28 = rand_strided((128, 768, 768), (589824, 1, 768), device='cuda:0', dtype=torch.bfloat16)
    permute_29 = rand_strided((128, 768, 768), (589824, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((1, 768, 768, 128), (75497472, 98304, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_2, primals_8, primals_9, primals_13, view, getitem_1, getitem_2, cat, cat_1, view_2, view_12, getitem_6, getitem_7, view_14, mm, view_16, mm_1, permute_14, permute_18, permute_21, permute_22, permute_28, permute_29, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

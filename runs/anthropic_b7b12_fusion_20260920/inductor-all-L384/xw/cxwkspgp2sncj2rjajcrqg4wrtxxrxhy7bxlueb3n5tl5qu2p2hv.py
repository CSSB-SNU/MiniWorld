# AOT ID: ['1_forward']
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


# kernel path: /home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/inductor-all-L384/x6/cx6ejx7rtkvf7vug53h73ltbxlbihdujb47jsifl3z5guctwyhyv.py
# Topologically Sorted Source Nodes: [t, WLt, t_1, WLgt, t_2, WRt, t_3, WRgt, stack, left_w, stack_1, right_w, cat], Original ATen: [aten.t, aten.clone, aten.stack, aten.view, aten.cat]
# Source node to ATen node mapping:
#   WLgt => clone_1
#   WLt => clone
#   WRgt => clone_3
#   WRt => clone_2
#   cat => cat_2
#   left_w => view_5
#   right_w => view_6
#   stack => cat, unsqueeze, unsqueeze_1
#   stack_1 => cat_1, unsqueeze_2, unsqueeze_3
#   t => permute
#   t_1 => permute_1
#   t_2 => permute_2
#   t_3 => permute_3
# Graph fragment:
#   %primals_7 : Tensor "bf16[256, 128][128, 1]cuda:0" = PlaceHolder[target=primals_7]
#   %primals_6 : Tensor "bf16[256, 128][128, 1]cuda:0" = PlaceHolder[target=primals_6]
#   %primals_9 : Tensor "bf16[256, 128][128, 1]cuda:0" = PlaceHolder[target=primals_9]
#   %primals_8 : Tensor "bf16[256, 128][128, 1]cuda:0" = PlaceHolder[target=primals_8]
#   %permute : Tensor "bf16[128, 256][1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%primals_6, [1, 0]), kwargs = {})
#   %clone : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_1 : Tensor "bf16[128, 256][1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%primals_7, [1, 0]), kwargs = {})
#   %clone_1 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_2 : Tensor "bf16[128, 256][1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%primals_8, [1, 0]), kwargs = {})
#   %clone_2 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_3 : Tensor "bf16[128, 256][1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%primals_9, [1, 0]), kwargs = {})
#   %clone_3 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_3,), kwargs = {memory_format: torch.contiguous_format})
#   %unsqueeze : Tensor "bf16[128, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%clone_1, 2), kwargs = {})
#   %unsqueeze_1 : Tensor "bf16[128, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%clone, 2), kwargs = {})
#   %cat : Tensor "bf16[128, 256, 2][512, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze, %unsqueeze_1], 2), kwargs = {})
#   %view_5 : Tensor "bf16[128, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%cat, [128, 512]), kwargs = {})
#   %unsqueeze_2 : Tensor "bf16[128, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%clone_3, 2), kwargs = {})
#   %unsqueeze_3 : Tensor "bf16[128, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%clone_2, 2), kwargs = {})
#   %cat_1 : Tensor "bf16[128, 256, 2][512, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_2, %unsqueeze_3], 2), kwargs = {})
#   %view_6 : Tensor "bf16[128, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%cat_1, [128, 512]), kwargs = {})
#   %cat_2 : Tensor "bf16[128, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_5, %view_6], 1), kwargs = {})
#   return %cat_2
triton_poi_fused_cat_clone_stack_t_view_0 = async_compile.triton('triton_poi_fused_cat_clone_stack_t_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_clone_stack_t_view_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 131072, 'x': 524288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_clone_stack_t_view_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = x1
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.broadcast_to(((x1) % 2), [YBLOCK, XBLOCK])
    tmp6 = tl.full([1, 1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1, 1], 1, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (y0 + 128*((((x1) // 2) % 256))), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp12 = tmp5 >= tmp8
    tmp13 = tl.full([1, 1], 2, tl.int64)
    tmp14 = tmp5 < tmp13
    tmp15 = tmp12 & tmp4
    tmp16 = tl.load(in_ptr1 + (y0 + 128*((((x1) // 2) % 256))), tmp15 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tl.where(tmp9, tmp11, tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp4, tmp17, tmp18)
    tmp20 = tmp0 >= tmp3
    tmp21 = tl.full([1, 1], 1024, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tl.broadcast_to((((-512) + x1) % 2), [YBLOCK, XBLOCK])
    tmp24 = tl.full([1, 1], 0, tl.int64)
    tmp25 = tmp23 >= tmp24
    tmp26 = tl.full([1, 1], 1, tl.int64)
    tmp27 = tmp23 < tmp26
    tmp28 = tmp27 & tmp20
    tmp29 = tl.load(in_ptr2 + (y0 + 128*(((((-512) + x1) // 2) % 256))), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp30 = tmp23 >= tmp26
    tmp31 = tl.full([1, 1], 2, tl.int64)
    tmp32 = tmp23 < tmp31
    tmp33 = tmp30 & tmp20
    tmp34 = tl.load(in_ptr3 + (y0 + 128*(((((-512) + x1) // 2) % 256))), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp35 = tl.where(tmp27, tmp29, tmp34)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp20, tmp35, tmp36)
    tmp38 = tl.where(tmp4, tmp19, tmp37)
    tl.store(out_ptr0 + (x1 + 1024*y0), tmp38, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/inductor-all-L384/oy/coy56nlywkdpi5bzgkbixpf4agz62hvmdpyzc2tl3exva4mlvwf5.py
# Topologically Sorted Source Nodes: [ds_2d, reshape_1, residual_flat, t_4, Wgt, reshape_3, trimul_parity_f567_sm90_default], Original ATen: [aten.view, aten.t, aten.clone, miniworld_engine.trimul_parity_f567_sm90]
# Source node to ATen node mapping:
#   Wgt => clone_4
#   ds_2d => view
#   reshape_1 => view_2
#   reshape_3 => view_8
#   residual_flat => view_4
#   t_4 => permute_4
#   trimul_parity_f567_sm90_default => trimul_parity_f567_sm90
# Graph fragment:
#   %primals_10 : Tensor "bf16[128, 128][128, 1]cuda:0" = PlaceHolder[target=primals_10]
#   %view : Tensor "bf16[384, 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_3, [384, 128]), kwargs = {})
#   %view_2 : Tensor "bf16[1, 384, 384, 128][18874368, 49152, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem, [1, 384, 384, 128]), kwargs = {})
#   %view_4 : Tensor "bf16[147456, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_1, [147456, 128]), kwargs = {})
#   %permute_4 : Tensor "bf16[128, 128][1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%primals_10, [1, 0]), kwargs = {})
#   %clone_4 : Tensor "bf16[128, 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_8 : Tensor "bf16[147456, 128][128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%view_2, [147456, 128]), kwargs = {})
#   %trimul_parity_f567_sm90 : [num_users=3] = call_function[target=torch.ops.miniworld_engine.trimul_parity_f567_sm90.default](args = (%getitem_5, %view_8, %primals_11, %clone_4, %view_4, %view, 384), kwargs = {})
#   return %buf14,%clone_4
triton_poi_fused_clone_t_trimul_parity_f567_sm90_view_1 = async_compile.triton('triton_poi_fused_clone_t_trimul_parity_f567_sm90_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_t_trimul_parity_f567_sm90_view_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 32768, 'x': 131072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_t_trimul_parity_f567_sm90_view_1(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x1), xmask & ymask).to(tl.float32)
    tl.store(out_ptr0 + (x1 + 128*y0), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x1 + 128*y0), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/inductor-all-L384/ei/ceihpzryiqkjvhlew4mcbauhj52226s5png4xqp4rtvb53xyx4ix.py
# Topologically Sorted Source Nodes: [t, WLt, t_1, WLgt, t_2, WRt, t_3, WRgt, t_11, t_12, t_13, t_14, W_stack], Original ATen: [aten.t, aten.clone, aten.cat]
# Source node to ATen node mapping:
#   WLgt => clone_1
#   WLt => clone
#   WRgt => clone_3
#   WRt => clone_2
#   W_stack => cat_3
#   t => permute
#   t_1 => permute_1
#   t_11 => permute_17
#   t_12 => permute_18
#   t_13 => permute_19
#   t_14 => permute_20
#   t_2 => permute_2
#   t_3 => permute_3
# Graph fragment:
#   %primals_7 : Tensor "bf16[256, 128][128, 1]cuda:0" = PlaceHolder[target=primals_7]
#   %primals_6 : Tensor "bf16[256, 128][128, 1]cuda:0" = PlaceHolder[target=primals_6]
#   %primals_9 : Tensor "bf16[256, 128][128, 1]cuda:0" = PlaceHolder[target=primals_9]
#   %primals_8 : Tensor "bf16[256, 128][128, 1]cuda:0" = PlaceHolder[target=primals_8]
#   %permute : Tensor "bf16[128, 256][1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%primals_6, [1, 0]), kwargs = {})
#   %clone : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_1 : Tensor "bf16[128, 256][1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%primals_7, [1, 0]), kwargs = {})
#   %clone_1 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_2 : Tensor "bf16[128, 256][1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%primals_8, [1, 0]), kwargs = {})
#   %clone_2 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_3 : Tensor "bf16[128, 256][1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%primals_9, [1, 0]), kwargs = {})
#   %clone_3 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_3,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_17 : Tensor "bf16[256, 128][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%clone_1, [1, 0]), kwargs = {})
#   %permute_18 : Tensor "bf16[256, 128][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%clone, [1, 0]), kwargs = {})
#   %permute_19 : Tensor "bf16[256, 128][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%clone_3, [1, 0]), kwargs = {})
#   %permute_20 : Tensor "bf16[256, 128][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%clone_2, [1, 0]), kwargs = {})
#   %cat_3 : Tensor "bf16[1024, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%permute_17, %permute_18, %permute_19, %permute_20],), kwargs = {})
#   return %cat_3
triton_poi_fused_cat_clone_t_2 = async_compile.triton('triton_poi_fused_cat_clone_t_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_clone_t_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 786432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_clone_t_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
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
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 128*((-256) + x1)), tmp9, other=0.0).to(tl.float32)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 768, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 128*((-512) + x1)), tmp14, other=0.0).to(tl.float32)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 1024, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr3 + (x0 + 128*((-768) + x1)), tmp16, other=0.0).to(tl.float32)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tl.store(out_ptr0 + (x2), tmp22, None)
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
        assert_size_stride(primals_2, (1, 384, 384), (147456, 384, 1))
        assert_size_stride(primals_3, (1, 1, 384, 128), (49152, 49152, 128, 1))
        assert_size_stride(primals_4, (128, ), (1, ))
        assert_size_stride(primals_5, (128, ), (1, ))
        assert_size_stride(primals_6, (256, 128), (128, 1))
        assert_size_stride(primals_7, (256, 128), (128, 1))
        assert_size_stride(primals_8, (256, 128), (128, 1))
        assert_size_stride(primals_9, (256, 128), (128, 1))
        assert_size_stride(primals_10, (128, 128), (128, 1))
        assert_size_stride(primals_11, (128, 256), (256, 1))
        assert_size_stride(primals_12, (256, ), (1, ))
        assert_size_stride(primals_13, (256, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            # Topologically Sorted Source Nodes: [reshape, layernorm_fwd_default], Original ATen: [aten.view, miniworld_engine.layernorm_fwd]
            buf0 = torch.ops.miniworld_engine.layernorm_fwd.default(reinterpret_tensor(primals_1, (147456, 128), (128, 1), 0), primals_4, primals_5, None, 1e-05, False, 147456)
            del primals_5
            buf1 = buf0[0]
            assert_size_stride(buf1, (147456, 128), (128, 1), 'torch.ops.miniworld_engine.layernorm_fwd.default')
            assert_alignment(buf1, 16, 'torch.ops.miniworld_engine.layernorm_fwd.default')
            buf2 = buf0[1]
            assert_size_stride(buf2, (147456, ), (1, ), 'torch.ops.miniworld_engine.layernorm_fwd.default')
            assert_alignment(buf2, 16, 'torch.ops.miniworld_engine.layernorm_fwd.default')
            buf3 = buf0[2]
            assert_size_stride(buf3, (147456, ), (1, ), 'torch.ops.miniworld_engine.layernorm_fwd.default')
            assert_alignment(buf3, 16, 'torch.ops.miniworld_engine.layernorm_fwd.default')
            del buf0
            buf4 = empty_strided_cuda((128, 1024), (1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [t, WLt, t_1, WLgt, t_2, WRt, t_3, WRgt, stack, left_w, stack_1, right_w, cat], Original ATen: [aten.t, aten.clone, aten.stack, aten.view, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_clone_stack_t_view_0.run(primals_7, primals_6, primals_9, primals_8, buf4, 128, 1024, stream=stream0)
            # Topologically Sorted Source Nodes: [reshape_1, reshape_2, reshape_3, trimul_front_parity_sm90_default], Original ATen: [aten.view, miniworld_engine.trimul_front_parity_sm90]
            buf5 = torch.ops.miniworld_engine.trimul_front_parity_sm90.default(buf1, buf4, reinterpret_tensor(primals_2, (147456, ), (1, ), 0), True, 0, 0, 0, 0, 0)
            del buf4
            buf6 = buf5[0]
            assert_size_stride(buf6, (512, 147456), (147456, 1), 'torch.ops.miniworld_engine.trimul_front_parity_sm90.default')
            assert_alignment(buf6, 16, 'torch.ops.miniworld_engine.trimul_front_parity_sm90.default')
            buf7 = buf5[1]
            assert_size_stride(buf7, (1024, 147456), (147456, 1), 'torch.ops.miniworld_engine.trimul_front_parity_sm90.default')
            assert_alignment(buf7, 16, 'torch.ops.miniworld_engine.trimul_front_parity_sm90.default')
            del buf5
            # Topologically Sorted Source Nodes: [getitem_2, left, getitem_3, right, lf, rf, tri], Original ATen: [aten.slice, aten.view, miniworld_engine.trimul_triton_contract_fwd]
            buf8 = torch.ops.miniworld_engine.trimul_triton_contract_fwd.default(reinterpret_tensor(buf6, (256, 384, 384), (147456, 384, 1), 0), reinterpret_tensor(buf6, (256, 384, 384), (147456, 384, 1), 37748736), 128)
            buf9 = buf8
            assert_size_stride(buf9, (256, 384, 384), (147456, 384, 1), 'torch.ops.miniworld_engine.trimul_triton_contract_fwd.default')
            assert_alignment(buf9, 16, 'torch.ops.miniworld_engine.trimul_triton_contract_fwd.default')
            del buf8
            # Topologically Sorted Source Nodes: [reshape_6, view_2, layernorm_linear_materialize_default], Original ATen: [aten.view, aten.t, miniworld_engine.layernorm_linear_materialize]
            buf10 = torch.ops.miniworld_engine.layernorm_linear_materialize.default(reinterpret_tensor(buf9, (147456, 256), (1, 147456), 0), primals_12, primals_13, 1e-05, 147456)
            del primals_13
            buf11 = buf10[0]
            assert_size_stride(buf11, (147456, 256), (256, 1), 'torch.ops.miniworld_engine.layernorm_linear_materialize.default')
            assert_alignment(buf11, 16, 'torch.ops.miniworld_engine.layernorm_linear_materialize.default')
            buf12 = buf10[1]
            assert_size_stride(buf12, (147456, ), (1, ), 'torch.ops.miniworld_engine.layernorm_linear_materialize.default')
            assert_alignment(buf12, 16, 'torch.ops.miniworld_engine.layernorm_linear_materialize.default')
            buf13 = buf10[2]
            assert_size_stride(buf13, (147456, ), (1, ), 'torch.ops.miniworld_engine.layernorm_linear_materialize.default')
            assert_alignment(buf13, 16, 'torch.ops.miniworld_engine.layernorm_linear_materialize.default')
            del buf10
            buf14 = empty_strided_cuda((128, 128), (128, 1), torch.bfloat16)
            buf20 = empty_strided_cuda((128, 128), (128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [ds_2d, reshape_1, residual_flat, t_4, Wgt, reshape_3, trimul_parity_f567_sm90_default], Original ATen: [aten.view, aten.t, aten.clone, miniworld_engine.trimul_parity_f567_sm90]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_t_trimul_parity_f567_sm90_view_1.run(primals_10, buf14, buf20, 128, 128, stream=stream0)
            del primals_10
            # Topologically Sorted Source Nodes: [ds_2d, reshape_1, residual_flat, t_4, Wgt, reshape_3, trimul_parity_f567_sm90_default], Original ATen: [aten.view, aten.t, aten.clone, miniworld_engine.trimul_parity_f567_sm90]
            buf15 = torch.ops.miniworld_engine.trimul_parity_f567_sm90.default(buf11, buf1, primals_11, buf14, reinterpret_tensor(primals_1, (147456, 128), (128, 1), 0), reinterpret_tensor(primals_3, (384, 128), (128, 1), 0), 384)
            del buf14
            buf16 = buf15[0]
            assert_size_stride(buf16, (147456, 128), (128, 1), 'torch.ops.miniworld_engine.trimul_parity_f567_sm90.default')
            assert_alignment(buf16, 16, 'torch.ops.miniworld_engine.trimul_parity_f567_sm90.default')
            buf17 = buf15[1]
            assert_size_stride(buf17, (147456, 128), (128, 1), 'torch.ops.miniworld_engine.trimul_parity_f567_sm90.default')
            assert_alignment(buf17, 16, 'torch.ops.miniworld_engine.trimul_parity_f567_sm90.default')
            buf18 = buf15[2]
            assert_size_stride(buf18, (147456, 128), (128, 1), 'torch.ops.miniworld_engine.trimul_parity_f567_sm90.default')
            assert_alignment(buf18, 16, 'torch.ops.miniworld_engine.trimul_parity_f567_sm90.default')
            del buf15
            buf19 = empty_strided_cuda((1024, 128), (128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [t, WLt, t_1, WLgt, t_2, WRt, t_3, WRgt, t_11, t_12, t_13, t_14, W_stack], Original ATen: [aten.t, aten.clone, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_clone_t_2.run(primals_7, primals_6, primals_9, primals_8, buf19, 131072, stream=stream0)
            del primals_6
            del primals_7
            del primals_8
            del primals_9
        return (reinterpret_tensor(buf16, (1, 384, 384, 128), (18874368, 49152, 128, 1), 0), primals_4, primals_12, reinterpret_tensor(primals_3, (384, 128), (128, 1), 0), reinterpret_tensor(primals_1, (147456, 128), (128, 1), 0), buf2, buf3, reinterpret_tensor(primals_2, (147456, ), (1, ), 0), buf1, buf7, reinterpret_tensor(buf6, (256, 384, 384), (147456, 384, 1), 0), reinterpret_tensor(buf6, (256, 384, 384), (147456, 384, 1), 37748736), reinterpret_tensor(buf9, (147456, 256), (1, 147456), 0), buf11, buf12, buf13, buf17, buf18, reinterpret_tensor(primals_11, (256, 128), (1, 256), 0), buf19, reinterpret_tensor(buf20, (128, 128), (1, 128), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 384, 384, 128), (18874368, 49152, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = rand_strided((1, 384, 384), (147456, 384, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_3 = rand_strided((1, 1, 384, 128), (49152, 49152, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_7 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_8 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_9 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_10 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_11 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_12 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

# AOT ID: ['1_backward']
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


# kernel path: /home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/inductor-all-L384/yo/cyoxso2ezneznlx2uxgmoxiyessepsw7df3t4jp4ngelzy43vsbn.py
# Topologically Sorted Source Nodes: [getitem_9, t_9, dWRg], Original ATen: [aten.slice, aten.t, aten.clone]
# Source node to ATen node mapping:
#   dWRg => clone_7
#   getitem_9 => slice_5
#   t_9 => permute_15
# Graph fragment:
#   %mm_3 : Tensor "bf16[1024, 128][128, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %slice_5 : Tensor "bf16[256, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_3, 0, 512, 768), kwargs = {})
#   %permute_15 : Tensor "bf16[128, 256][1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%slice_5, [1, 0]), kwargs = {})
#   %clone_7 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_15,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_7
triton_poi_fused_clone_slice_t_0 = async_compile.triton('triton_poi_fused_clone_slice_t_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_slice_t_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 65536, 'x': 131072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_slice_t_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (65536 + y0 + 128*x1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x1 + 256*y0), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/inductor-all-L384/k4/ck4kycejahr3ffte3wyh6giknsp6wimk4yhdkqmhtihyaukobevo.py
# Topologically Sorted Source Nodes: [getitem_10, t_10, dWR], Original ATen: [aten.slice, aten.t, aten.clone]
# Source node to ATen node mapping:
#   dWR => clone_8
#   getitem_10 => slice_6
#   t_10 => permute_16
# Graph fragment:
#   %mm_3 : Tensor "bf16[1024, 128][128, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %slice_6 : Tensor "bf16[256, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_3, 0, 768, 9223372036854775807), kwargs = {})
#   %permute_16 : Tensor "bf16[128, 256][1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%slice_6, [1, 0]), kwargs = {})
#   %clone_8 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_16,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_8
triton_poi_fused_clone_slice_t_1 = async_compile.triton('triton_poi_fused_clone_slice_t_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_slice_t_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 65536, 'x': 131072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_slice_t_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (98304 + y0 + 128*x1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x1 + 256*y0), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/inductor-all-L384/eh/cehbatck7wpu65t7qftjdoxpqcfugq3ay43zca3hvkovo7abuhy5.py
# Topologically Sorted Source Nodes: [getitem_7, t_7, dWLg], Original ATen: [aten.slice, aten.t, aten.clone]
# Source node to ATen node mapping:
#   dWLg => clone_5
#   getitem_7 => slice_3
#   t_7 => permute_13
# Graph fragment:
#   %mm_3 : Tensor "bf16[1024, 128][128, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %slice_3 : Tensor "bf16[256, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_3, 0, 0, 256), kwargs = {})
#   %permute_13 : Tensor "bf16[128, 256][1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%slice_3, [1, 0]), kwargs = {})
#   %clone_5 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_13,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_5
triton_poi_fused_clone_slice_t_2 = async_compile.triton('triton_poi_fused_clone_slice_t_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_slice_t_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 65536, 'x': 131072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_slice_t_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x1 + 256*y0), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/inductor-all-L384/sq/csqi5yvn7kjtibaid35o37y56wwsn6566ta62opoax6jk54asizz.py
# Topologically Sorted Source Nodes: [getitem_8, t_8, dWL], Original ATen: [aten.slice, aten.t, aten.clone]
# Source node to ATen node mapping:
#   dWL => clone_6
#   getitem_8 => slice_4
#   t_8 => permute_14
# Graph fragment:
#   %mm_3 : Tensor "bf16[1024, 128][128, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %slice_4 : Tensor "bf16[256, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_3, 0, 256, 512), kwargs = {})
#   %permute_14 : Tensor "bf16[128, 256][1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%slice_4, [1, 0]), kwargs = {})
#   %clone_6 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_14,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_6
triton_poi_fused_clone_slice_t_3 = async_compile.triton('triton_poi_fused_clone_slice_t_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_slice_t_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 65536, 'x': 131072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_slice_t_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (32768 + y0 + 128*x1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x1 + 256*y0), tmp0, xmask & ymask)
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
        primals_4, primals_12, view, view_1, getitem_1, getitem_2, view_7, view_8, getitem_4, view_11, view_12, permute_5, getitem_5, getitem_6, getitem_7, getitem_9, getitem_10, permute_8, cat_3, permute_22, tangents_1 = args
        args.clear()
        assert_size_stride(primals_4, (128, ), (1, ))
        assert_size_stride(primals_12, (256, ), (1, ))
        assert_size_stride(view, (384, 128), (128, 1))
        assert_size_stride(view_1, (147456, 128), (128, 1))
        assert_size_stride(getitem_1, (147456, ), (1, ))
        assert_size_stride(getitem_2, (147456, ), (1, ))
        assert_size_stride(view_7, (147456, ), (1, ))
        assert_size_stride(view_8, (147456, 128), (128, 1))
        assert_size_stride(getitem_4, (1024, 147456), (147456, 1))
        assert_size_stride(view_11, (256, 384, 384), (147456, 384, 1))
        assert_size_stride(view_12, (256, 384, 384), (147456, 384, 1))
        assert_size_stride(permute_5, (147456, 256), (1, 147456))
        assert_size_stride(getitem_5, (147456, 256), (256, 1))
        assert_size_stride(getitem_6, (147456, ), (1, ))
        assert_size_stride(getitem_7, (147456, ), (1, ))
        assert_size_stride(getitem_9, (147456, 128), (128, 1))
        assert_size_stride(getitem_10, (147456, 128), (128, 1))
        assert_size_stride(permute_8, (256, 128), (1, 256))
        assert_size_stride(cat_3, (1024, 128), (128, 1))
        assert_size_stride(permute_22, (128, 128), (1, 128))
        assert_size_stride(tangents_1, (1, 384, 384, 128), (18874368, 49152, 128, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            # Topologically Sorted Source Nodes: [reshape, trimul_gate_elem_bwd_ew_default], Original ATen: [aten.view, miniworld_engine.trimul_gate_elem_bwd_ew]
            buf0 = torch.ops.miniworld_engine.trimul_gate_elem_bwd_ew.default(reinterpret_tensor(tangents_1, (147456, 128), (128, 1), 0), getitem_9, getitem_10, view, 384, from_preact=False)
            del getitem_10
            del getitem_9
            del view
            buf1 = buf0[0]
            assert_size_stride(buf1, (147456, 128), (128, 1), 'torch.ops.miniworld_engine.trimul_gate_elem_bwd_ew.default')
            assert_alignment(buf1, 16, 'torch.ops.miniworld_engine.trimul_gate_elem_bwd_ew.default')
            buf2 = buf0[1]
            assert_size_stride(buf2, (147456, 128), (128, 1), 'torch.ops.miniworld_engine.trimul_gate_elem_bwd_ew.default')
            assert_alignment(buf2, 16, 'torch.ops.miniworld_engine.trimul_gate_elem_bwd_ew.default')
            del buf0
            buf3 = empty_strided_cuda((128, 128), (128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [t, dWg], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(view_8, (128, 147456), (1, 128), 0), buf2, out=buf3)
            buf4 = empty_strided_cuda((256, 147456), (147456, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [t_3, matmul], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(permute_8, reinterpret_tensor(buf1, (128, 147456), (1, 128), 0), out=buf4)
            del permute_8
            buf5 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [t_3, dW], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1, (128, 147456), (1, 128), 0), getitem_5, out=buf5)
            del buf1
            del getitem_5
            # Topologically Sorted Source Nodes: [dx_normed, layernorm_linear_bwd_mmajor_default], Original ATen: [aten.t, miniworld_engine.layernorm_linear_bwd_mmajor]
            buf6 = torch.ops.miniworld_engine.layernorm_linear_bwd_mmajor.default(reinterpret_tensor(buf4, (147456, 256), (1, 147456), 0), permute_5, primals_12, getitem_6, getitem_7, [1, 147456], 147456)
            del buf4
            del getitem_6
            del getitem_7
            del permute_5
            del primals_12
            buf7 = buf6[0]
            assert_size_stride(buf7, (147456, 256), (1, 147456), 'torch.ops.miniworld_engine.layernorm_linear_bwd_mmajor.default')
            assert_alignment(buf7, 16, 'torch.ops.miniworld_engine.layernorm_linear_bwd_mmajor.default')
            buf8 = buf6[1]
            assert_size_stride(buf8, (256, ), (1, ), 'torch.ops.miniworld_engine.layernorm_linear_bwd_mmajor.default')
            assert_alignment(buf8, 16, 'torch.ops.miniworld_engine.layernorm_linear_bwd_mmajor.default')
            buf9 = buf6[2]
            assert_size_stride(buf9, (256, ), (1, ), 'torch.ops.miniworld_engine.layernorm_linear_bwd_mmajor.default')
            assert_alignment(buf9, 16, 'torch.ops.miniworld_engine.layernorm_linear_bwd_mmajor.default')
            del buf6
            # Topologically Sorted Source Nodes: [t_6, d_tri, trimul_triton_contract_bwd_default], Original ATen: [aten.t, aten.view, miniworld_engine.trimul_triton_contract_bwd]
            buf10 = torch.ops.miniworld_engine.trimul_triton_contract_bwd.default(reinterpret_tensor(buf7, (256, 384, 384), (147456, 384, 1), 0), view_11, view_12, 128)
            del buf7
            del view_11
            del view_12
            buf11 = buf10[0]
            assert_size_stride(buf11, (256, 384, 384), (147456, 384, 1), 'torch.ops.miniworld_engine.trimul_triton_contract_bwd.default')
            assert_alignment(buf11, 16, 'torch.ops.miniworld_engine.trimul_triton_contract_bwd.default')
            buf12 = buf10[1]
            assert_size_stride(buf12, (256, 384, 384), (147456, 384, 1), 'torch.ops.miniworld_engine.trimul_triton_contract_bwd.default')
            assert_alignment(buf12, 16, 'torch.ops.miniworld_engine.trimul_triton_contract_bwd.default')
            del buf10
            # Topologically Sorted Source Nodes: [d_left_1, d_right_1, dL2, dR2, dconc], Original ATen: [aten.view, miniworld_engine.trimul_front_bwd_dconcat]
            buf13 = torch.ops.miniworld_engine.trimul_front_bwd_dconcat.default(reinterpret_tensor(buf11, (37748736, ), (1, ), 0), reinterpret_tensor(buf12, (37748736, ), (1, ), 0), getitem_4, 147456, 256, 384, view_7)
            del buf11
            del buf12
            del getitem_4
            del view_7
            buf14 = buf13
            assert_size_stride(buf14, (1024, 147456), (147456, 1), 'torch.ops.miniworld_engine.trimul_front_bwd_dconcat.default')
            assert_alignment(buf14, 16, 'torch.ops.miniworld_engine.trimul_front_bwd_dconcat.default')
            del buf13
            buf15 = empty_strided_cuda((1024, 128), (128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [dWs], Original ATen: [aten.mm]
            extern_kernels.mm(buf14, view_8, out=buf15)
            del view_8
            # Topologically Sorted Source Nodes: [t_15, dx_1], Original ATen: [aten.t, miniworld_engine.trimul_input_dual_bwd_sm90]
            buf16 = torch.ops.miniworld_engine.trimul_input_dual_bwd_sm90.default(buf2, reinterpret_tensor(buf14, (147456, 1024), (1, 147456), 0), permute_22, cat_3, 384)
            del buf14
            del buf2
            del cat_3
            del permute_22
            buf17 = buf16
            assert_size_stride(buf17, (147456, 128), (128, 1), 'torch.ops.miniworld_engine.trimul_input_dual_bwd_sm90.default')
            assert_alignment(buf17, 16, 'torch.ops.miniworld_engine.trimul_input_dual_bwd_sm90.default')
            del buf16
            buf18 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_9, t_9, dWRg], Original ATen: [aten.slice, aten.t, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_slice_t_0.run(buf15, buf18, 128, 256, stream=stream0)
            buf19 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_10, t_10, dWR], Original ATen: [aten.slice, aten.t, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_slice_t_1.run(buf15, buf19, 128, 256, stream=stream0)
            buf20 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_7, t_7, dWLg], Original ATen: [aten.slice, aten.t, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_slice_t_2.run(buf15, buf20, 128, 256, stream=stream0)
            buf21 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_8, t_8, dWL], Original ATen: [aten.slice, aten.t, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_slice_t_3.run(buf15, buf21, 128, 256, stream=stream0)
            del buf15
            # Topologically Sorted Source Nodes: [reshape, trimul_input_ln_residual_bwd_default], Original ATen: [aten.view, miniworld_engine.trimul_input_ln_residual_bwd]
            buf22 = torch.ops.miniworld_engine.trimul_input_ln_residual_bwd.default(buf17, view_1, primals_4, getitem_1, getitem_2, reinterpret_tensor(tangents_1, (147456, 128), (128, 1), 0), 147456)
            del buf17
            del getitem_1
            del getitem_2
            del primals_4
            del tangents_1
            del view_1
            buf23 = buf22[0]
            assert_size_stride(buf23, (147456, 128), (128, 1), 'torch.ops.miniworld_engine.trimul_input_ln_residual_bwd.default')
            assert_alignment(buf23, 16, 'torch.ops.miniworld_engine.trimul_input_ln_residual_bwd.default')
            buf24 = buf22[1]
            assert_size_stride(buf24, (128, ), (1, ), 'torch.ops.miniworld_engine.trimul_input_ln_residual_bwd.default')
            assert_alignment(buf24, 16, 'torch.ops.miniworld_engine.trimul_input_ln_residual_bwd.default')
            buf25 = buf22[2]
            assert_size_stride(buf25, (128, ), (1, ), 'torch.ops.miniworld_engine.trimul_input_ln_residual_bwd.default')
            assert_alignment(buf25, 16, 'torch.ops.miniworld_engine.trimul_input_ln_residual_bwd.default')
            del buf22
        return (reinterpret_tensor(buf23, (1, 384, 384, 128), (18874368, 49152, 128, 1), 0), None, None, buf24, buf25, reinterpret_tensor(buf21, (256, 128), (1, 256), 0), reinterpret_tensor(buf20, (256, 128), (1, 256), 0), reinterpret_tensor(buf19, (256, 128), (1, 256), 0), reinterpret_tensor(buf18, (256, 128), (1, 256), 0), reinterpret_tensor(buf3, (128, 128), (1, 128), 0), buf5, buf8, buf9, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    view_1 = rand_strided((147456, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_1 = rand_strided((147456, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_2 = rand_strided((147456, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_7 = rand_strided((147456, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    view_8 = rand_strided((147456, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_4 = rand_strided((1024, 147456), (147456, 1), device='cuda:0', dtype=torch.bfloat16)
    view_11 = rand_strided((256, 384, 384), (147456, 384, 1), device='cuda:0', dtype=torch.bfloat16)
    view_12 = rand_strided((256, 384, 384), (147456, 384, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_5 = rand_strided((147456, 256), (1, 147456), device='cuda:0', dtype=torch.bfloat16)
    getitem_5 = rand_strided((147456, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_6 = rand_strided((147456, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((147456, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_9 = rand_strided((147456, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_10 = rand_strided((147456, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_8 = rand_strided((256, 128), (1, 256), device='cuda:0', dtype=torch.bfloat16)
    cat_3 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_22 = rand_strided((128, 128), (1, 128), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((1, 384, 384, 128), (18874368, 49152, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_4, primals_12, view, view_1, getitem_1, getitem_2, view_7, view_8, getitem_4, view_11, view_12, permute_5, getitem_5, getitem_6, getitem_7, getitem_9, getitem_10, permute_8, cat_3, permute_22, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

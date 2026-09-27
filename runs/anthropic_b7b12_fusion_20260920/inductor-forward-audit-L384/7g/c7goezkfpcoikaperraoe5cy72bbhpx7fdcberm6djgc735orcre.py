# AOT ID: ['0_inference']
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


# kernel path: /home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/inductor-forward-audit-L384/j2/cj2n4grdbhe6udchkpww43hvhrk2zyjdaxtdu5b6mhhjjs3uv6w4.py
# Topologically Sorted Source Nodes: [t, contiguous], Original ATen: [aten.t, aten.clone]
# Source node to ATen node mapping:
#   contiguous => clone
#   t => permute
# Graph fragment:
#   %arg0_1 : Tensor "bf16[256, 128][128, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %permute : Tensor "bf16[128, 256][1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg0_1, [1, 0]), kwargs = {})
#   %clone : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone
triton_poi_fused_clone_t_0 = async_compile.triton('triton_poi_fused_clone_t_0', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_t_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 65536, 'x': 131072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_t_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/inductor-forward-audit-L384/je/cjenum65lnn356rpub5knatlsgss5ftmwrjvotwv2caf4dryzxvs.py
# Topologically Sorted Source Nodes: [t_4, contiguous_4], Original ATen: [aten.t, aten.clone]
# Source node to ATen node mapping:
#   contiguous_4 => clone_4
#   t_4 => permute_4
# Graph fragment:
#   %arg4_1 : Tensor "bf16[128, 128][128, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %permute_4 : Tensor "bf16[128, 128][1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg4_1, [1, 0]), kwargs = {})
#   %clone_4 : Tensor "bf16[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_4,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_4
triton_poi_fused_clone_t_1 = async_compile.triton('triton_poi_fused_clone_t_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 128}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_t_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 32768, 'x': 65536}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_t_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: /home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/inductor-forward-audit-L384/kl/cklyonnl3sppq5a6assw5cohmkwguygjr3psm6gove3c3ddm2ac4.py
# Topologically Sorted Source Nodes: [gate, reshape, proj, reshape_1, stack], Original ATen: [aten.cat, aten.view, aten.stack]
# Source node to ATen node mapping:
#   gate => cat
#   proj => cat_1
#   reshape => view
#   reshape_1 => view_1
#   stack => cat_2
# Graph fragment:
#   %arg1_1 : Tensor "bf16[256, 128][128, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg3_1 : Tensor "bf16[256, 128][128, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg0_1 : Tensor "bf16[256, 128][128, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg2_1 : Tensor "bf16[256, 128][128, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %cat : Tensor "bf16[512, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%arg1_1, %arg3_1],), kwargs = {})
#   %view : Tensor "bf16[16, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%cat, [-1, 32, 128]), kwargs = {})
#   %cat_1 : Tensor "bf16[512, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%arg0_1, %arg2_1],), kwargs = {})
#   %view_1 : Tensor "bf16[16, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%cat_1, [-1, 32, 128]), kwargs = {})
#   %cat_2 : Tensor "bf16[16, 64, 128][8192, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view, %view_1], 1), kwargs = {})
#   return %cat_2
triton_poi_fused_cat_stack_view_2 = async_compile.triton('triton_poi_fused_cat_stack_view_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_stack_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 786432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_stack_view_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = ((xindex // 128) % 64)
    x2 = xindex // 8192
    x0 = (xindex % 128)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 32*x2 + (x1)
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 256, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (x0 + 128*(32*x2 + (x1))), tmp10, other=0.0).to(tl.float32)
    tmp12 = tmp5 >= tmp8
    tmp13 = tl.full([1], 512, tl.int64)
    tmp14 = tmp5 < tmp13
    tmp15 = tmp12 & tmp4
    tmp16 = tl.load(in_ptr1 + (x0 + 128*((-256) + 32*x2 + (x1))), tmp15, other=0.0).to(tl.float32)
    tmp17 = tl.where(tmp9, tmp11, tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp4, tmp17, tmp18)
    tmp20 = tmp0 >= tmp3
    tmp21 = tl.full([1], 64, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = 32*x2 + ((-32) + x1)
    tmp24 = tl.full([1], 0, tl.int64)
    tmp25 = tmp23 >= tmp24
    tmp26 = tl.full([1], 256, tl.int64)
    tmp27 = tmp23 < tmp26
    tmp28 = tmp27 & tmp20
    tmp29 = tl.load(in_ptr2 + (x0 + 128*(32*x2 + ((-32) + x1))), tmp28, other=0.0).to(tl.float32)
    tmp30 = tmp23 >= tmp26
    tmp31 = tl.full([1], 512, tl.int64)
    tmp32 = tmp23 < tmp31
    tmp33 = tmp30 & tmp20
    tmp34 = tl.load(in_ptr3 + (x0 + 128*((-256) + 32*x2 + ((-32) + x1))), tmp33, other=0.0).to(tl.float32)
    tmp35 = tl.where(tmp27, tmp29, tmp34)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp20, tmp35, tmp36)
    tmp38 = tl.where(tmp4, tmp19, tmp37)
    tl.store(out_ptr0 + (x3), tmp38, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
        args.clear()
        assert_size_stride(arg0_1, (256, 128), (128, 1))
        assert_size_stride(arg1_1, (256, 128), (128, 1))
        assert_size_stride(arg2_1, (256, 128), (128, 1))
        assert_size_stride(arg3_1, (256, 128), (128, 1))
        assert_size_stride(arg4_1, (128, 128), (128, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [t, contiguous], Original ATen: [aten.t, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_t_0.run(arg0_1, buf0, 128, 256, stream=stream0)
            buf1 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [t_1, contiguous_1], Original ATen: [aten.t, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_t_0.run(arg1_1, buf1, 128, 256, stream=stream0)
            buf2 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [t_2, contiguous_2], Original ATen: [aten.t, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_t_0.run(arg2_1, buf2, 128, 256, stream=stream0)
            buf3 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [t_3, contiguous_3], Original ATen: [aten.t, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_t_0.run(arg3_1, buf3, 128, 256, stream=stream0)
            buf4 = empty_strided_cuda((128, 128), (128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [t_4, contiguous_4], Original ATen: [aten.t, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_t_1.run(arg4_1, buf4, 128, 128, stream=stream0)
            del arg4_1
            buf5 = empty_strided_cuda((16, 64, 128), (8192, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [gate, reshape, proj, reshape_1, stack], Original ATen: [aten.cat, aten.view, aten.stack]
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_stack_view_2.run(arg1_1, arg3_1, arg0_1, arg2_1, buf5, 131072, stream=stream0)
            del arg0_1
            del arg1_1
            del arg2_1
            del arg3_1
        return (buf0, buf1, buf2, buf3, buf4, reinterpret_tensor(buf5, (1024, 128), (128, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

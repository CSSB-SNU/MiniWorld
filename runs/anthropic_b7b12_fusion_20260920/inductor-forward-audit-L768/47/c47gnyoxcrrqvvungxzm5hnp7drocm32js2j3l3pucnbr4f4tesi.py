
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_clone_stack_t_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 65536, 'x': 1376256}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_clone_stack_t_view_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1088
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = x1 + 256*y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 32768, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128*(((x1 + 256*y0) % 256)) + ((((x1 + 256*y0) // 256) % 128))), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 65536, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (128*((((-32768) + x1 + 256*y0) % 256)) + (((((-32768) + x1 + 256*y0) // 256) % 128))), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1, 1], 98304, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (128*((((-65536) + x1 + 256*y0) % 256)) + (((((-65536) + x1 + 256*y0) // 256) % 128))), tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1, 1], 131072, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (128*((((-98304) + x1 + 256*y0) % 256)) + (((((-98304) + x1 + 256*y0) // 256) % 128))), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1, 1], 147456, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (128*((((-131072) + x1 + 256*y0) % 128)) + (((((-131072) + x1 + 256*y0) // 128) % 128))), tmp24 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1, 1], 278528, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tl.load(in_ptr5 + ((-147456) + x1 + 256*y0), tmp26 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp30 = tl.where(tmp24, tmp25, tmp29)
    tmp31 = tl.where(tmp19, tmp20, tmp30)
    tmp32 = tl.where(tmp14, tmp15, tmp31)
    tmp33 = tl.where(tmp9, tmp10, tmp32)
    tmp34 = tl.where(tmp4, tmp5, tmp33)
    tl.store(out_ptr0 + (x1 + 256*y0), tmp34, xmask & ymask)

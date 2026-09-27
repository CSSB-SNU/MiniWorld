
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

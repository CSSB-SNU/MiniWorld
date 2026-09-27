
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

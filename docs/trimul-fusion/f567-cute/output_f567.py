"""Hopper F567: independent dual GEMM + saved projection/gate + dropout/residual.

Derived from the engine's from-scratch TM2 TMA/WGMMA implementation. No
cuequiv dependency. The candidate policy is external to this kernel.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90h
import torch
from cuda.bindings import driver as cuda

from miniworld_engine.kernels._compile import opaque
from cutlass import BFloat16, Float32, Int32
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum
from quack import copy_utils as quack_copy


_WARP_GROUP_THREADS = 128  # SM90 WGMMA instruction contract


class F567Sm90:
    """Two independent K reductions, tiled M/N, TMA loads and WGMMA math.

    Whole K operands are staged once. tile_k is a TMA/WGMMA chunk size, not
    a pretend pipeline-stage knob. Grouped M ordering controls weight reuse.
    """
    def __init__(self, N, KP, KG, L, tile_m, tile_n, tile_k, group_m):
        self.N, self.KP, self.KG, self.L = N, KP, KG, L
        self.tile_m, self.tile_n, self.tile_k = tile_m, tile_n, tile_k
        self.group_m = group_m
        self.g_loop = (KG + tile_k - 1) // tile_k
        self.p_loop = (KP + tile_k - 1) // tile_k
        self.num_threads = _WARP_GROUP_THREADS * (tile_m // 64)
        self.shared_storage = None


    @cute.kernel
    def kernel(
        self,
        tma_atom_X1: cute.CopyAtom,
        tX1: cute.Tensor,
        tma_atom_X2: cute.CopyAtom,
        tX2: cute.Tensor,
        tma_atom_W1: cute.CopyAtom,
        tW1: cute.Tensor,
        tma_atom_W2: cute.CopyAtom,
        tW2: cute.Tensor,
        atomO: cute.CopyAtom, atomP: cute.CopyAtom, atomG: cute.CopyAtom, atomR: cute.CopyAtom,
        sO_layout: cute.ComposedLayout,
        tO: cute.Tensor,
        tP: cute.Tensor, tG: cute.Tensor, tR: cute.Tensor, tD: cute.Tensor, tB: cute.Tensor,
        sX1_layout: cute.ComposedLayout, sX2_layout: cute.ComposedLayout,
        sW1_layout: cute.ComposedLayout, sW2_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        tx_bytes_total: Int32,
    ):
        TILE_M: cutlass.Constexpr[int] = self.tile_m
        TILE_N: cutlass.Constexpr[int] = self.tile_n
        TILE_K: cutlass.Constexpr[int] = self.tile_k
        G_LOOP: cutlass.Constexpr[int] = self.g_loop
        P_LOOP: cutlass.Constexpr[int] = self.p_loop

        tidx, _, _ = cute.arch.thread_idx()
        pid, _, _ = cute.arch.block_idx()
        nm = cute.ceil_div(tO.shape[0], TILE_M)
        nn = cute.ceil_div(self.N, TILE_N)
        first_m = (pid // (self.group_m * nn)) * self.group_m
        actual_group = min(self.group_m, nm - first_m)
        local = pid % (self.group_m * nn)
        m_block = first_m + local % actual_group
        n_block = local // actual_group
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sX1 = storage.sX1.get_tensor(sX1_layout.outer, swizzle=sX1_layout.inner)
        sX2 = storage.sX2.get_tensor(sX2_layout.outer, swizzle=sX2_layout.inner)
        sW1 = storage.sW1.get_tensor(sW1_layout.outer, swizzle=sW1_layout.inner)
        sW2 = storage.sW2.get_tensor(sW2_layout.outer, swizzle=sW2_layout.inner)
        sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)

        mbar_full_ptr = storage.mbar_full.data_ptr()
        if warp_idx == 0:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_X1)
                cpasync.prefetch_descriptor(tma_atom_X2)
                cpasync.prefetch_descriptor(tma_atom_W1)
                cpasync.prefetch_descriptor(tma_atom_W2)
                cpasync.prefetch_descriptor(atomR)
                cute.arch.mbarrier_init(mbar_full_ptr, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        gX1 = cute.local_tile(tX1, (TILE_M, TILE_K), (m_block, None))
        gX2 = cute.local_tile(tX2, (TILE_M, TILE_K), (m_block, None))
        gW1 = cute.local_tile(tW1, (TILE_N, TILE_K), (n_block, None))
        gW2 = cute.local_tile(tW2, (TILE_N, TILE_K), (n_block, None))


        load_X1, _, _ = quack_copy.tma_get_copy_fn(
            tma_atom_X1, 0, cute.make_layout(1), gX1, sX1,
        )
        load_X2, _, _ = quack_copy.tma_get_copy_fn(
            tma_atom_X2, 0, cute.make_layout(1), gX2, sX2,
        )
        load_W1, _, _ = quack_copy.tma_get_copy_fn(
            tma_atom_W1, 0, cute.make_layout(1), gW1, sW1,
        )
        load_W2, _, _ = quack_copy.tma_get_copy_fn(
            tma_atom_W2, 0, cute.make_layout(1), gW2, sW2,
        )

        gR=cute.local_tile(tR,(TILE_M,TILE_N),(m_block,n_block))
        load_R,_,_=quack_copy.tma_get_copy_fn(atomR,0,cute.make_layout(1),gR,sO,single_stage=True)
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_full_ptr, tx_bytes_total)
            load_R(tma_bar_ptr=mbar_full_ptr)
            for k in cutlass.range_constexpr(G_LOOP):
                load_X1(src_idx=k, dst_idx=k, tma_bar_ptr=mbar_full_ptr)
                load_W1(src_idx=k, dst_idx=k, tma_bar_ptr=mbar_full_ptr)
            for k in cutlass.range_constexpr(P_LOOP):
                load_X2(src_idx=k, dst_idx=k, tma_bar_ptr=mbar_full_ptr)
                load_W2(src_idx=k, dst_idx=k, tma_bar_ptr=mbar_full_ptr)

        cute.arch.mbarrier_wait(mbar_full_ptr, Int32(0))

        cute.arch.fence_view_async_shared()

        thr_mma = tiled_mma.get_slice(tidx)
        acc_shape = thr_mma.partition_shape_C((TILE_M, TILE_N))
        acc_G = cute.make_fragment(acc_shape, Float32)
        acc_V = cute.make_fragment(acc_shape, Float32)

        tCsX1 = thr_mma.make_fragment_A(thr_mma.partition_A(sX1))
        tCsX2 = thr_mma.make_fragment_A(thr_mma.partition_A(sX2))
        tCsW1 = thr_mma.make_fragment_B(thr_mma.partition_B(sW1))
        tCsW2 = thr_mma.make_fragment_B(thr_mma.partition_B(sW2))

        warpgroup.fence()
        mma_G = cute.make_mma_atom(tiled_mma.op)
        mma_V = cute.make_mma_atom(tiled_mma.op)
        mma_G.set(warpgroup.Field.ACCUMULATE, False)
        mma_V.set(warpgroup.Field.ACCUMULATE, False)

        per_stage_k = cute.size(tCsX1.shape[2])
        for stage in cutlass.range_constexpr(G_LOOP):
            for ki in cutlass.range_constexpr(per_stage_k):
                cute.gemm(mma_G, acc_G, tCsX1[None, None, ki, stage],
                          tCsW1[None, None, ki, stage], acc_G)
                mma_G.set(warpgroup.Field.ACCUMULATE, True)
        for stage in cutlass.range_constexpr(P_LOOP):
            for ki in cutlass.range_constexpr(per_stage_k):
                cute.gemm(mma_V, acc_V, tCsX2[None, None, ki, stage],
                          tCsW2[None, None, ki, stage], acc_V)
                mma_V.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()
        warpgroup.wait_group(0)

        coords = thr_mma.partition_C(cute.make_identity_tensor((TILE_M, TILE_N)))
        rR=cute.make_fragment_like(acc_G,Float32)
        rD=cute.make_fragment_like(acc_G,Float32)
        rB=cute.make_fragment_like(acc_G,Float32)
        rR.fill(0);rD.fill(0);rB.fill(0)
        for i in cutlass.range(cute.size(acc_G), unroll_full=True):
            row = cutlass.Int64(m_block) * TILE_M + coords[i][0]
            col = n_block * TILE_N + coords[i][1]
            if row < tO.shape[0] and col < self.N:
                rR[i]=sO[coords[i][0],coords[i][1]].to(Float32)
                rD[i]=tD[row%self.L,col].to(Float32)
                rB[i]=tB[col].to(Float32)
        pv=(acc_V.load()+rB.load()).to(BFloat16)
        gv=1.0/(1.0+cute.math.exp(-acc_G.load().to(BFloat16).to(Float32),fastmath=True))
        out_frag=cute.make_fragment_like(acc_G,BFloat16)
        out_frag.store((pv.to(Float32)*gv*rD.load()+rR.load()).to(BFloat16))
        cute.arch.barrier()
        store_op=sm90h.get_smem_store_op(LayoutEnum.ROW_MAJOR,BFloat16,Float32)
        copyC=cute.make_tiled_copy_C(store_op,tiled_mma).get_slice(tidx)
        for which in cutlass.range_constexpr(3):
            if cutlass.const_expr(which==0):
                target=tO;atom=atomO
            elif cutlass.const_expr(which==1):
                out_frag.store(pv);target=tP;atom=atomP
            else:
                out_frag.store(gv.to(BFloat16));target=tG;atom=atomG
            cute.copy(store_op,copyC.retile(out_frag),copyC.partition_D(sO))
            cute.arch.fence_view_async_shared();cute.arch.barrier()
            if warp_idx==0:
                dst=cute.local_tile(target,(TILE_M,TILE_N),(m_block,n_block))
                so,go=cpasync.tma_partition(atom,0,cute.make_layout(1),
                    cute.group_modes(sO,0,cute.rank(sO)),cute.group_modes(dst,0,cute.rank(dst)))
                cute.copy(atom,so,go)
                with cute.arch.elect_one():
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0,read=True)
            cute.arch.barrier()

    @cute.jit
    def __call__(
        self,
        mX1: cute.Tensor,
        mX2: cute.Tensor,
        mW1: cute.Tensor,
        mW2: cute.Tensor,
        mO: cute.Tensor,
        mP: cute.Tensor, mG: cute.Tensor, mR: cute.Tensor, mD: cute.Tensor, mB: cute.Tensor,
        stream: cuda.CUstream,
    ):
        TILE_M: cutlass.Constexpr[int] = self.tile_m
        TILE_N: cutlass.Constexpr[int] = self.tile_n
        TILE_K: cutlass.Constexpr[int] = self.tile_k
        G_LOOP: cutlass.Constexpr[int] = self.g_loop
        P_LOOP: cutlass.Constexpr[int] = self.p_loop

        M = mX1.shape[0]
        m_blocks = cute.ceil_div(M, TILE_M) * cute.ceil_div(self.N, TILE_N)

        atom = warpgroup.make_smem_layout_atom(
            sm90h.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, BFloat16, TILE_K), BFloat16)
        sX1_layout = cute.tile_to_shape(atom, (TILE_M, TILE_K, G_LOOP), order=(0,1,2))
        sX2_layout = cute.tile_to_shape(atom, (TILE_M, TILE_K, P_LOOP), order=(0,1,2))
        sW1_layout = cute.tile_to_shape(atom, (TILE_N, TILE_K, G_LOOP), order=(0,1,2))
        sW2_layout = cute.tile_to_shape(atom, (TILE_N, TILE_K, P_LOOP), order=(0,1,2))

        tiled_mma = sm90h.make_trivial_tiled_mma(
            BFloat16, BFloat16,
            warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            Float32,
            (TILE_M // 64, 1, 1),
            (64, TILE_N),
        )

        sX_stage = cute.slice_(sX1_layout, (None, None, 0))
        sW_stage = cute.slice_(sW1_layout, (None, None, 0))
        tma_atom_X1, tma_X1 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mX1, sX_stage, (TILE_M, TILE_K),
        )
        tma_atom_X2, tma_X2 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mX2, sX_stage, (TILE_M, TILE_K),
        )
        tma_atom_W1, tma_W1 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mW1, sW_stage, (TILE_N, TILE_K),
        )
        tma_atom_W2, tma_W2 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mW2, sW_stage, (TILE_N, TILE_K),
        )
        out_atom=warpgroup.make_smem_layout_atom(
            sm90h.get_smem_layout_atom(LayoutEnum.ROW_MAJOR,BFloat16,TILE_N),BFloat16)
        sO_layout=cute.tile_to_shape(out_atom,(TILE_M,TILE_N),order=(0,1))
        atomO,tO=cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(),mO,sO_layout,(TILE_M,TILE_N))
        atomP,tP=cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(),mP,sO_layout,(TILE_M,TILE_N))
        atomG,tG=cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(),mG,sO_layout,(TILE_M,TILE_N))
        so=cute.struct.Align[cute.struct.MemRange[BFloat16,cute.cosize(sO_layout)],1024]
        atomR,tR=cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(),mR,sO_layout,(TILE_M,TILE_N))
        tx_bytes_total = ((TILE_M + TILE_N) * TILE_K * (G_LOOP + P_LOOP) + TILE_M*TILE_N) * 2
        sx1 = cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(sX1_layout)], 1024]
        sx2 = cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(sX2_layout)], 1024]
        sw1 = cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(sW1_layout)], 1024]
        sw2 = cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(sW2_layout)], 1024]
        @cute.struct
        class SharedStorage:
            mbar_full: cute.struct.MemRange[cutlass.Int64, 1]
            sX1: sx1
            sX2: sx2
            sW1: sw1
            sW2: sw2
            sO: so

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_X1, tma_X1,
            tma_atom_X2, tma_X2,
            tma_atom_W1, tma_W1,
            tma_atom_W2, tma_W2,
            atomO,atomP,atomG,atomR,sO_layout,
            tO,tP,tG,tR,mD,mB,
            sX1_layout, sX2_layout, sW1_layout, sW2_layout,
            tiled_mma,
            Int32(tx_bytes_total),
        ).launch(
            grid=[m_blocks, 1, 1],
            stream=stream,
            block=[self.num_threads, 1, 1],  # 128 * (tile_m//64) warpgroups
        )



_COMPILE_CACHE = {}


def _fake(norm, x, wp, wg, residual, dropscale, seq_len, bias=None, config=None):
    return tuple(norm.new_empty((norm.shape[0], wp.shape[0])) for _ in range(3))


def _launch(norm, x, wp, wg_nk, residual, dropscale, seq_len, bias, outputs, config):
    tensors = (x, norm, wg_nk, wp, *outputs, residual, dropscale, bias)
    args = [from_dlpack(t, assumed_align=16) for t in tensors]
    args.append(cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream))
    from miniworld_engine.autotune.native import tensor_key
    key = (str(x.device), tensor_key(*tensors, extra=(seq_len,)), tuple(sorted(config.items())))
    if key not in _COMPILE_CACHE:
        kernel = F567Sm90(wp.shape[0], norm.shape[1], x.shape[1], seq_len, **config)
        _COMPILE_CACHE[key] = cute.compile(kernel, *args)
    _COMPILE_CACHE[key](*args)


def output_f567_impl(norm: torch.Tensor, x: torch.Tensor, wp: torch.Tensor,
                      wg: torch.Tensor, residual: torch.Tensor, dropscale: torch.Tensor,
                      seq_len: int, bias: torch.Tensor | None = None,
                      config: dict | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from miniworld_engine.autotune.cute_config import f567_candidates
    from miniworld_engine.autotune.native import choose_config, tensor_key
    m, kp = norm.shape
    kg, n = wg.shape
    operands = (norm, x, wp, wg, residual, dropscale)
    if not norm.is_cuda or torch.cuda.get_device_capability(norm.device) != (9, 0):
        raise ValueError('F567 CuTe requires SM90')
    if any(t.dtype != torch.bfloat16 or t.device != norm.device for t in operands):
        raise ValueError('F567 requires BF16 operands on the same CUDA device')
    if min(m,kp,kg,n,seq_len) <= 0 or any(v % 8 for v in (kp,kg,n)):
        raise ValueError('F567 TMA requires positive dimensions and widths aligned to 8 BF16 elements')
    if (tuple(x.shape)!=(m,kg) or tuple(wp.shape)!=(n,kp)
            or tuple(residual.shape)!=(m,n) or tuple(dropscale.shape)!=(seq_len,n)):
        raise ValueError('F567 input shapes disagree')
    if any(not t.is_contiguous() for t in (norm,x,wp,residual,dropscale)):
        raise ValueError('F567 expects contiguous activation/projection weight/residual/drop scale')
    if any(t.data_ptr()%16 for t in operands):
        raise ValueError('F567 requires 16-byte-aligned operands')
    if bias is None:
        bias = torch.zeros(n,device=norm.device,dtype=norm.dtype)
    if bias.shape != (n,) or bias.dtype != norm.dtype or bias.device != norm.device or not bias.is_contiguous() or bias.data_ptr()%16:
        raise ValueError('F567 bias layout/dtype/device mismatch')
    wg_nk = wg.t().contiguous()
    outputs = _fake(norm,x,wp,wg,residual,dropscale,seq_len)
    limit = torch.cuda.get_device_properties(norm.device).shared_memory_per_block_optin
    candidates = f567_candidates(kp,kg,limit)
    if config is None:
        config = choose_config('trimul_output_f567_sm90_cute', candidates,
            dtype=str(norm.dtype), bucket=tensor_key(norm,x,wp,wg_nk,residual,dropscale,bias,extra=(seq_len,limit)),
            device_index=norm.device.index,
            run=lambda c: _launch(norm,x,wp,wg_nk,residual,dropscale,seq_len,bias,outputs,c))
    if config not in candidates:
        raise ValueError('F567 config is outside the supported resource-feasible grid')
    _launch(norm,x,wp,wg_nk,residual,dropscale,seq_len,bias,outputs,config)
    return outputs


@opaque(fake=_fake, name='trimul_output_f567_sm90')
def output_f567_sm90(norm: torch.Tensor, x: torch.Tensor, wp: torch.Tensor,
                      wg: torch.Tensor, residual: torch.Tensor, dropscale: torch.Tensor,
                      seq_len: int, bias: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return output_f567_impl(norm,x,wp,wg,residual,dropscale,seq_len,bias)

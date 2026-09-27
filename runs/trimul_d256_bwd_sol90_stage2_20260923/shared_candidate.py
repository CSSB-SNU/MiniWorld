"""Explicit experiment selection for full validation; never changes dispatch."""
import os


def attach(plan):
    name = os.environ.get('SHARED_CANDIDATE', '')
    if not name:
        return None
    if name in ('saved_products','fused_saved_b1'):
        from saved_input_stats import enable as input_enable
        from saved_products import enable as product_enable
        base=input_enable(plan,gamma_cache=True)
        metadata=product_enable(plan);metadata['input_stats']=base
        if name=='fused_saved_b1':
            from fused_saved_b1 import enable as fuse
            products=metadata;metadata=fuse(plan);metadata['products']=products
        print('SHARED_CANDIDATE',metadata,flush=True)
        return metadata
    if name in ('input_stats','input_stats_gamma'):
        from saved_input_stats import enable
        metadata=enable(plan,gamma_cache=name=='input_stats_gamma')
        print('SHARED_CANDIDATE',metadata,flush=True)
        return metadata
    pipes = {'dx_pipe': '1', 'dx_pipe_unroll': '2', 'dx_unroll': '3',
             'dx_pipe_nobar': '4', 'dx_nobar': '5'}
    splits = {'dx_split': '1', 'dx_split_uniform': '2'}
    if name not in ('dx_n256', 'dx_rows128', 'dx_rows256', 'source_pairs') and name not in pipes and name not in splits:
        raise ValueError(f'Unknown SHARED_CANDIDATE: {name}')
    # Construct the control Training with the checkpoint's N128 finish, then
    # replace just the stage under test. Restore the environment afterwards.
    if any(os.environ.get(k, '0') != '0' for k in ('DX_N256', 'DX_PIPE', 'DX_SPLIT')):
        raise ValueError('Use SHARED_CANDIDATE with DX_N256=0 DX_PIPE=0 DX_SPLIT=0 for an explicit control')
    if name == 'source_pairs':
        from source_pairs import PairB7
        kernel = PairB7(plan.p, plan.b7)
        plan.b7 = kernel
    else:
        saved = os.environ.get('DX_N256')
        saved_pipe = os.environ.get('DX_PIPE')
        saved_split = os.environ.get('DX_SPLIT')
        try:
            os.environ['DX_N256'] = '0' if name == 'dx_rows128' or name in splits or name in pipes else '1'
            os.environ['DX_PIPE'] = pipes.get(name, '0')
            os.environ['DX_SPLIT'] = splits.get(name, '0')
            if name == 'dx_n256' or name in splits or name in pipes:
                from dx_ln import DxLN
                kernel = DxLN(plan.p, plan.b7.splits)
            else:
                from dx_ln_rows import DxLNRows
                kernel = DxLNRows(plan.p, plan.b7.splits)
        finally:
            if saved is None:
                os.environ.pop('DX_N256', None)
            else:
                os.environ['DX_N256'] = saved
            if saved_pipe is None:
                os.environ.pop('DX_PIPE', None)
            else:
                os.environ['DX_PIPE'] = saved_pipe
            if saved_split is None:
                os.environ.pop('DX_SPLIT', None)
            else:
                os.environ['DX_SPLIT'] = saved_split
        plan.b7.wide_finish = kernel
    metadata = dict(name=name, cubin=str(kernel.cubin), smem=kernel.smem,
                    splits=plan.b7.splits)
    if hasattr(kernel, 'grid'):
        metadata['grid'] = kernel.grid
    if hasattr(kernel, 'threads'):
        metadata['threads'] = kernel.threads
    print('SHARED_CANDIDATE', metadata, flush=True)
    return metadata

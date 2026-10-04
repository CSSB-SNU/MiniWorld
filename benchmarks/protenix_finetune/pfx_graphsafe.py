"""Device-side replacements for the host syncs in Protenix's training step (same distributions / values), and a fixed MSA
depth (Protenix draws a random depth on the host every call) so the step has static shapes for CUDA graphs. A real module
(not the runpy'd step script) so torch.compile can resolve these functions' globals."""
import os
import torch

N_TOKEN = None


MSA_ROWS = None      # [rows] bool: the rows of the current MSA sample that are real (random_masked mode), else None
MSA_K = None         # [1] float: their count


def random_masked_msa(feat_dict, dim_dict, cutoff=512, lower_bound=1, strategy="random"):
    """Protenix's sampling (depth k ~ U{min(lower_bound, n) .. n}, rows a random subset, cut at `cutoff`) with the depth drawn
    on the device: every row of a random permutation is kept (a static shape) and only the first k are marked real. The OPM
    -- the one op that mixes rows -- reads the mark; every other MSA op is row-wise, so the real rows see exactly Protenix."""
    global MSA_ROWS, MSA_K
    n = feat_dict["msa"].size(dim_dict["msa"])
    dev = feat_dict["msa"].device
    rows = min(n, cutoff) if cutoff > 0 else n
    k = torch.randint(low=min(lower_bound, n), high=n + 1, size=(1,), device=dev)
    idx = torch.randperm(n, device=dev)[:rows]
    MSA_ROWS = torch.arange(rows, device=dev) < k
    MSA_K = torch.clamp(k, max=rows).float()
    return {key: torch.index_select(feat_dict[key], dim, idx) for key, dim in dim_dict.items()}


def fixed_msa(feat_dict, dim_dict, cutoff=512, lower_bound=1, strategy="random"):
    n = feat_dict["msa"].size(dim_dict["msa"])
    want = os.environ["PFX_MSA_DEPTH"]
    d = (n + 1) // 2 if want == "half" else int(want)
    d = max(1, min(n, d, cutoff if cutoff > 0 else n))
    idx = torch.randperm(n, device=feat_dict["msa"].device)[:d]
    return {k: torch.index_select(feat_dict[k], dim, idx) for k, dim in dim_dict.items()}


def rot_dev(N_sample=1):
    """Haar-uniform rotations on the device: normalised Gaussian quaternions (scipy's Rotation.random is host-side)."""
    q = torch.randn(N_sample, 4, device="cuda", dtype=torch.float64)
    w, x, y, z = (q / q.norm(dim=-1, keepdim=True)).unbind(-1)
    r = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)
    return r.reshape(N_sample, 3, 3).float()


_agg = None


def agg_static(x_atom, atom_to_token_idx, n_token=None, reduce="mean"):
    """aggregate_atom_to_token with the token count given (else scatter sizes its output from index.max(): a host sync)."""
    return _agg(x_atom, atom_to_token_idx, n_token=N_TOKEN if n_token is None else n_token, reduce=reduce)


def install():
    global _agg
    import protenix.model.utils as pu
    import protenix.model.modules.transformer as pt
    import protenix.model.modules.pairformer as pf
    if os.environ.get("PFX_MSA_DEPTH") == "random_masked":
        pf.sample_msa_feature_dict_random_without_replacement = random_masked_msa
    elif os.environ.get("PFX_MSA_DEPTH"):
        pf.sample_msa_feature_dict_random_without_replacement = fixed_msa
    if os.environ.get("PFX_GRAPH_SAFE", "1") == "1":
        pu.uniform_random_rotation = rot_dev
        if _agg is None:
            _agg = pt.aggregate_atom_to_token
        pt.aggregate_atom_to_token = agg_static

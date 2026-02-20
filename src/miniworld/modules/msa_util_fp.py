import torch
import torch.nn.functional as F
from jaxtyping import Float

from miniworld.data.features.batch_edge_backprop import MSAFeatures

@torch.no_grad()
def init_msa(
    msa: MSAFeatures,
    recycle_idx: int,
    num_res_class: int = 32,
    use_fingerprint: bool = False,
    fp_table: torch.Tensor | None = None,
) -> Float[torch.Tensor, "B N L C"]:
    """Initialize MSA features for a given recycle index."""
    print('[Info] use_fingerprint inside init_msa: ', use_fingerprint)
    
    msa_sequences = msa.aligned_sequences[:, recycle_idx]
    msa_has_deletion = msa.has_deletion[:, recycle_idx]
    msa_deletion_value = msa.deletion_value[:, recycle_idx].float()

    if use_fingerprint:
        if fp_table is None:
            raise ValueError("fp_table must be provided when use_fingerprint=True.")
        fp_table = fp_table.to(msa_sequences.device)
        msa_sequences = F.embedding(msa_sequences.long(), fp_table)
    else:
        msa_sequences = torch.nn.functional.one_hot(
            msa_sequences.long(),
            num_classes=num_res_class,
        )
    msa_feat = torch.cat(
        [
            msa_sequences,
            msa_has_deletion.unsqueeze(-1).to(msa_sequences.dtype),
            msa_deletion_value.unsqueeze(-1).to(msa_sequences.dtype),
        ],
        dim=-1,
    )

    return msa_feat.float()
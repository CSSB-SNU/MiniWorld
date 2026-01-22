import torch
from jaxtyping import Float

from miniworld.data.features.batch_edge_backprop import MSAFeatures


@torch.no_grad()
def init_msa(
    msa: MSAFeatures,
    recycle_idx: int,
    num_res_class: int = 32,
) -> Float[torch.Tensor, "B N L C"]:
    """Initialize MSA features for a given recycle index."""
    msa_sequences = msa.aligned_sequences[:, recycle_idx]
    msa_has_deletion = msa.has_deletion[:, recycle_idx]
    msa_deletion_value = msa.deletion_value[:, recycle_idx].float()

    msa_sequences = torch.nn.functional.one_hot(
        msa_sequences.long(),
        num_classes=num_res_class,
    )
    msa_feat = torch.cat(
        [
            msa_sequences,
            msa_has_deletion.unsqueeze(-1),
            msa_deletion_value.unsqueeze(-1),
        ],
        dim=-1,
    )
    return msa_feat.float()

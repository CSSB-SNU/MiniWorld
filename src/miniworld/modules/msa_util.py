import torch
from jaxtyping import Float

from miniworld.data.features.batch_edge_backprop import MSAFeatures, SequenceFeatures


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


@torch.no_grad()
def init_msa_with_embedding(
    msa: MSAFeatures,
    token_embedding: torch.Tensor,
    recycle_idx: int,
    num_res_class: int = 32,
) -> Float[torch.Tensor, "B N L C"]:
    """Initialize MSA features for a given recycle index."""
    msa_sequences = msa.aligned_sequences[:, recycle_idx]
    msa_has_deletion = msa.has_deletion[:, recycle_idx]
    msa_deletion_value = msa.deletion_value[:, recycle_idx].float()

    token_embedding = token_embedding.to(msa.aligned_sequences.device)
    msa_sequences = torch.nn.functional.one_hot(
        msa_sequences.long(),
        num_classes=num_res_class,
    )  # B N L num_res_class
    msa_sequences = torch.einsum(
        "bnlc,cd->bnld",
        msa_sequences.float(),
        token_embedding.float(),
    )  # B N L embedding_dim
    msa_feat = torch.cat(
        [
            msa_sequences,
            msa_has_deletion.unsqueeze(-1),
            msa_deletion_value.unsqueeze(-1),
        ],
        dim=-1,
    )
    return msa_feat.float()


@torch.no_grad()
def init_token_single_msa(
    msa: MSAFeatures,
    sequence: SequenceFeatures,
    num_res_class: int = 32,
) -> Float[torch.Tensor, "B L_token d_single_token_init"]:
    """Initialize token single features with token embedding."""
    device = msa.aligned_sequences.device
    dtype = sequence.token_type.dtype
    token_type = torch.nn.functional.one_hot(
        sequence.token_type.long(),
        num_classes=num_res_class,
    ).to(device, dtype=dtype)

    return torch.concat(
        [
            token_type,
            msa.profile.to(dtype=dtype),
            msa.deletion_mean.unsqueeze(-1).to(dtype=dtype),
        ],
        dim=-1,
    )


@torch.no_grad()
def init_token_single_msa_with_embedding(
    msa: MSAFeatures,
    sequence: SequenceFeatures,
    token_embedding: torch.Tensor,
    num_res_class: int = 32,
) -> Float[torch.Tensor, "B L_token d_single_token_init"]:
    """Initialize token single features with token embedding."""
    device = msa.aligned_sequences.device
    dtype = token_embedding.dtype
    token_embedding = token_embedding.to(msa.aligned_sequences.device)
    token_type = torch.nn.functional.one_hot(
        sequence.token_type.long(),
        num_classes=num_res_class,
    ).to(device, dtype=dtype)
    token_type = torch.einsum(
        "blc,cd->bld",
        token_type.float(),
        token_embedding.float(),
    )  # B L embedding_dim
    msa_profile = torch.einsum(
        "blc,cd->bld",
        msa.profile.float(),
        token_embedding.float(),
    )  # B L embedding_dim

    token_single_msa = torch.concat(
        [
            token_type,
            msa_profile,
            msa.deletion_mean.unsqueeze(-1).to(dtype=dtype),
        ],
        dim=-1,
    )
    return token_single_msa.float()

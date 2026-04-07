import torch
from jaxtyping import Bool, Float

from miniworld.data.features import MSAFeatures, SequenceFeatures, TemplateFeatures


@torch.compile
@torch.no_grad()
def init_msa(
    msa: MSAFeatures,
    num_res_class: int = 32,
    dtype: torch.dtype = torch.float32,
) -> tuple[Float[torch.Tensor, "B N L C"], Bool[torch.Tensor, "B N"]]:
    """Initialize MSA features for a given recycle index."""
    msa_mask = msa.mask
    msa_sequences = msa.aligned_sequences
    msa_has_deletion = msa.has_deletion
    msa_deletion_value = msa.deletion_value.float()

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
    msa_feat = msa_feat * msa_mask[:, :, None, None]
    return msa_feat.to(dtype=dtype), msa_mask.bool()


@torch.no_grad()
def init_msa_with_embedding(
    msa: MSAFeatures,
    token_embedding: torch.Tensor,
    num_res_class: int = 32,
    dtype: torch.dtype = torch.float32,
) -> tuple[Float[torch.Tensor, "B N L C"], Bool[torch.Tensor, "B N"]]:
    """Initialize MSA features for a given recycle index."""
    msa_mask = msa.mask
    msa_sequences = msa.aligned_sequences
    msa_has_deletion = msa.has_deletion
    msa_deletion_value = msa.deletion_value.float()

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
    return msa_feat.to(dtype=dtype), msa_mask.bool()


@torch.no_grad()
def init_token_single_msa(
    msa: MSAFeatures,
    sequence: SequenceFeatures,
    num_res_class: int = 32,
    dtype: torch.dtype = torch.float32,
) -> Float[torch.Tensor, "B L_token d_single_token_init"]:
    """Initialize token single features with token embedding."""
    device = msa.aligned_sequences.device
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
    dtype: torch.dtype = torch.float32,
) -> Float[torch.Tensor, "B L_token d_single_token_init"]:
    """Initialize token single features with token embedding."""
    device = msa.aligned_sequences.device
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


# MiniWorld-style template features
@torch.compile
@torch.no_grad()
def init_template_feat(
    template: TemplateFeatures,
    dtype: torch.dtype = torch.float32,
    positive_cutoff: float = 6.0,
    negative_cutoff: float = 12.0,
) -> Float[torch.Tensor, "B L L 4"]:
    """Initialize MiniWorld-style template pair classes.

    Output channels:
        [:, :, :, 0]: definite contact
        [:, :, :, 1]: definite negative
        [:, :, :, 2]: ambiguous distance or mixture with ambiguous templates only
        [:, :, :, 3]: multistate (both contact and negative observed)

    Unknown / masked pairs are encoded as all zeros.
    """
    template_ids = template.ids  # (B, T, L)
    template_mask = template.mask[:, :, None, None].bool()  # (B, T, 1, 1)

    # Ignore inter-chain pairs when building per-template pair classes.
    same_chain = (
        template_ids[:, :, :, None] == template_ids[:, :, None, :]
    )  # (B, T, L, L)

    cb_xyz = template.cb_xyz  # (B, T, L, 3)
    cb_mask = template.cb_mask.bool()  # (B, T, L)
    cb_dist = torch.norm(
        cb_xyz[:, :, :, None, :] - cb_xyz[:, :, None, :, :],
        dim=-1,
    )  # (B, T, L, L)
    cb_pair_mask = cb_mask[:, :, :, None] & cb_mask[:, :, None, :]  # (B, T, L, L)
    valid_pair_mask = template_mask & same_chain & cb_pair_mask

    unknown = torch.full_like(cb_dist, 4, dtype=torch.long)
    is_contact = valid_pair_mask & (cb_dist < positive_cutoff)
    is_negative = valid_pair_mask & (cb_dist > negative_cutoff)
    is_ambiguous = (
        valid_pair_mask & (cb_dist >= positive_cutoff) & (cb_dist <= negative_cutoff)
    )

    per_template_feat = torch.where(
        is_contact,
        torch.zeros_like(unknown),
        unknown,
    )
    per_template_feat = torch.where(
        is_negative,
        torch.ones_like(per_template_feat),
        per_template_feat,
    )
    per_template_feat = torch.where(
        is_ambiguous,
        torch.full_like(per_template_feat, 2),
        per_template_feat,
    )

    has_contact = (per_template_feat == 0).any(dim=1)
    has_negative = (per_template_feat == 1).any(dim=1)
    has_ambiguous = (per_template_feat == 2).any(dim=1)
    has_known = (per_template_feat != 4).any(dim=1)

    both_contact_and_negative = has_contact & has_negative
    ambiguous_only = (~both_contact_and_negative) & has_ambiguous
    contact_only = has_contact & ~has_negative & ~has_ambiguous
    negative_only = has_negative & ~has_contact & ~has_ambiguous

    contact_feat = torch.full_like(has_contact, 4, dtype=torch.long)
    contact_feat = torch.where(
        both_contact_and_negative,
        torch.full_like(contact_feat, 3),
        contact_feat,
    )
    contact_feat = torch.where(
        ambiguous_only,
        torch.full_like(contact_feat, 2),
        contact_feat,
    )
    contact_feat = torch.where(
        contact_only,
        torch.zeros_like(contact_feat),
        contact_feat,
    )
    contact_feat = torch.where(
        negative_only,
        torch.ones_like(contact_feat),
        contact_feat,
    )
    contact_feat = torch.where(
        has_known,
        contact_feat,
        torch.full_like(contact_feat, 4),
    )

    # Unknown / masked pairs stay all-zero instead of using a dedicated channel.
    contact_feat = torch.nn.functional.one_hot(
        contact_feat.clamp(max=3),
        num_classes=4,
    ) * (contact_feat != 4).unsqueeze(-1)

    return contact_feat.to(dtype=dtype)


@torch.inference_mode()
def apply_template_dropout(
    contact_feat: Float[torch.Tensor, "B L L 4"],
    prob: float = 0.0,
    dtype: torch.dtype = torch.float32,
) -> Float[torch.Tensor, "B L L 4"]:
    """Apply template dropout to MiniWorld-style template pair classes."""
    if prob > 0.0:
        keep_mask = (
            torch.rand(
                (4,),
                device=contact_feat.device,
            )
            >= prob
        )
        contact_feat = contact_feat * keep_mask.view(1, 1, 1, 4).to(contact_feat.dtype)

    return contact_feat.to(dtype=dtype)

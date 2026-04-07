import torch
from jaxtyping import Bool, Float

from miniworld.data.features import MSAFeatures, SequenceFeatures, TemplateFeatures

import torch.nn.functional as F

@torch.no_grad()
def init_msa_explicit(
    msa: MSAFeatures,
    token_embedding: torch.Tensor,              # (V, D)
    profile32_to_fp_index: torch.Tensor,        # (32,), long
    dtype: torch.dtype = torch.float32,
) -> tuple[Float[torch.Tensor, "B N L C"], Bool[torch.Tensor, "B N"]]:
    """Initialize MSA features using fingerprint embedding lookup (no one_hot)."""
    msa_mask = msa.mask
    msa_sequences = msa.aligned_sequences
    msa_has_deletion = msa.has_deletion
    msa_deletion_value = msa.deletion_value.float()

    device = msa.aligned_sequences.device
    emb = token_embedding.to(device).float()
    idx32 = profile32_to_fp_index.to(device)

    # Map canonical class ids (0..31) -> fingerprint table row ids (0..V-1)
    msa_sequences = msa_sequences.long().clamp_(0, idx32.numel() - 1)
    fp_ids = idx32[msa_sequences]                                  # (B, N, L)
    msa_sequences_fp = F.embedding(fp_ids, emb)                    # (B, N, L, D)

    msa_feat = torch.cat(
        [
            msa_sequences_fp,
            msa_has_deletion.unsqueeze(-1),
            msa_deletion_value.unsqueeze(-1),
        ],
        dim=-1,
    )
    return msa_feat.to(dtype=dtype), msa_mask.bool()

@torch.no_grad()
def init_token_single_msa_explicit(
    msa: MSAFeatures,
    sequence: SequenceFeatures,
    token_embedding: torch.Tensor,
    profile32_to_fp_index: torch.Tensor,    # (32,), long
    token_type_is_fp_index: bool = True,
    dtype: torch.dtype = torch.float32,
) -> Float[torch.Tensor, "B L_token d_single_token_init"]:
    device = msa.aligned_sequences.device
    dtype = sequence.token_type.dtype
    emb = token_embedding.to(device)
    idx32 = profile32_to_fp_index.to(device).long()

    # 1) token identity in fingerprint space
    token_ids = sequence.token_type.long()
    if not token_type_is_fp_index:
        token_ids = idx32[token_ids.clamp(0, idx32.numel() - 1)]
    token_fp = F.embedding(token_ids, emb).to(dtype=dtype) 

    # 2) msa.profile (B, L, 32) -> fingerprint-space msa_profile (B, L, D_fp)
    fp_rows_for_32 = emb.index_select(0, idx32)             # (32, D_fp)
    msa_profile = torch.einsum(
        "blc,cd->bld",
        msa.profile.float(),
        fp_rows_for_32,
    ).to(dtype=dtype)

    # 3) final token single init
    return torch.cat(
        [
            token_fp,
            msa_profile,
            msa.deletion_mean.unsqueeze(-1).to(dtype=dtype),
        ],
        dim=-1,
    )
    
# MiniWorld-style template features
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
    template_mask = template.mask[:, :, None, None]  # (B, T, 1, 1)

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

    per_template_feat = torch.full_like(cb_dist, 4, dtype=torch.long)
    per_template_feat[valid_pair_mask & (cb_dist < positive_cutoff)] = 0
    per_template_feat[valid_pair_mask & (cb_dist > negative_cutoff)] = 1
    per_template_feat[
        valid_pair_mask & (cb_dist >= positive_cutoff) & (cb_dist <= negative_cutoff)
    ] = 2

    has_contact = (per_template_feat == 0).any(dim=1)
    has_negative = (per_template_feat == 1).any(dim=1)
    has_ambiguous = (per_template_feat == 2).any(dim=1)
    has_known = (per_template_feat != 4).any(dim=1)

    contact_feat = torch.full_like(has_contact, 4, dtype=torch.long)
    contact_feat[has_contact & has_negative] = 3
    contact_feat[~(has_contact & has_negative) & has_ambiguous] = 2
    contact_feat[has_contact & ~has_negative & ~has_ambiguous] = 0
    contact_feat[has_negative & ~has_contact & ~has_ambiguous] = 1
    contact_feat[~has_known] = 4

    # Unknown / masked pairs stay all-zero instead of using a dedicated channel.
    contact_feat = torch.nn.functional.one_hot(
        contact_feat.clamp(max=3),
        num_classes=4,
    ) * (contact_feat != 4).unsqueeze(-1)

    return contact_feat.to(dtype=dtype)

from __future__ import annotations

from pydantic import BaseModel
from team_gm.modules import ImplementationType


class SharedConfig(BaseModel):
    """Shared configuration for StructureFlow model."""

    d_single: int = 384
    d_single_atom: int = 128
    d_single_token: int = 768
    d_single_token_input: int = 441  # 441 = 384 + 32 + 24 + 1

    d_pair: int = 128
    d_pair_template: int = 128
    d_pair_atom: int = 16

    d_time: int = 256

    r_max: int = 32
    s_max: int = 2

    dgram_bins_template: int = 39
    relpos_bins: int = 32
    noise_freq: int = 256
    num_res_class: int = 32

    n_distogram_bins: int = 64

    implementation: ImplementationType = ImplementationType.PYTORCH
    use_checkpoint: bool = False

    # AF3 SI table 5 ref_* features. False drops ref_element and ref_charge from
    # the atom encoders, leaving ref_pos / ref_mask / ref_space_uid -- so the
    # reference conformer geometry is still given but its chemistry is not, and
    # a token-identity embedding has to supply that instead. Narrows
    # atom_single_init from 6 channels to 4; atom_pair_init is unaffected,
    # being built from ref_pos and ref_space_uid. Leave True to keep existing
    # checkpoints loadable.
    use_reference_chemistry: bool = True

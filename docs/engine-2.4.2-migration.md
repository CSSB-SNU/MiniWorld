# miniworld-engine 2.4.2: what moved

miniworld-engine 2.4.2 removed `miniworld_engine.architecture`, `miniworld_engine.ops` and moved `integrations` / `tools` / `_typecheck`
(engine `docs/CHANGELOG.md`, 2.4.2 "Removed"). MiniWorld pins `miniworld-engine==2.4.2`.

| was | now |
|---|---|
| `architecture.diffusion.*`, `architecture.utils.{align,se3,transform}`, `architecture.tensor_layout` | `foldforge.diffusion` (`.geometry.{alignment,se3,kabsch}`, `.tensor_layout`) |
| `architecture.exceptions`, `architecture._engine_impl`, `architecture.blocks._engine_impl` | `foldforge.layers.implementation` |
| `architecture.blocks.pairformer` | `foldforge.blocks.pairformer` |
| `architecture.blocks.{diffusion_transformer,msa_module,pair_bias_hoist,rope_swa_af3_transformer,swa_atom_transformer,template}` | `miniworld.blocks.*` |
| `architecture.primitives` (Linear / MPLinear / LayerNorm / `convert_linears_to_mp`) | `miniworld.layers.primitives` |
| `architecture.layers.{embeddings,adaln,augmented_attention}`, `architecture.layers.ops` | `miniworld.layers.{embeddings,adaln,augmented_attention,functional}` |
| `architecture.typing.typecheck` | `miniworld_engine.utils.typecheck.typecheck` |
| `modules.swa_atom_attention.module.build_local_structure_neighbor_indices` (deleted in 2.4.2) | `miniworld.blocks.swa_neighbors` |
| `ops.augmented_attention_pair_bias` / `ops.layer_norm_linear` | `kernels.augmented_attention.whole_op.augmented_attention_pair_bias` / `kernels.layernorm_linear.interface.layer_norm_linear` |
| `miniworld_engine.integrations.*` | `miniworld_engine.kernels.integrations.*` (`optimizer` -> `miniworld_engine.utils.optimizer`) |

The shared pieces come from FoldForge (`foldforge` is a path dependency); MiniWorld imports only the self-contained
`foldforge.diffusion`, `foldforge.blocks` and `foldforge.layers`.

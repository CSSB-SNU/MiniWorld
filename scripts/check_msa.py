import sys
from pathlib import Path
sys.path.insert(0, str(Path("scripts").resolve()))
from omegaconf import OmegaConf
from miniworld.configs.models import AtomSWAConfig
from miniworld.models.distogram_only import MiniSWAModel
m = OmegaConf.to_container(OmegaConf.load("configs/miniworld/model/medium_distogram.yaml"), resolve=True)
m["trunk"]["pairformer"]["n_block"] = 48
mdl = MiniSWAModel(MiniSWAModel.Config(
    shared=m["shared"], input_feat_embbeder=m["input_feat_embbeder"],
    atom_swa=AtomSWAConfig(enabled=True, backend="flash", swa_window_size=1000000), trunk=m["trunk"]))
b = mdl.msa_module.blocks
print("MSA blocks=", len(b),
      "| block0 has MPWA=", hasattr(b[0], "msa_pair_weighted_averaging"),
      "transition_msa=", hasattr(b[0], "transition_msa"),
      "| last block MPWA=", hasattr(b[-1], "msa_pair_weighted_averaging"))
print("total params %.2fM" % (sum(p.numel() for p in mdl.parameters()) / 1e6))

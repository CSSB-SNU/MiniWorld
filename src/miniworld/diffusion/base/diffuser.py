"""Re-export shim — promoted to team_gm.diffusion.base.diffuser."""
from team_gm.diffusion.base.diffuser import Diffuser, _expand_to_trailing_dims

__all__ = ["Diffuser", "_expand_to_trailing_dims"]

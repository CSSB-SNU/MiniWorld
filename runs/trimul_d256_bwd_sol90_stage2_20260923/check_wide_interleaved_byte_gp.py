"""Screen warp-local expansion after 4D TMA byte-plane interleaving."""
from pathlib import Path
body=Path(__file__).with_name('check_wide_split_byte_gp.py').read_text()
body=body.replace('from wide_split_byte_gp import SplitBytePre,SplitByteGP','from wide_split_byte_gp import SplitBytePre\nfrom wide_interleaved_byte_gp import InterleavedByteGP as SplitByteGP')
body=body.replace('SplitBytePre(plan.pre,xor)','SplitBytePre(plan.pre,xor,interleaved=True)')
body=body.replace('wide-split-byte-gp','wide-interleaved-byte-gp')
exec(compile(body,__file__,'exec'))

"""Smaller dW opcodes avoid N128's 90-register operand lower bound."""
from pathlib import Path
body=Path(__file__).with_name('check_d256_three_aliased_source.py').read_text()
body=body.replace('op=ThreeAliasedSource(plan,readwait)','op=ThreeAliasedSource(plan,readwait,tile_n=64)')
body=body.replace("'d256-three-aliased-source'","'d256-three-aliased-n64'")
exec(compile(body,__file__,'exec'))

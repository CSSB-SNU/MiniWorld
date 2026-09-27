"""Alias projection Z/weights with attention Q/bias; project KV before Q/gate."""
from pathlib import Path

R = Path(__file__).resolve().parent
base = (R / 'qkv_compact_remat/fused.cu').read_text()
old = '''  union alignas(1024) Scratch {
    struct { array_aligned<Element,8192,1024> z[Projectors]; array_aligned<Element,8192,1024> w[2]; } proj;
    array_aligned<Element,4096,1024> bias[Consumers][Stages];
  };
  struct Shared {
    array_aligned<Element,2048,1024> q[(Consumers>2*Projectors?Consumers:2*Projectors)],kv[2][Capacity/64];
'''
new = '''  union alignas(1024) Scratch {
    struct { array_aligned<Element,8192,1024> z[Projectors]; array_aligned<Element,8192,1024> w[2]; } proj;
    struct {
      array_aligned<Element,2048,1024> q[Consumers];
      array_aligned<Element,4096,1024> bias[Consumers][Stages];
    } attn;
  };
  struct Shared {
    array_aligned<Element,2048,1024> kv[2][Capacity/64];
'''
assert old in base
s = base.replace(old, new, 1)
# Each projector writes K/V first. The Q/gate MMA then retires its Z reads;
# its two outputs can overwrite the first half of the now-dead Z tile.
s = s.replace('for(int pair=0;pair<2;++pair) {',
              'for(int turn=0;turn<2;++turn) {\n        int pair=1-turn;', 1)
s = s.replace('pair==0?s.q[c+which*Projectors].data():s.kv[which][qt].data()',
              'pair==0?s.scratch.proj.z[c].data()+which*2048:s.kv[which][qt].data()')
assert 's.q[c+which*Projectors]' not in s
s = s.replace('s.scratch.bias', 's.scratch.attn.bias')
s = s.replace('s.q[c].data()', 's.scratch.attn.q[c].data()')
s = s.replace('s.q[ep_c].data()', 's.scratch.attn.q[ep_c].data()')
assert 's.q[' not in s
s = s.replace('// Reused Q/gate sources must retire; global completion can overlap later tiles.',
              '// Q/gate sources alias Z: retire source reads before loading the next Z tile.')
s = s.replace('launch<384,6,2,3>', 'launch<384,6,2,6>')
s = s.replace('launch<768,6,2,4>', 'launch<768,6,2,6>')
s = s.replace('launch<1024,6,1,2>', 'launch<1024,6,1,4>')
s = s.replace('Config<1024,6,1,2>::Shared', 'Config<1024,6,1,4>::Shared')
for name, code in [('qkv_compact_alias', s),
                   ('qkv_compact_alias4', s.replace('launch<384,6,2,6>', 'launch<384,6,2,3>')
                    .replace('launch<768,6,2,6>', 'launch<768,6,2,4>'))]:
    target = R / name
    assert not (target / 'build-ready.json').exists(), name
    target.mkdir(exist_ok=True)
    (target / 'fused.cu').write_text(code)

# Independent control: retain the prior layout but wait only until the Q/gate
# save group retires, leaving the newest K/V save group in flight.
s = base.replace('if(lane==0)tma_store_wait<0>();', 'if(lane==0)tma_store_wait<1>();', 1)
target = R / 'qkv_compact_save_overlap'
assert not (target / 'build-ready.json').exists()
target.mkdir(exist_ok=True)
(target / 'fused.cu').write_text(s)

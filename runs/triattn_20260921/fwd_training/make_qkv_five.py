"""Full QKV resident control: five consumers, compared at the four-projection boundary."""
from pathlib import Path

r = Path(__file__).resolve().parent
s = (r / 'resident_kv_parallel6b/fused.cu').read_text()
s = s.replace('launch<768,6,2>', 'launch<768,5,2>')
s = s.replace('launch<1024,6,1>', 'launch<1024,5,1>')
s = s.replace('Config<1024,6,1>::Shared', 'Config<1024,5,1>::Shared')
s = s.replace(
    'launch_forward(torch::Tensor z,torch::Tensor wq,torch::Tensor wk,torch::Tensor wv,torch::Tensor b)',
    'launch_forward(torch::Tensor z,torch::Tensor wq,torch::Tensor wk,torch::Tensor wv,torch::Tensor wg,torch::Tensor b)')
s = s.replace('for(auto const& w:{wq,wk,wv})', 'for(auto const& w:{wq,wk,wv,wg})')
s = s.replace('  auto q=torch::empty_like(z)',
              '  auto gate=at::linear(z,wg);\n  auto q=torch::empty_like(z)')
s = s.replace('return {view(o),lse,view(q),view(k),view(v)};',
              'return {view(o),lse,view(q),view(k),view(v),gate};')
assert 'torch::Tensor wg,torch::Tensor b)' in s
d = r / 'qkv_five'
assert not (d / 'build-ready.json').exists()
d.mkdir(exist_ok=True)
(d / 'fused.cu').write_text(s)

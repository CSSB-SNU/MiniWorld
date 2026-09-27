"""Put completion before compiler-generated loop-backedge accumulator copies."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--artifact',required=True)
p.add_argument('--width',type=int,choices=(32,64),required=True)
p.add_argument('--consumers',type=int,choices=(2,4),required=True)
a=p.parse_args();r=Path(__file__).resolve().parent
base='pipeclean32_c4' if a.width==32 else 'pipeclean64_c2'
s=(r/base/'fused.cu').read_text();oldcons=4 if a.width==32 else 2
start=s.index('      auto step=[&](int kt){');end=s.index('      issue_qk(score,0);',start)
part=s[start:end]
part=part.replace('        warpgroup_commit_batch();',
'''        warpgroup_commit_batch();
        // NVCC emits loop-phi register copies before the next step's first wait.
        // Retire writers here so those copies cannot force WGMMA serialization.
        warpgroup_wait<0>();
        warpgroup_fence_operand(score);warpgroup_fence_operand(out);
        warpgroup_fence_operand(sums);warpgroup_fence_operand(pr);''')
s=s[:start]+part+s[end:]
for cap in (384,768,1024):
    s=s.replace(f'launch<{cap},{oldcons},2,{oldcons}>',f'launch<{cap},{a.consumers},2,{a.consumers}>')
s=s.replace(f'Config<1024,{oldcons},2,{oldcons}>::Shared',f'Config<1024,{a.consumers},2,{a.consumers}>::Shared')
s=s.replace(f'qkv_attention_n{a.width}_copy_pipeline',f'qkv_attention_n{a.width}_endwait')
folder=r/a.artifact;assert not (folder/'build-ready.json').exists()
folder.mkdir(exist_ok=True);(folder/'fused.cu').write_text(s)

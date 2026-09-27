"""Release each bias tile after its last warp-local ldmatrix read, before softmax."""
from pathlib import Path
r = Path(__file__).resolve().parent
s = (r / 'cooperative_cluster2_v2/fused.cu').read_text()
s = s.replace('s.bfull[b].init(1); s.bempty[b].init(Config::CR);',
              's.bfull[b].init(4); s.bempty[b].init(4*Config::CR);')
old = '''  if(tid==0) {
    s.bfull[0].arrive_and_expect_tx(4096*sizeof(Element));
    if(nt>1)s.bfull[1].arrive_and_expect_tx(4096*sizeof(Element));
  }'''
assert old in s
s = s.replace(old, '''  if(tid%32==0) {
    if(tid==0) {
      s.bfull[0].arrive_and_expect_tx(4096*sizeof(Element));
      if(nt>1)s.bfull[1].arrive_and_expect_tx(4096*sizeof(Element));
    } else {
      s.bfull[0].arrive();
      if(nt>1)s.bfull[1].arrive();
    }
  }''')
old = '''        if(tid==0 && kt>0 && kt+1<nt) {
          int free_stage=(kt+1)%2;
          s.bfull[free_stage].arrive_and_expect_tx(4096*sizeof(Element));
          s.bempty[free_stage].arrive(uint32_t(0),uint32_t(1));
          load(kt+1);
        }'''
assert old in s
s = s.replace(old, '        if(tid==0 && kt>0 && kt+1<nt)load(kt+1);')
old = '        float alpha[2], ls[2]={0.f,0.f};'
assert old in s
s = s.replace(old, '''        if(kt+2<nt) {
          // A warp owns 16 distinct bias rows. Its collective ldmatrix reads
          // and dependent scalar accesses finish before its release arrives.
          __syncwarp();
          if(lane%32==0) {
            if(lane==0)s.bfull[stage].arrive_and_expect_tx(4096*sizeof(Element));
            else s.bfull[stage].arrive();
            // Eight arrivals (four per CTA) prove all recipients are ready.
            s.bempty[stage].arrive(uint32_t(0),uint32_t(1));
          }
        }
        float alpha[2], ls[2]={0.f,0.f};''')
p = r / 'cooperative_cluster2_early'
p.mkdir(exist_ok=True)
(p / 'fused.cu').write_text(s)

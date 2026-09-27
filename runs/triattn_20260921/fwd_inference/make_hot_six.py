"""Retain six attention warpgroups; compare quantized denominator and peak guard."""
from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'hot4r/fused.cu').read_text()
for cap,stages in ((384,2),(768,1),(1024,1)):
    s=s.replace('launch<%d,4,2,4>'%cap,'launch<%d,6,%d,6>'%(cap,stages))
s=s.replace('Config<1024,4,2,4>::Shared','Config<1024,6,1,6>::Shared')
peak=s.replace('    float l[2];', '    float l[2],peak[2];')
peak=peak.replace('    clear(out);', '    clear(out);peak[0]=peak[1]=0.f;')
peak=peak.replace('if constexpr(Safe) ls[mi]+=score(x); else ls[mi]+=float(Element(score(x)));',
'''ls[mi]+=score(x);
          if constexpr(!Safe) peak[mi]=fmaxf(peak[mi],score(x));''')
peak=peak.replace('          l[mi]+=__shfl_xor_sync(0xffffffffu,l[mi],2);',
'''          l[mi]+=__shfl_xor_sync(0xffffffffu,l[mi],2);
          peak[mi]=fmaxf(peak[mi],__shfl_xor_sync(0xffffffffu,peak[mi],1));
          peak[mi]=fmaxf(peak[mi],__shfl_xor_sync(0xffffffffu,peak[mi],2));''')
peak=peak.replace('    #pragma unroll\n    for(int x=0;x<size(out);++x) bad=',
'''    bad=bad || peak[0]>.9f*l[0] || peak[1]>.9f*l[1];
    #pragma unroll
    for(int x=0;x<size(out);++x) bad=''')
for name,source in [('hot6r',s),('hot6p',peak)]:
    path=r/name
    assert not (path/'build-ready.json').exists()
    path.mkdir(exist_ok=True)
    (path/'fused.cu').write_text(source)

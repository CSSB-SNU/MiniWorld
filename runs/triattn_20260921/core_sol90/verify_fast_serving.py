"""Verify installed dispatch, masks, opt-out, and both shape-specific hot kernels."""
from pathlib import Path
R=Path(__file__).resolve().parents[1]
src=R/'core_tiles/bench_serving.py'
s=src.read_text()
s=s.replace("and 'Traits<0>' in n", "and ('Traits<0>' in n or 'Traits<1073741824>' in n or 'Traits<536870912>' in n)")
marker='  results[name].update(bitwise_equal=True,kernels=kernels)'
s=s.replace(marker,'''  if name=='broadcast':
   flag='Traits<1073741824>' if a.length==768 else 'Traits<'+os.environ.get('VERIFY_FLAG1024','536870912')+'>'
   assert any('ta_core_broadcast::triattn_m1_kernel' in key and flag in key for key in kernels),kernels
'''+marker)
exec(compile(s,str(src),'exec'),dict(__file__=str(src),__name__='__main__'))

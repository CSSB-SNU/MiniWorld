"""Reuse the established M1 accuracy cases on an isolated compiled candidate."""
from pathlib import Path
import os
HERE=Path(__file__).resolve().parent
mode=os.environ['SOL_VARIANT']
sanitizer=os.environ.get('SOL_SANITIZER')=='1'
src=HERE.parent/'core_tiles'/('check_sanitizer.py' if sanitizer else 'check_broadcast.py')
s=src.read_text()
s=s.replace('from build_broadcast import triattn_m1 as candidate\ncandidate._build()', '''candidate=module('candidate_check',R/'core_sol90'/MODE/'triattn_m1.py')
candidate._EXT=module('triattn_sol_'+MODE,R/'core_sol90/build'/('triattn_sol_'+MODE)/('triattn_sol_'+MODE+'.so'))''')
if sanitizer:
 start=s.index('from build_broadcast import triattn_m1 as candidate')
 end=s.index("os.environ['FPF_TRIATT_MASK_STAGE']",start)
 s=s[:start]+'''candidate=module('candidate_check',R/'core_sol90'/MODE/'triattn_m1.py')
candidate._EXT=module('triattn_sol_'+MODE,R/'core_sol90/build'/('triattn_sol_'+MODE)/('triattn_sol_'+MODE+'.so'))
'''+s[end:]
s=s.replace("'0,256'", "'0'")
if mode=='basefast768' or os.environ.get('SOL_FULLCASE')=='1':
 full_length=int(os.environ.get('SOL_FULL_LENGTH','768'))
 assert full_length in (768,1024)
 s=s.replace('for L in [768,1024]:','for L in [%d]:'%full_length)
 s=s.replace('B,N,H=2,7,4','B,N,H=2,%d,4'%full_length)
 s=s.replace('B,N,H=2,2,1','B,N,H=1,%d,4'%full_length)
if mode in ('installed','packaged','producerpackaged','f40packaged','fixeddescpackaged','qsmallpackaged','qthreepackaged','q1024packaged'):
 root="R/'core_sol90/qsmall_package'" if mode=='qsmallpackaged' else "R/'core_sol90/fixeddesc_package'" if mode=='fixeddescpackaged' else "R/'oc/opt_core/kernels/triattn_core_broadcast'" if mode=='installed' else "R/'core_sol90/producer_package'" if mode=='producerpackaged' else "R/'core_sol90/f40_package'" if mode=='f40packaged' else "R/'core_sol90/fast_package'"
 if mode=='qthreepackaged':root="R/'core_sol90/qthree_package'"
 if mode=='q1024packaged':root="R/'core_sol90/q1024_package'"
 s=s.replace("candidate=module('candidate_check',R/'core_sol90'/MODE/'triattn_m1.py')", "candidate=module('candidate_check',("+root+")/'build_source.py')")
 s=s.replace("candidate._EXT=module('triattn_sol_'+MODE,R/'core_sol90/build'/('triattn_sol_'+MODE)/('triattn_sol_'+MODE+'.so'))", "candidate._EXT=module('triattn_broadcast',("+root+")/'triattn_broadcast.so')")
if os.environ.get('CHECK_PATTERNS'):
 import re
 s=re.sub(r"for pattern in \[.*?\]:", 'for pattern in '+repr(os.environ['CHECK_PATTERNS'].split(','))+':',s)
if os.environ.get('CHECK_SCALE'):
 s=s.replace('32**-.5',repr(float(os.environ['CHECK_SCALE'])))
s=s.replace("R/'core_tiles/check_broadcast.json'", "R/'core_sol90'/('check_'+MODE+'.json')")
s=s.replace("R/'core_tiles/check_sanitizer.json'", "R/'core_sol90'/('sanitize_'+MODE+'.json')")
full_suffix='-full1024.json' if os.environ.get('SOL_FULL_LENGTH')=='1024' else '-full.json'
if os.environ.get('SOL_FULLCASE')=='1':s=s.replace("+MODE+'.json'", "+MODE+'"+full_suffix+"'")
if os.environ.get('CHECK_SCALE'):
 suffix=full_suffix if os.environ.get('SOL_FULLCASE')=='1' else '.json'
 s=s.replace("+MODE+'"+suffix+"'", "+MODE+'-scale"+suffix+"'")
exec(compile(s,str(src),'exec'),dict(__file__=str(src),__name__='__main__',MODE=mode))

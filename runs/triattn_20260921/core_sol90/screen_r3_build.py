"""Check both kernels, including their initial dynamic-register pool budget."""
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()
good=True;found=0
for chunk in re.split(r'(?=ptxas info\s+: Compiling entry function)',s):
 first=chunk.splitlines()[0] if chunk else ''
 if 'attentionILb' not in first:continue
 found+=1
 registers=re.search(r'Used (\d+) registers',chunk)
 spills=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',chunk)
 # 3*160 consumer regs +32 producer regs require 65536 physical registers.
 # At512 threads the launch must reserve at least128 per thread.
 enough=registers is not None and int(registers.group(1))*512 >= (3*160+32)*128
 clean=spills is not None and int(spills.group(2))==int(spills.group(3))==0
 print(first,'INITIAL',registers.group(1) if registers else None,'POOL_OK',enough,'NO_SPILL',clean,flush=True)
 good &= enough and clean
good &= found==2 and not any(code in s for code in ('C7512','C7514','C7515'))
sys.exit(0 if good else 2)

"""Experimental initial CTA register pool metadata; machine code is unchanged.

Only accepts a single kernel with explicit dynamic allocation. A conservative
control-flow traversal checks every register operand before changing the initial
allocation. Original cubins are immutable and both hashes are recorded.
"""
from pathlib import Path
import hashlib,json,os,re,struct,subprocess


def initial_pool_modes(cubin,kernel,producer=96,consumer=160):
    initial=128
    assert producer+consumer==256
    cubin=Path(cubin)
    sass=subprocess.check_output([os.environ['CUDA_HOME']+'/bin/cuobjdump','--dump-sass',str(cubin)],text=True)
    assert sass.count('Function : ')==1 and 'Function : '+kernel in sass
    inst={}
    for line in sass.splitlines():
        m=re.search(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',line)
        if m:inst[int(m[1],16)]=m[2]
    assert inst and sum('TRY_ALLOC.CTAPOOL' in x for x in inst.values())>=1
    assert sum('DEALLOC.CTAPOOL' in x for x in inst.values())>=1
    queue=[(0,initial)];seen=set();caps={}
    while queue:
        pc,cap=queue.pop()
        if (pc,cap) in seen:continue
        seen.add((pc,cap));op=inst[pc];caps.setdefault(cap,0)
        regs=[int(v) for v in re.findall(r'\bR(\d+)\b',op)]
        assert all(v<cap for v in regs),(hex(pc),cap,op)
        caps[cap]=max(caps[cap],max(regs,default=0))
        if 'USETMAXREG' in op:cap=int(re.search(r'0x([0-9a-f]+)$',op)[1],16)
        assert not any(s in op for s in ('BRX','JMX','CALL','RET ')),op
        branch=re.search(r'\bBRA\s+0x([0-9a-f]+)',op)
        if branch:
            queue.append((int(branch[1],16),cap))
            if not op.startswith('@'):continue
        if 'EXIT' in op and not op.startswith('@'):continue
        if pc+16 in inst:queue.append((pc+16,cap))
    assert producer in caps and consumer in caps,caps
    assert initial*256>=128*(producer+consumer)
    data=bytearray(cubin.read_bytes());original_hash=hashlib.sha256(data).hexdigest()
    h=struct.unpack_from('<16sHHIQQQIHHHHHH',data)
    assert h[0][:6]==b'\x7fELF\x02\x01'
    sections=[struct.unpack_from('<IIQQQQIIQQ',data,h[6]+i*h[11]) for i in range(h[12])]
    st=sections[h[13]];names=data[st[4]:st[4]+st[5]]
    changed=[]
    for s in sections:
        name=bytes(names[s[0]:]).split(b'\0',1)[0].decode()
        if name!='.nv.info':continue
        start,end=s[4],s[4]+s[5]
        while start<end:
            fmt,kind,size=struct.unpack_from('<BBH',data,start)
            assert fmt==4,(fmt,kind,size)
            if kind==0x2f:
                assert size==8
                symbol,old=struct.unpack_from('<II',data,start+4)
                assert old==consumer,(old,consumer)
                struct.pack_into('<I',data,start+8,initial)
                changed.append(dict(offset=start+8,symbol=symbol,old=old,new=initial))
            start+=4+size
    assert len(changed)==1,changed
    digest=hashlib.sha256(data).hexdigest()
    out=cubin.with_name(digest+'.initial-pool.cubin');out.write_bytes(data)
    record=dict(original=str(cubin),original_sha256=original_hash,patched=str(out),patched_sha256=digest,changes=changed,register_bounds=caps,reachable_states=len(seen),machine_code_unchanged=True)
    out.with_suffix('.json').write_text(json.dumps(record,indent=2))
    return out,record

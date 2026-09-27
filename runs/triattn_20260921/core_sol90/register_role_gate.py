"""Check emitted register repartition for this 512-thread, four-WG family."""
import re


def emitted_register_roles(sass):
    return dict(
        dealloc_values=sorted(set(int(v,16) for v in re.findall(r'USETMAXREG\.DEALLOC\.CTAPOOL\s+0x([0-9a-f]+)',sass))),
        alloc_values=sorted(set(int(v,16) for v in re.findall(r'USETMAXREG\.TRY_ALLOC\.CTAPOOL\s+UP\d+,\s*0x([0-9a-f]+)',sass))))


def check_register_roles(sass, roles=None, initial_registers=128):
    if roles is None:
        roles=dict(producer=32,consumers=[160,160,160])
    emitted=emitted_register_roles(sass)
    if len(roles['consumers'])!=3:
        raise ValueError('This gate requires three complete consumer warpgroups')
    if emitted['dealloc_values']!=[roles['producer']] or emitted['alloc_values']!=sorted(set(roles['consumers'])):
        raise ValueError(('Emitted register roles differ',emitted,roles))
    pool=128*(roles['producer']+sum(roles['consumers']))
    if pool>512*initial_registers or pool>65536:
        raise ValueError(('Register pool overflow',pool,512*initial_registers))
    return dict(register_roles=roles,role_register_pool=pool,emitted=emitted)

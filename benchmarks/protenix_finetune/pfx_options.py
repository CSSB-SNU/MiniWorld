"""Options of the Protenix fine-tuning harness (formerly ``PFX_*`` environment variables).

Entry scripts call ``pfx_options.parse(sys.argv)`` before importing anything else of the harness; the other modules read
``pfx_options.OPTIONS``. A flag is ``--name`` (or ``--no-name`` for the ones that are on by default); a value is ``--name=value``.
The paths of the Protenix checkout and its library variables (``PROTENIX_SRC``, ``PROTENIX_CKPT``, ``PROTENIX_EXTRA_SITE``,
``LAYERNORM_TYPE``) stay environment variables: the Protenix code reads them.
"""
import dataclasses


@dataclasses.dataclass
class Options:
    det: bool = False
    g4_time: bool = True
    no_rezero: bool = False
    dit_bf16: bool = False
    dit_ckpt: bool = False
    dit_fused_infer: bool = True
    pair_stack: bool = True
    template: bool = True
    msa_stack: bool = True
    emb_atom: bool = False
    fast_t: bool = True
    template_batch: bool = True
    ckpt: bool = False
    graph_safe: bool = True
    loss_slddt_cuda: bool = True
    diff_only: bool = False
    full: bool = False
    freeze_trunk: bool = False
    fp32_master: bool = True
    fused_opt: bool = False
    side_conf: bool = False
    side_nojoin: bool = False
    conf_late: bool = False
    no_rollout: bool = False
    loss_gs: bool = False
    loss_compile: bool = False
    loss_surrogate: bool = False
    no_opt: bool = False
    diff_cache: bool = False
    stages: bool = False
    feats: str = ''
    msa_depth: str = ''
    extra: str = ''
    trainer: str = 'protenix'
    compile: str = ''
    capture: str = ''
    patch: str = ''
    input_json: str = ''
    triangle_attention: str = 'cuequivariance'
    triangle_multiplicative: str = 'cuequivariance'
    nrep: int = 4
    ncyc: int = 4
    nsamp: int = 48
    g4_stress: int = 0


OPTIONS = Options()


def parse(argv):
    """Consume the recognised ``--option`` arguments of ``argv`` into ``OPTIONS``; return the remaining arguments."""
    known = {f.name: f for f in dataclasses.fields(Options)}
    rest = [argv[0]]
    for arg in argv[1:]:
        if not arg.startswith("--"):
            rest.append(arg)
            continue
        key, eq, value = arg[2:].partition("=")
        negate = key.startswith("no-") and key[3:].replace("-", "_") in known and not eq
        name = (key[3:] if negate else key).replace("-", "_")
        if name not in known:
            rest.append(arg)
            continue
        kind = known[name].type
        if kind in ("bool", bool):
            setattr(OPTIONS, name, (not negate) if not eq else value.lower() in ("1", "true", "yes", "on"))
        elif kind in ("int", int):
            setattr(OPTIONS, name, int(value))
        else:
            setattr(OPTIONS, name, value)
    return rest

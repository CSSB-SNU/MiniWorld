"""Attach the bitwise-qualified pilot's fixed gate epilogue and DW layout."""
from pathlib import Path
import json
import torch
from prefix_gate_epi import PrefixGateEpi
from lt_contract import LtBmm

def attach(plan):
    p=plan.p;d=p.D
    plan.prefix_gate=PrefixGateEpi(plan)
    plan.prefix_gate_workspace=torch.empty(64*1024*1024,device=p.x.device,dtype=torch.uint8)
    op=LtBmm(plan.prefix_gate.prefix.unsqueeze(0),p.xn.reshape(p.M,d).unsqueeze(0),p.dwg.unsqueeze(0),plan.prefix_gate_workspace)
    choice=json.loads(Path(__file__).with_name('prefix_gate_choices.json').read_text())[f'{d}_{p.n}']
    op.index=choice['index'];assert list(op.heuristics[op.index].algo.data)==choice['algo']
    plan.prefix_gate_dw=op;plan.prefix_gate_choice=choice
    original_run=plan.schedule.run
    def run(name):
        if name=='dwg':op()
        else:original_run(name)
    plan.schedule.run=run;plan.dx.copy_prefix=lambda:None
    if d==256:plan.b1.prepare=plan.prefix_gate
    else:plan.b1.epi=plan.prefix_gate
    plan.artifacts.append(plan.prefix_gate.cubin)

def configuration(plan):
    return dict(layout='column_major_direct_dx_prefix',grid_factor=4,dwg=plan.prefix_gate_choice)

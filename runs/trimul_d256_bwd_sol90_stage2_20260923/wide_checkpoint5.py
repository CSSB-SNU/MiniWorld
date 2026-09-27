"""Qualified-candidate assembly: early mask load, D512 tile output, fixed GEMMs."""
from pathlib import Path
import json
from wide_checkpoint4 import Training as Previous,configuration as previous_configuration
from wide_mask_source import MaskSource
from wide_tile_output import TileOutput
from wide_lt_schedule import LtSchedule

class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        p=self.p
        self.mask_source=MaskSource(p,self.b7,use_offsets=True,prefetch=True,n128=False,mask_mode='prefetch')
        self.b7.source_only=self.mask_source
        self.artifacts.append(self.mask_source.cubin)
        if p.D==512:
            self.product_output=TileOutput(self.f,p,(self.b1.proj,self.b1.gate),16,128)
            self.artifacts.extend((self.product_output.cubin,self.product_output.epi_cubin))
        self.schedule=LtSchedule(self)
        choices=json.loads(Path(__file__).with_name('wide_checkpoint5_lt.json').read_text())[f'{p.D}-{p.n}']
        for name,choice in choices.items():
            op=self.schedule.kernels[name];op.index=choice['index']
            assert list(op.heuristics[op.index].algo.data)==choice['algo'],(name,choice)
            self.schedule.selected[name]=choice

    def forward(self,saved=True):
        if not saved:return super().forward(saved=False)
        return self.schedule.forward()

    def backward(self):return self.schedule.backward()


def configuration(plan):
    result=previous_configuration(plan)
    result['source']='prefetch_offset_n64_mask_prefetch'
    if plan.p.D==512:result['output']=['tile',16,128]
    result['lt']=plan.schedule.selected
    return result

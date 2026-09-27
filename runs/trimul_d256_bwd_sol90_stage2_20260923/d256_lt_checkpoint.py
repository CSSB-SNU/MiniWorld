"""D256 prefix checkpoint with frozen bitwise-preserving Lt GEMMs."""
from pathlib import Path
import json
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from d256_prefix_checkpoint import Training as Previous,configuration as previous_configuration,configure
from d256_lt_schedule import LtSchedule

class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        self.schedule=LtSchedule(self)
        choices=json.loads(Path(__file__).with_name('d256_lt_checkpoint_choices.json').read_text())[str(self.p.n)]
        for name,choice in choices.items():
            op=self.schedule.kernels[name];op.index=choice['index']
            assert list(op.heuristics[op.index].algo.data)==choice['algo'],name
            self.schedule.selected[name]=choice
        self.metadata['lt']=choices

    def forward(self):
        self._guard()
        with T.native_context(self.p.x.device):return self.schedule.forward()

    def backward(self):
        self._guard()
        if not self._has_forward:raise RuntimeError('Run forward before backward')
        with T.native_context(self.p.x.device):return self.schedule.backward()

def configuration(plan):
    result=previous_configuration(plan);result['lt']=plan.schedule.selected;return result

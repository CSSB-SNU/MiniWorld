"""Wide candidate with shared LN gamma and D512 TMA preactivation saves."""
from pathlib import Path
import json
from wide_checkpoint5 import Training as Previous,configuration as previous_configuration
from wide_cached_input import CachedInput
from wide_cached_ln import CachedLN
from wide_saved_front_tma import SavedFrontTma
from wide_saved_split_source import SavedSplitSource

class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        p=self.p;self.baseline_front=self.f.front;self.saved_source=None
        splits=32
        if p.D==512:
            choice=json.loads(Path(__file__).with_name('wide_checkpoint6_dw.json').read_text())[str(p.n)]
            splits=choice['splits']
            self.pre=p.x.new_empty((p.D//8,p.M,64))
            self.f.front=SavedFrontTma(self.baseline_front,self.pre)
            self.saved_source=SavedSplitSource(p,self.pre,self.f.mask,splits)
            op=self.saved_source.matmul;op.index=choice['index']
            assert list(op.heuristics[op.index].algo.data)==choice['algo']
            self.b7.source_only=self.saved_source
            self.artifacts.extend((self.f.front.cubin,self.saved_source.cubin))
        self.input_tma=CachedInput(p,16,128,4,splits=splits)
        self.dx.reduce_only=self.input_tma
        self.ln=CachedLN(p,32 if p.D==384 else 16,True,False,2)
        self.artifacts.extend((self.input_tma.cubin,self.ln.cubin))
        self.input_splits=splits

    def forward(self,saved=True):
        if saved:return super().forward()
        current=self.f.front
        try:
            self.f.front=self.baseline_front
            return super().forward(saved=False)
        finally:self.f.front=current


def configuration(plan):
    result=previous_configuration(plan)
    result['ln_gamma']='shared'
    result['input_splits']=plan.input_splits
    if plan.saved_source is not None:
        op=plan.saved_source.matmul
        result['source']='saved_preactivation_split_lt'
        result['front']='preactivation_tma_existing_staging'
        result['saved_bytes']=plan.pre.numel()*2
        result['source_lt']=dict(index=op.index,algo=list(op.heuristics[op.index].algo.data))
    return result

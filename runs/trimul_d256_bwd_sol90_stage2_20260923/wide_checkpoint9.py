"""Overlap forward saves and emit output-gate gradient into the dX prefix."""
from wide_checkpoint8 import Training as Previous,configuration as previous_configuration
from wide_saved_front_overlap import SavedOverlapFront
import prefix_gate_checkpoint as gate

class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        self.f.front=SavedOverlapFront(self.baseline_front,self.pre)
        self.artifacts.append(self.f.front.cubin)
        gate.attach(self)

def configuration(plan):
    result=previous_configuration(plan)
    result.update(front='overlapped_channel_major_preactivation_tma',output_gate=gate.configuration(plan))
    return result

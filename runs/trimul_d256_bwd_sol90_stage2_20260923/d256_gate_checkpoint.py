"""Frozen D256 Lt checkpoint with direct output-gate dX prefix publication."""
from d256_lt_checkpoint import Training as Previous,configuration as previous_configuration,configure
import prefix_gate_checkpoint as gate

class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        gate.attach(self)
        self.metadata['output_gate']=gate.configuration(self)

def configuration(plan):
    result=previous_configuration(plan);result['output_gate']=gate.configuration(plan);return result

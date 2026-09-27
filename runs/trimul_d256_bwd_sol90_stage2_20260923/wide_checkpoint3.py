"""Explicit candidate: prefetched source plus D512 CTA input-LN aggregation."""
from wide_checkpoint2 import Training as Previous
from wide_pipe_source import PipeSource
from wide_input_reduce import InputReduce


class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        self.pipe=PipeSource(self.p,self.b7,use_offsets=True,prefetch=True,n128=False)
        self.b7.source_only=self.pipe
        self.artifacts.append(self.pipe.cubin)
        self.input_reduce=None
        if self.p.D==512:
            self.input_reduce=InputReduce(self.p,256,3)
            self.dx.reduce_only=self.input_reduce
            self.artifacts.append(self.input_reduce.cubin)

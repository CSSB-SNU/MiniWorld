"""Keep source and output dW on disjoint SM sets during their fork."""
import torch
from d256_late_output_dw import LateOutputDW
from green_sm_partition import GreenPartition

class PartitionedOutputDW(LateOutputDW):
    def __init__(self,plan,small_sms=16,split_flags=0):
        super().__init__(plan,'source',False)
        self.partition=GreenPartition(small_sms,split_flags)
        self.side,self.compute=self.partition.streams
        self.compute_done=torch.cuda.Event()

    def fork(self,main_op):
        assert not self.launched
        self.launched=True;main=torch.cuda.current_stream();self.ready.record(main)
        with torch.cuda.stream(self.side):
            self.side.wait_event(self.ready)
            self.dwp();self.oldrun('dwg');self.done.record(self.side)
        with torch.cuda.stream(self.compute):
            self.compute.wait_event(self.ready);main_op();self.compute_done.record(self.compute)
        main.wait_event(self.compute_done)

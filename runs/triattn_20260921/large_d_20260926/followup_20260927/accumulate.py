import torch
from build import extension
def dgrad(gs,ws,epilogue='accumulate'):
    # Keep vendor GEMM throughput while eliminating separate BF16 dX tensors/adds.
    dx=torch.mm(gs[0],ws[0])
    for i in (1,2,3):torch.addmm(dx,gs[i],ws[i],beta=1,out=dx)
    return extension(epilogue,512).accumulate(dx,gs[4],ws[4])

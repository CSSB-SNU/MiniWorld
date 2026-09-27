"""Reorder independent output-weight gradients without changing any arithmetic."""

class ReorderedBackward:
    def __init__(self,plan,mode):
        assert mode in ('dw_first','ln_first','dw_after_source','dw_last')
        self.plan=plan;self.mode=mode

    def weights(self):
        p=self.plan
        if getattr(p,'split_dwp',None) is not None:p.split_dwp()
        else:p.schedule.run('dwp')
        p.schedule.run('dwg')

    def __call__(self):
        p=self.plan;b=p.b1;s=p.schedule;mode=self.mode
        if p.p.D==256:b.prepare()
        else:
            if getattr(p,'delta_output',None) is not None:
                p.delta_backward.launch();p.delta_projection()
            b.epi.launch((1056,1,1),(256,1,1),[b.ep],0)
        if mode=='dw_first':self.weights()
        s.run('dn')
        (b.ln if p.p.D==256 else p.ln)()
        if mode=='ln_first':self.weights()
        for name in ('bc0','bc1','bc2','bc3'):s.run(name)
        p.b7.source_only()
        if mode=='dw_after_source':self.weights()
        if p.p.D==256:p.dx()
        else:
            p.dx.copy_prefix();p.dx.pack_weights();s.run('dx');p.dx.reduce_only()
        # Projection partials occupy [0,2*D*D) in each 11*D*D split;
        # gate/input partials occupy the disjoint suffix. The projection
        # operation performs its own reduction; input LN does not consume it.
        if mode=='dw_last':self.weights()
        return p.p.outputs

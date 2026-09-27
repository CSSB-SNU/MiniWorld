"""D256 saved-products checkpoint with an explicit dense forward output."""
import os
from selected_current import Training as Previous
from shared_candidate import attach
from wide_dense_output import DenseOutput


def configure():
    os.environ.update(SHARED_CANDIDATE='saved_products',SAVE_EPI_GRID='1056',SAVE_EPI_THREADS='256',
        GP_OFF='1',GP_WARP='1',DX_LN='1',DX_WARP='1',SAVE_NORM='1',DX_N256='0',DX_PIPE='0',DX_SPLIT='0',
        DX_IN_STATS='0',DX_GAMMA_CACHE='0',LN_THREADS='128',LN_MINBLOCKS='3',LN_AGG='1',LN_CACHE='0',
        LN_STATS_TMA='0',LN_STORE_READ='0',LN_GAMMA_SMEM='0',DN_SLIM='0')


class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        configure()
        super().__init__(leaves,mask,ds,dy)
        self.metadata=attach(self)
        products=self.b1.prepare.original
        self.f.output=DenseOutput(self.f,self.p,(products.proj,products.gate),128)
        self.artifacts=[self.f.output.cubin,self.metadata['input_stats']['front_cubin'],
                        self.metadata['input_stats']['cubin'],self.b1.ln.cubin,
                        self.b7.source.unit.cubin_path,self.b1.prepare.k.unit.cubin_path]
        self.metadata['dense_output_cubin']=str(self.f.output.cubin)


def configuration(plan):
    return dict(output=['dense',128,1],source='d256_saved_products',
                input_stats=True,input_gamma_cache=True,ln_threads=128,ln_minblocks=3,
                splits=plan.b7.splits)

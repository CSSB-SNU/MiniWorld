"""Two resident CTAs split dW columns; only the first publishes the common GP."""
def shard_columns(body):
    body=body.replace('CH=128*D','CH=64*D')
    body=body.replace('for(int c=0;c<D/64;++c)tma_load_2d(sm+slot*INPUT+c*8192,&p.xn,bar+slot,c*64,row);',
                      'for(int c=0;c<D/128;++c)tma_load_2d(sm+slot*INPUT+c*8192,&p.xn,bar+slot,c*64+(blockIdx.x%2)*(D/2),row);')
    old='int rank=blockIdx.x%RANKS,split=blockIdx.x/RANKS,tiles=p.M/64;'
    assert body.count(old)==2
    body=body.replace(old,'int rank=(blockIdx.x/2)%RANKS,split=blockIdx.x/(2*RANKS),tiles=p.M/64;')
    body=body.replace('(WG*NC+n)*64+c;', '(blockIdx.x%2)*(D/2)+n*64+c;')
    old='''   store_gp(p.gmap+2*side,dg+4096,row,col);store_gp(p.gmap+2*side+1,dg,row,col);
   tma_store_commit();mbar_arrive(bar+3+slot);tma_store_wait_all();'''
    new='''   if(blockIdx.x%2==0){store_gp(p.gmap+2*side,dg+4096,row,col);store_gp(p.gmap+2*side+1,dg,row,col);tma_store_commit();}
   mbar_arrive(bar+3+slot);tma_store_wait_all();'''
    assert body.count(old)==1;body=body.replace(old,new)
    body=body.replace('__launch_bounds__(384,1)','__launch_bounds__(256,2)')
    body=body.replace('mbar_init(bar+b,b>=5?2:1)','mbar_init(bar+b,1)')
    body=body.replace('setmaxnreg_inc<200>();if(threadIdx.x<256)pipe_consume<0>(p,sm,bar);else pipe_consume<1>(p,sm,bar);',
                      'setmaxnreg_inc<160>();pipe_consume<0>(p,sm,bar);')
    return body.replace('mw_wide_saved_pipe','mw_wide_saved_shard')

"""Four M128 consumers plus one independent, nonblocking TMA producer warp.

REJECTED14800: naive 544*120=65280 ignores subpartition allocation. ptxas
reduces the limit to96, with C7512 and heavy spills. cuda_occupancy.h rounds
17 warps to20 for the assumed-CTA register limit; at120 this is76800 registers.
Do not relaunch or treat the initial120-register hypothesis as a valid budget.
No partial warpgroup executes setmaxnreg. Producer polls both queues; it never
blocks on one while a consumer needs the other queue to make progress.
"""


def transform(s):
    def replace(old, new, count=1):
        nonlocal s
        assert s.count(old) == count, (old, s.count(old), count)
        s = s.replace(old, new)

    replace('// L768 persistent CTA: one K/V fill, six query tiles, two consumer warpgroups.',
            '// L768: four M128 consumers and a separate single-warp TMA producer.')
    replace('Consumers=512,Threads=512', 'Consumers=512,Threads=544')
    replace('int L;float scale; };', 'int L;float scale;int zero; };')
    # Consumers occupy complete physical warpgroups 0..3. The final warp is
    # ordinary CUDA code with no WGMMA or dynamic register repartition.
    start=s.index(' if(tid==0){\n  s.qr.arrive_and_expect_tx')
    end=s.index(' int wg=',start)
    old=s[start:end]
    qload=old[:old.index('\n\n  load_kv(0)')].replace(' if(tid==0){','  if(tid==Consumers){',1)
    s=s[:start]+''' if(tid>=Consumers){
'''+qload+'''
   int kt=0,seq=0;
   #pragma unroll 1
   while(kt<6 || seq<48){
    if(kt<6 && (kt<Stages || s.empty[kt%Stages].test_wait(((kt/Stages)-1)&1))){
     load_kv(kt);++kt;
    }
    if(seq<48 && (seq<8 || s.be[seq&7].test_wait(((seq/8)-1)&1))){
     load_bias(seq);++seq;
    }
   }
  }
  return;
 }
'''+s[end:]
    replace('    if(tid==0 && seq/8+2<6)load_kv(seq/8+2);\n','')
    replace('    if(tid==0 && seq+8<48)load_bias(seq+8);\n','',2)
    replace('auto init=[&](auto& a,int seq)',
            'auto init=[&](auto& a,int seq,uint32_t packed_dep=0)')
    replace('s.bias[seq&7]+u*512+t*4',
            's.bias[seq&7]+u*512+t*4+packed_dep')
    replace('''    pack(old,previous);warpgroup_fence_operand(previous);
    init(old,seq+2);issue_qk(old,seq+2,hc);''','''    pack(old,previous);warpgroup_fence_operand(previous);
    auto packed_words=recast<uint32_t>(previous);uint32_t dependency=0;
    #pragma unroll
    for(int j=0;j<8;++j)dependency|=packed_words(j);
    dependency &= uint32_t(p.zero);
    // Forces the old score to die before the new bias loads. P's last use
    // retired at wait2 above; these reads precede its new PV issue below.
    init(old,seq+2,dependency);issue_qk(old,seq+2,hc);''')
    replace('valid.data_ptr<int>(),L,float(scale)};',
            'valid.data_ptr<int>(),L,float(scale),int(q.size(-2)>>40)};')
    replace(' attention<false><<<ct,Threads,sizeof(Shared),stream>>>(p);',''' int active=0;cudaFuncAttributes resource{};
 C10_CUDA_CHECK(cudaFuncGetAttributes(&resource,attention<false>));
 TORCH_CHECK(resource.numRegs<=120 && resource.localSizeBytes==0,"compact producer resource gate");
 C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,attention<false>,Threads,sizeof(Shared)));
 TORCH_CHECK(active>=1,"compact producer must have a resident CTA");
 attention<false><<<ct,Threads,sizeof(Shared),stream>>>(p);''')
    assert 'setmaxnreg' not in s
    return s

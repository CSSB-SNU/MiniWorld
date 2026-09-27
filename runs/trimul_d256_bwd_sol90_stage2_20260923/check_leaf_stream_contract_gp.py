from pathlib import Path
source=Path(__file__).with_name('check_stream_contract_gp.py').read_text()
source=source.replace('from wide_stream_contract_gp import StreamContractGP','from wide_leaf_stream_contract_gp import LeafStreamContractGP')
source=source.replace("groups=int(os.environ.get('GP_GROUPS','2'));read_credit=os.environ.get('GP_READ_CREDIT','0')=='1';op=StreamContractGP(plan,groups=groups,read_credit=read_credit)","groups=2;read_credit=True;op=LeafStreamContractGP(plan,minblocks=int(os.environ.get('GP_MINBLOCKS','2')))")
source=source.replace('for grid in (132,264):','for grid in (264,528):')
source=source.replace('local_bytes=op.local_bytes)','local_bytes=op.local_bytes,occupancy=op.occupancy)')
source=source.replace('result-stream-contract-gp-','result-leaf-stream-contract-gp-')
exec(compile(source,__file__,'exec'))

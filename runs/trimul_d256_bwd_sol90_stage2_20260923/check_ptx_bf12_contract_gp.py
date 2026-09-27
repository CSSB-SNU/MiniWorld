from pathlib import Path
source=Path(__file__).with_name('check_bf12_contract_gp.py').read_text()
source=source.replace('wide_checkpoint14','wide_checkpoint15')
source=source.replace('from wide_bf12_contract_gp import PackedPre,PackedContractGP','from wide_bf12_contract_gp import PackedPre\nfrom wide_ptx_bf12_contract_gp import PtxPackedContractGP as PackedContractGP')
source=source.replace('result-bf12-contract-gp-','result-ptx-bf12-contract-gp-')
source=source.replace('cubin=str(op.cubin))','cubin=str(op.cubin),registers=op.registers,local_bytes=op.local_bytes)\n    record["ptx_all_bf16_patterns"]=op.validate_decoder();assert record["ptx_all_bf16_patterns"]')
exec(compile(source,__file__,'exec'))

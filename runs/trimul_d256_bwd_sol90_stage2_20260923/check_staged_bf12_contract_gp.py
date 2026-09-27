from pathlib import Path
source=Path(__file__).with_name('check_bf12_contract_gp.py').read_text()
source=source.replace('from wide_bf12_contract_gp import PackedPre,PackedContractGP','from wide_bf12_contract_gp import PackedPre\nfrom wide_staged_bf12_contract_gp import StagedPackedContractGP as PackedContractGP')
source=source.replace('result-bf12-contract-gp-','result-staged-bf12-contract-gp-')
source=source.replace('cubin=str(op.cubin))','cubin=str(op.cubin),registers=op.registers,local_bytes=op.local_bytes)')
exec(compile(source,__file__,'exec'))

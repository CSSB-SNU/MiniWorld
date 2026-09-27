from pathlib import Path
source=Path(__file__).with_name('check_bf12_saved_front.py').read_text()
source=source.replace('from wide_bf12_contract_gp import PackedPre,PackedContractGP','from wide_bf12_contract_gp import PackedPre\nfrom wide_staged_bf12_contract_gp import StagedPackedContractGP as PackedContractGP')
source=source.replace('result-bf12-contract-gp-','result-staged-bf12-contract-gp-')
source=source.replace('result-bf12-saved-front-','result-staged-bf12-saved-front-')
exec(compile(source,__file__,'exec'))

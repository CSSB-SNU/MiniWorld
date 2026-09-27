from pathlib import Path
import json,subprocess
root=Path(__file__).resolve().parent
for name in ('result-channel-worker-ln-D512-L768-18270.json','result-channel-worker-ln-D384-L768-18284.json'):
 result=json.loads((root/name).read_text())
 for c in result['candidates']:
  print(name,{k:c[k] for k in ('rows','registers','occupancy')},flush=True)
  subprocess.run(['cuobjdump','--dump-resource-usage',c['cubin']],check=True)

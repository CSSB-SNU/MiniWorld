from pathlib import Path
import json,re,time,subprocess,shutil,hashlib,datetime
R=Path(__file__).resolve().parent;L=R.parent
meta=json.loads((R/'provenance.json').read_text());log=L/'slurm-13228.log'
print('Waiting for completed epoch checkpoint >468, existing job stays live meanwhile',flush=True)
deadline=time.time()+2400
while time.time()<deadline:
 text=log.read_text(errors='replace')[-200000:]
 matches=re.findall(r'Saved checkpoint to .*?/checkpoints/last\.pt \(epoch=(\d+), step=(\d+)\)',text)
 if matches and int(matches[-1][0])>468:
  epoch,step=map(int,matches[-1]);break
 time.sleep(.25)
else:raise RuntimeError('No new completed checkpoint; original job was NOT stopped')
print('Boundary found',epoch,step,flush=True)
shutil.copy2(meta['checkpoint'],R/'resume-8gpu.pt')
# The success log is emitted only after torch.save completes; validate before submitting.
import torch
c=torch.load(R/'resume-8gpu.pt',map_location='cpu',weights_only=False)
assert (c['epoch'],c['global_step'])==(epoch,step)
for k in ['optimizer_state_dict','scheduler_state_dict','ema_state_dict']:assert k in c
record=dict(previous_job=16681,epoch=epoch,step=step,checkpoint_sha256=hashlib.sha256((R/'resume-8gpu.pt').read_bytes()).hexdigest(),wandb_id=meta['wandb_id'],switched_at=datetime.datetime.now().astimezone().isoformat())
(R/'switch8.json').write_text(json.dumps(record,indent=2))
out=subprocess.check_output(['sbatch','--parsable','--dependency=afterany:16681',str(R/'train8.sbatch')],text=True).strip()
record['new_job']=int(out.split(';')[0]);record['checkpoint_verified_before_stop']=True
(R/'switch8.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
phase1b=subprocess.check_output(['sbatch','--parsable',f"--dependency=afterok:{record['new_job']}",'--kill-on-invalid-dep=yes',str(R/'phase1b8.sbatch')],text=True).strip()
record['phase1b_job']=int(phase1b.split(';')[0])
(R/'switch8.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
# Only now stop old training: frozen checkpoint checked and successor already queued.
subprocess.run(['scancel','16681'],check=True)
record['old_cancel_requested']=True;(R/'switch8.json').write_text(json.dumps(record,indent=2))

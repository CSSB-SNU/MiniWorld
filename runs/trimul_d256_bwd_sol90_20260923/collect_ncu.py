from pathlib import Path
import os,subprocess,sys
R=Path(__file__).resolve().parent
env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'}
env['PYTHONNOUSERSITE']='1'
stem=R/'profile-L384'
cmd=['ncu','--profile-from-start','off','--section','SpeedOfLight','--metrics','gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active','--cache-control','none','--clock-control','none','--force-overwrite','-o',str(stem),'env','PYTHONPATH='+os.environ.get('PYTHONPATH',''),sys.executable,'-B',str(R/'check.py'),'--profile']
subprocess.run(cmd,env=env,check=True)
with stem.with_suffix('.csv').open('w') as f:subprocess.run(['ncu','--import',str(stem)+'.ncu-rep','--page','raw','--csv'],env=env,stdout=f,check=True)

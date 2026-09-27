"""Read the wide raw CSV schema emitted by this installed Nsight Compute version."""
import csv
import json
from pathlib import Path

r=Path(__file__).resolve().parent
records=[]
scale={'byte':1,'Kbyte':1e3,'Mbyte':1e6,'Gbyte':1e9}
for artifact,job in (('resident6','18580'),('hot4r','18580'),('hot6t','18592')):
    path=r/('ncu-%s-%s.csv'%(artifact,job))
    rows=list(csv.DictReader(path.open()))
    values=next(row for row in rows if row.get('ID')=='0');units=rows[0]
    keys=[k for k in values if k in ('dram__bytes_read.sum','dram__bytes_write.sum',
          'lts__t_sectors_op_read.sum','lts__t_sectors_op_write.sum','gpu__time_duration.sum',
          'sm__throughput.avg.pct_of_peak_sustained_elapsed','lts__throughput.avg.pct_of_peak_sustained_elapsed',
          'sm__warps_active.avg.pct_of_peak_sustained_active') or 'stalled' in k]
    record=dict(artifact=artifact,job=job,length=768,scope='native QKV+attention+gate core only',
                metrics={k:dict(value=float(values[k].replace(',','')),unit=units[k]) for k in keys})
    record['dram_total_bytes']=sum(float(values[k])*scale[units[k]] for k in ('dram__bytes_read.sum','dram__bytes_write.sum'))
    record['l2_read_write_bytes']=32*sum(float(values[k]) for k in ('lts__t_sectors_op_read.sum','lts__t_sectors_op_write.sum'))
    records.append(record)
    print(artifact,'HBM_MB',record['dram_total_bytes']/1e6,'L2_GB',record['l2_read_write_bytes']/1e9,
          'time',record['metrics']['gpu__time_duration.sum'])
(r/'algorithm-ncu.json').write_text(json.dumps(dict(records=records,
    method='5 warmups; cache-control none; clock-control none; 17 replay passes; use paired graphs for performance selection'),indent=2)+'\n')

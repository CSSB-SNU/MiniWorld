"""Preserve NCU CSV units: automatic display scales vary across reports."""
import csv

FIELDS=('ID','Kernel Name','gpu__time_duration.sum','dram__bytes_read.sum',
        'dram__bytes_write.sum','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active',
        'launch__registers_per_thread')

def read_rows(path):
    result=[]
    with path.open() as f:
        reader=csv.DictReader(f)
        units=next(reader)
        assert not units.get('ID'), 'Expected NCU units row before metrics'
        for row in reader:
            if not row.get('ID'):continue
            item={k:row[k] for k in FIELDS}
            item['units']={k:units[k] for k in FIELDS if units.get(k)}
            times={'ns':.001,'us':1,'ms':1000,'s':1e6,'nsecond':.001,'usecond':1,'msecond':1000,'second':1e6}
            sizes={'B':1e-6,'KB':.001,'MB':1,'GB':1000,'byte':1e-6,'Kbyte':.001,'Mbyte':1,'Gbyte':1000}
            scales={'gpu__time_duration.sum':('time_us',times),
                    'dram__bytes_read.sum':('dram_read_MB',sizes),
                    'dram__bytes_write.sum':('dram_write_MB',sizes)}
            for key,(dest,mapping) in scales.items():
                item[dest]=float(row[key].replace(',',''))*mapping[units[key]]
            result.append(item)
    return result

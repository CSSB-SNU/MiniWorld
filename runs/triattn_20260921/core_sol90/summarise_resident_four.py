from pathlib import Path
import hashlib,json,re,subprocess

HERE=Path(__file__).resolve().parent
installed=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast/triattn_broadcast.so'
expected='9365a4f35e018afed4adba21590ae2c1734bbccf98d1e53904ebfe2537b3e225'
assert hashlib.sha256(installed.read_bytes()).hexdigest()==expected
jobs={
 'resident4m64s3qr':14909,
 'resident4m64s3qrloop6':14911,'resident4m64s2qrloop8':14911,
 'resident4m64s3qrloop6dep':14915,'resident4m64s2qrloop8dep':14915,
 'resident4m64s3ssloop6dep':14920,'resident4m64s2ssloop8dep':14920,
 'resident4m64hybrid8':14924,'resident4m64hybrid3sp':14928,
}
rows=[]
for name,job in jobs.items():
 log=(HERE/('build-'+name+'.log')).read_text()
 assert 'BUILT '+name in log,name
 warnings=sorted(set(re.findall(r'C751[12457]|C7520',log)))
 resources=[]
 for symbol,stack,stores,loads,regs in re.findall(r'Function properties for (\S*attention\S*)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers',log):
  resources.append(dict(safe='ILb1' in symbol,stack=int(stack),stores=int(stores),loads=int(loads),initial_registers=int(regs)))
 assert len(resources)==2,(name,resources)
 clean=not warnings and all(x['stack']==x['stores']==x['loads']==0 and x['initial_registers']<=96 for x in resources)
 binary=HERE/('build_'+name)/('triattn_sol_'+name+'.so')
 sasstext=subprocess.check_output(['/usr/local/cuda-12.9/bin/cuobjdump','-sass',str(binary)]).decode()
 hot='Function : '+next(x for x in sasstext.split('Function : ') if 'attentionILb0' in x.splitlines()[0])
 (HERE/(name+'-hot.sass')).write_text(hot)
 ops={op:len(re.findall(r'\b'+re.escape(op)+r'\b',hot)) for op in ('STL','LDL','HGMMA','WARPGROUP.ARRIVE','WARPGROUP.DEPBAR','MUFU.EX2','STSM')}
 item=dict(name=name,job=job,installed=False,compiler_clean=clean,warnings=warnings,resources=resources,sass=ops,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
 bench=HERE/(name+'-L768.json')
 if clean:
  assert bench.exists(),name+' clean but no benchmark result yet'
  data=json.loads(bench.read_text())['results'];base=data['serving'];candidate=data[name]
  bh=next(v for k,v in base['kernels'].items() if 'Traits<1073741824>' in k)
  ch=next(v for k,v in candidate['kernels'].items() if 'attention<false>' in k)
  item.update(hot_us=ch,baseline_hot_us=bh,hot_ratio=ch/bh,core_us=candidate['core_us'],baseline_core_us=base['core_us'],core_ratio=candidate['core_us']/base['core_us'],core_bitwise=candidate['core_equal'],block_bitwise=candidate['block_delta_rms']==0.,fp64_rms=candidate['core_rms_fp64'],baseline_fp64_rms=base['core_rms_fp64'])
  item['disposition']='reject_latency' if ch>=bh else 'needs_paired_qualification'
 else:
  assert not bench.exists(),name+' rejected at compiler gate but benchmark exists'
  item['disposition']='reject_compiler'
 rows.append(item)
out=dict(installed_sha256=expected,installed_changed=False,sol90_achieved=False,experiments=rows)
(HERE/'resident-four-results.json').write_text(json.dumps(out,indent=2)+'\n')
for row in rows:print(row['name'],row['disposition'],row.get('hot_us'),row.get('baseline_hot_us'))

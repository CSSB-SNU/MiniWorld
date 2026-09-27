"""Small CPU-only report: measured existing Triton divided by our timings."""
from pathlib import Path
import json

root=Path(__file__).resolve().parent
rows=[]
for d in (128,256,384,512):
    for n in (384,768):
        if d==128:
            file=root/f'baseline-D128-D256-L{n}-{17324 if n==384 else 17323}.json'
            data=json.loads(file.read_text());assert data['complete']
            own=lambda scope:data['times'][f'D128_{scope}']['median_us']
            baseline=lambda scope:data['times'][f'Triton_D128_{scope}']['median_us']
        else:
            if d==256:
                files=[root/f'result-d256-gate-checkpoint-vs-triton-D256-L{n}-{18009 if n==384 else 17977}.json']
                for name in ('stats','permuted-ln','late-dw','contract','early-affine','spatial','pool'):
                    for candidate in root.glob(f'result-d256-{name}-checkpoint-vs-triton-D256-L{n}-*.json'):
                        result=json.loads(candidate.read_text())
                        if result.get('complete') and result.get('qualified'):files.append(candidate)
                file=max(files,key=lambda f:int(json.loads(f.read_text())['job']))
            else:
                files=[]
                for checkpoint in (14,15,16,17,18,19,20,21,23,24):
                    for candidate in root.glob(f'result-wide-checkpoint{checkpoint}-vs-triton-D{d}-L{n}-*.json'):
                        result=json.loads(candidate.read_text())
                        if result.get('complete') and result.get('qualified'):
                            files.append((checkpoint,int(result['job']),candidate))
                file=max(files)[2]
            data=json.loads(file.read_text());assert data['complete'] and data['qualified']
            own=lambda scope:data['times']['new_full' if scope=='full' else 'our_'+scope]['median_us']
            baseline=lambda scope:data['times']['triton_'+scope]['median_us']
        row=dict(D=d,L=n,source=file.name,job=data['job'],timings={})
        for scope in ('forward','backward','full'):
            a,b=baseline(scope),own(scope)
            row['timings'][scope]=dict(triton_us=a,ours_us=b,speedup=a/b,gap_to_1_5_us=max(0,b-a/1.5))
        row['target_met']=all(row['timings'][scope]['speedup']>=1.5 for scope in ('backward','full'))
        rows.append(row)
complete=all(row['target_met'] for row in rows if row['D']>=256)
report=dict(goal_complete=complete,criterion='Existing Triton / ours >= 1.5 for backward and directly measured F+B at all six D>=256 shapes',rows=rows)
(root/'qualified_speedups.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['기존 Triton 대비 검증된 비교 결과. 시간 단위는 ms이며, 배율은 기존 / 우리 커널이다. F+B는 직접 측정했다.','',
       '| D | L | 기존 BWD → 우리 BWD | BWD 배율 | 기존 F+B → 우리 F+B | F+B 배율 | 1.5배 충족 |',
       '|---:|---:|---:|---:|---:|---:|:---:|']
for row in rows:
    b,f=row['timings']['backward'],row['timings']['full']
    lines.append(f'| {row["D"]} | {row["L"]} | {b["triton_us"]/1000:.3f} → {b["ours_us"]/1000:.3f} | {b["speedup"]:.4f}× | {f["triton_us"]/1000:.3f} → {f["ours_us"]/1000:.3f} | {f["speedup"]:.4f}× | {"충족" if row["target_met"] else "미달"} |')
lines+=['', '전체 목표: '+('충족' if complete else '미달. 최적화 진행 중.'), '', 'D128은 기존 검증 결과를 비교 기준으로 포함했다. 엔진 dispatch는 변경하지 않았다.','']
lines += [f'- D{row["D"]} L{row["L"]}: [{row["source"]}]({row["source"]})' for row in rows]
(root/'qualified_speedups.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:14]))

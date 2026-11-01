# -*- coding: utf-8 -*-
"""Generate neutral mechanism-effect report from completed benchmark JSON."""
import json,sys,csv
from pathlib import Path
import numpy as np

def analyze(data):
    rows=[];contrasts=[]
    for domain in ('logic','math'):
        ds=data['datasets'][domain];raw=data[f'{domain}_raw']
        for mode,stat in raw.items():
            arr=np.asarray(stat['seed_by_item_correct'],dtype=int)
            tok=np.asarray(stat['seed_by_item_tokens'],dtype=float)
            for mechanism in ('control','boundary','transition','recovery'):
                ii=[j for j,x in enumerate(ds) if x['mechanism']==mechanism]
                if not ii:continue
                a=arr[:,ii];t=tok[:,ii]
                rows.append(dict(domain=domain,mechanism=mechanism,mode=mode,questions=len(ii),accuracy=round(float(a.mean())*100,2),all_seed_correct=round(float((a.sum(0)==a.shape[0]).mean())*100,2),variable=round(float(((a.sum(0)>0)&(a.sum(0)<a.shape[0])).mean())*100,2),all_wrong=round(float((a.sum(0)==0).mean())*100,2),tokens_per_correct=round(float(t.sum()/a.sum()),2) if a.sum() else None))
        for bmode in ('M8_Thermo_Full_PD','M3_Fixed_Standard','M5_Mirostat_v2','M1_Greedy','M2_Fixed_Low'):
            if bmode not in raw:continue
            a=np.asarray(raw['M9_Adaptive_Hybrid']['seed_by_item_correct'],dtype=int)
            b=np.asarray(raw[bmode]['seed_by_item_correct'],dtype=int)
            for group in ('all','control','boundary','transition','recovery'):
                ii=[j for j,x in enumerate(ds) if group=='all' or x['mechanism']==group]
                dif=a[:,ii]-b[:,ii]
                contrasts.append(dict(domain=domain,group=group,against=bmode,m9_minus_comparator_pp=round(float(dif.mean())*100,3),m9_exclusive_correct=int(((a[:,ii]==1)&(b[:,ii]==0)).sum()),comparator_exclusive_correct=int(((a[:,ii]==0)&(b[:,ii]==1)).sum())))
    return rows,contrasts

def main():
    src=Path(sys.argv[1] if len(sys.argv)>1 else 'qwen3b_v14_results_s200.json')
    data=json.loads(src.read_text(encoding='utf-8')); rows, contrasts=analyze(data)
    for suffix,records in [('mechanisms',rows),('contrasts',contrasts)]:
        out=src.with_name(src.stem+'_'+suffix+'.csv')
        with out.open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
        print('[Saved]',out)
    print('NOTE: Descriptive paired differences only. Categories not a randomized treatment; no causal claim.')
if __name__=='__main__':main()

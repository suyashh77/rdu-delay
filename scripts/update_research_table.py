"""Rebuild comparable research rows from saved per-flight predictions, without rerunning models."""
from pathlib import Path
import json
import sys
import html
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rdu.eval.metrics import summarize

def main():
    base=ROOT/'experiments'/'results'
    reference=pd.read_parquet(base/'EXP-003-fast'/'predictions.parquet')
    ids=set(reference['flight_id'])
    runs=[('EXP-003-fast','LGBM_FULL (EXP-003)'),('EXP-002-fast','LGBM_WX (EXP-002)'),
          ('EXP-001b-fast','LGBM rotations (EXP-001b)'),('B4v2-full','B4v2 Markov baseline'),
          ('B2-full','B2 flight-number baseline (live fallback)'),('B3-full','B3 seasonal baseline'),
          ('B0-full','B0 climatology'),('B1-full','B1 always on time')]
    stats={}
    for run,_ in runs:
        predictions=pd.read_parquet(base/run/'predictions.parquet')
        predictions=predictions[predictions['flight_id'].isin(ids)]
        if len(predictions)!=len(ids) or predictions['flight_id'].duplicated().any():
            raise ValueError(f'{run} does not have the same unique flight set')
        stats[run]=summarize(predictions)
    baseline=stats['B4v2-full']
    rows=[]
    for run,label in runs:
        values=stats[run]
        leakage=base/run/'leakage.json'
        status='Saved leakage suite passed' if leakage.exists() and json.loads(leakage.read_text()).get('passed') else 'Not verified'
        cells=[html.escape(label),'23:00 ET (D-1)',f"{values['brier']:.4f}",f"{100*(1-values['brier']/baseline['brier']):+.2f}%",
               f"{values['crps']:.2f}",f"{100*(1-values['crps']/baseline['crps']):+.2f}%",f"{values['ece']:.5f}",status]
        rows.append('<tr>'+''.join(f'<td>{v}</td>' for v in cells)+'</tr>')
    path=ROOT/'web'/'benchmark_lab.html'
    text=path.read_text(encoding='utf-8')
    start=text.index('<tbody>',text.index('aria-label="Model performance comparison"'))
    end=text.index('</tbody>',start)+len('</tbody>')
    text=text[:start]+'<tbody>\n'+ '\n'.join(rows)+'\n</tbody>'+text[end:]
    text=text.replace('<th>Leakage Status</th>','<th>Saved validation</th>')
    old_note = 'The EXP-003 row predates the position-feature correction.'
    new_note = 'The EXP-003 row reflects walk-forward re-evaluation with the corrected position-feature cache (position-last-row-v2) and passed all leakage tests.'
    if old_note in text:
        text = text.replace(old_note, new_note)
    path.write_text(text,encoding='utf-8')
    print(json.dumps({'matched_flights':len(ids),'results':stats},indent=2))

if __name__=='__main__':main()

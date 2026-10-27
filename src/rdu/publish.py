"""Publish auditable schedule-baseline forecasts and independently observed outcomes.

The research LGBM_FULL model needs historical feature stores with no live equivalent.
Until that integration passes acceptance, use the existing B2 flight-number baseline.
Never assign the research model's metrics or champion status to these forecasts.
"""
import hashlib
import json
import math
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import airportsdata
import numpy as np
import pandas as pd

from rdu.eval.metrics import summaries, score, summarize, CANCEL
from rdu.eval.walkforward import cutoff_utc
from rdu.models.baselines import B2FlightNumber, History, pooled_hist

ROOT = Path(__file__).resolve().parents[2]
ET = ZoneInfo('America/New_York')
MODEL_VERSION = 'B2-flight-number-v1'
HOLDOUT_START = date(2025, 7, 1)
PROB_KEYS = ['p_on_time','p_d15_44','p_d45_89','p_d90_plus','p_cancelled']
AIRPORTS = airportsdata.load('IATA')
CARRIERS = {'AA':'American Airlines','DL':'Delta Air Lines','UA':'United Airlines','WN':'Southwest Airlines',
            '9E':'Endeavor Air','YX':'Republic Airways','OO':'SkyWest Airlines','MQ':'Envoy Air',
            'OH':'PSA Airlines','B6':'JetBlue','F9':'Frontier Airlines','NK':'Spirit Airlines',
            'AS':'Alaska Airlines','G4':'Allegiant Air'}

def utc(value):
    if not value: return None
    t = datetime.fromisoformat(str(value).replace('Z','+00:00'))
    if t.tzinfo is None: raise ValueError('Timestamp must include timezone')
    return t.astimezone(timezone.utc)

def iso(t):
    return t.isoformat() if t is not None else None

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def atomic_json(path, payload, js=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(',',':'))
    if js: text = 'window.DEPARTURES_DATA = ' + text + ';\n'
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(text, encoding='utf-8')
    os.replace(temp, path)

def snapshot_runs(folder, endpoint, before=None):
    """Read timestamped pages as distinct runs; never merge incomplete runs silently."""
    groups, current = [], []
    pattern = re.compile(rf'^{endpoint}_(\d{{8}}T\d{{6}}\d*Z)_p(\d+)\.json$')
    for path in sorted(folder.glob(f'{endpoint}_*.json')):
        m = pattern.match(path.name)
        if not m: continue
        stamp = m[1][:-1]
        fmt = '%Y%m%dT%H%M%S%f' if len(stamp)>15 else '%Y%m%dT%H%M%S'
        retrieved = datetime.strptime(stamp, fmt).replace(tzinfo=timezone.utc)
        if before is not None and retrieved >= before: continue
        if int(m[2]) == 1 and current:
            groups.append(current); current=[]
        current.append((int(m[2]), retrieved, path))
    if current: groups.append(current)
    runs=[]
    for group in groups:
        records={}; complete=False; error=False; last=None
        for _, _, path in group:
            try:
                response=read_json(path)
                for record in response[endpoint]:
                    if record.get('fa_flight_id'): records[record['fa_flight_id']]=record
                last=response
            except (ValueError, KeyError, TypeError): error=True
        pages=[g[0] for g in group]
        complete=not error and pages==list(range(1,len(pages)+1)) and last is not None and not (last.get('links') or {}).get('next')
        runs.append({'records':records, 'complete':complete, 'retrieved_at':group[-1][1],
                     'files':[g[2].name for g in group]})
    return runs

def latest_snapshot(folder, endpoint, before=None):
    runs=snapshot_runs(folder, endpoint, before)
    return runs[-1] if runs else None

def outcome_fields(record, now, pred=None):
    """Zero delay estimates do not establish that a flight has left the gate."""
    cancelled=bool(record.get('cancelled'))
    actual=utc(record.get('actual_out'))
    scheduled=utc(record.get('scheduled_out'))
    if actual is not None and actual > now: actual=None
    delay=(actual-scheduled).total_seconds()/60 if actual and scheduled and not cancelled else None
    if cancelled: status,badge='Cancelled','act-cancel'
    elif delay is None: status,badge='Outcome pending','act-scheduled'
    elif delay<0: status,badge=f'Early ({abs(delay):.0f}m)','act-ontime'
    elif delay<15: status,badge=f'On time (+{delay:.0f}m)','act-ontime'
    else: status,badge=f'Delayed +{delay:.0f}m',('act-delay-high' if delay>=45 else 'act-delay-mod')
    scoreable = pred is not None and pred.get('forecast_status') in ('published','historical_backtest')
    q10,q90=(pred or {}).get('q10'),(pred or {}).get('q90')
    bounded=max(0,min(180,delay)) if delay is not None else None
    in80=(q10<=bounded<=q90) if scoreable and bounded is not None and q10 is not None and q90 is not None else None
    return {'act_dep':actual.astimezone(ET).strftime('%H:%M') if actual and not cancelled else '--:--',
            'act_dep_ampm':actual.astimezone(ET).strftime('%I:%M %p').lstrip('0') if actual and not cancelled else '--',
            'actual_out_utc':iso(actual) if not cancelled else None,
            'act_delay':round(delay,1) if delay is not None else None,
            'cancelled':cancelled,'diverted':bool(record.get('diverted')), 'outcome_known':cancelled or delay is not None,
            'act_status':status,'act_badge':badge,'in80':in80,
            'err_min':round(pred['exp_delay']-bounded,1) if scoreable and bounded is not None else None}

def daily_stats(flights):
    known=[f for f in flights if f.get('outcome_known')]
    flown=[f for f in known if not f['cancelled'] and f.get('act_delay') is not None]
    delayed=[f for f in flown if f['act_delay']>=15]
    scored=[f for f in flights if f.get('in80') is not None]
    predicted=[f for f in flights if f.get('p_delay15') is not None]
    def avg(seq): return round(sum(seq)/len(seq),1) if seq else None
    return {'total_flights':len(flights),'known_outcomes':len(known),'pending_outcomes':len(flights)-len(known),
            'on_time_flights':len(flown)-len(delayed),'delayed_flights':len(delayed),
            'cancelled_flights':sum(f['cancelled'] for f in known),
            'observed_delay_pct':round(100*len(delayed)/len(known),1) if known else None,
            'predicted_delay_pct':avg([100*f['p_delay15'] for f in predicted]),
            'avg_actual_delay_min':avg([max(0,min(180,f['act_delay'])) for f in flown]),
            'avg_predicted_delay_min':avg([f['exp_delay'] for f in predicted]),
            'interval_coverage_pct':avg([100*int(f['in80']) for f in scored]), 'scored_flights':len(scored)}

def risk(p, cancel):
    if cancel>.08: return 'Cancellation risk','badge-cancel'
    if p>=.28: return 'High delay risk','badge-high'
    if p>=.12: return 'Moderate risk','badge-mod'
    return 'Low delay risk','badge-low'

def live_records(snapshot, target):
    rows=[]
    for fid, record in snapshot['records'].items():
        scheduled=utc(record.get('scheduled_out'))
        if scheduled is None or scheduled.astimezone(ET).date()!=target: continue
        carrier=record.get('operator_iata') or record.get('operator')
        ident=record.get('ident_iata') or record.get('ident') or ''
        if not carrier:
            match=re.match(r'^([A-Z0-9]{2})(\d+)$',ident)
            carrier=match[1] if match else None
        # General aviation registrations must not exclude NK (Spirit) airline flights.
        if not carrier: continue
        number=record.get('flight_number')
        if number is None:
            match=re.search(r'(\d+)$',ident); number=match[1] if match else None
        if number is None or not str(number).isdigit(): continue
        dest_info=record.get('destination') or {}
        dest=dest_info.get('code_iata') or dest_info.get('code') or 'Unknown'
        local=scheduled.astimezone(ET)
        rows.append({'id':fid,'flight_no':f'{carrier} {number}','carrier':carrier,
                     'carrier_name':CARRIERS.get(carrier,carrier),'flight_number':int(number),
                     'tail':record.get('registration') or 'Unknown','gate':record.get('gate_origin') or '--',
                     'terminal':record.get('terminal_origin') or '--','dest':dest,
                     'dest_city':dest_info.get('city') or AIRPORTS.get(dest,{}).get('city',dest),
                     'dest_name':dest_info.get('name') or AIRPORTS.get(dest,{}).get('name',dest),
                     'sched_dep':local.strftime('%H:%M'),'sched_dep_ampm':local.strftime('%I:%M %p').lstrip('0'),
                     'scheduled_out_utc':iso(scheduled),'sched_arr':'--:--','sched_arr_ampm':'--',
                     'leg':None,'n_legs':None,'turn_min':None,'prev_origin':None,
                     'direction':'dep','forecast_date':target,'rdu_local_hour':local.hour,'is_live':True})
    return sorted(rows,key=lambda r:(r['scheduled_out_utc'],r['id']))

def explain_baseline(flights, model, P):
    """Exact sequential B2 pooling changes, not causal claims or SHAP values."""
    targets = pd.DataFrame(flights)
    targets['month'] = pd.to_datetime(targets['forecast_date']).dt.month
    stages = [pooled_hist(model.train, model.y, targets, model.levels[:i])
              for i in range(1, len(model.levels))] + [P]
    rates = np.stack([stage[:, 15:181].sum(1) for stage in stages], axis=1)
    groups = [model.train.groupby(cols).size().to_dict() for cols in model.levels[1:]]
    labels = ['Airline history', 'Seasonal airline history', 'Flight-number history']
    for i, f in enumerate(flights):
        changes = np.diff(rates[i])
        n_flight = int(groups[2].get(tuple(targets.iloc[i][c] for c in model.levels[3]), 0))
        n_seasonal = int(groups[1].get(tuple(targets.iloc[i][c] for c in model.levels[2]), 0))
        n_airline = int(groups[0].get(tuple(targets.iloc[i][c] for c in model.levels[1]), 0))
        counts = [n_airline, n_seasonal, n_flight]
        if n_flight == 0:
            driver = int(np.argmax(np.abs(changes[:2])))
        else:
            driver = int(np.argmax(np.abs(changes)))
        delta = float(changes[driver])
        f['risk_evidence'] = {
            'method': 'sequential_B2_pooling', 'rdu_delay_probability': float(rates[i, 0]),
            'stages': [{'factor': label, 'sample_count': count,
                        'delay_probability': float(rates[i, j+1]),
                        'change_percentage_points': float(changes[j]*100)}
                       for j, (label, count) in enumerate(zip(labels, counts))],
            'matching_flight_count': n_flight}
        if abs(delta) >= .005:
            sign = '+' if delta > 0 else ''
            if driver == 2:
                if n_flight < 20:
                    f['why_tag'] = f"Flight history ({sign}{delta*100:.1f} pp, n={n_flight})"
                else:
                    f['why_tag'] = f"Flight history ({sign}{delta*100:.1f} pp)"
            elif driver == 1:
                f['why_tag'] = f"Seasonal baseline ({sign}{delta*100:.1f} pp" + (", no flight history)" if n_flight == 0 else ")")
            elif driver == 0:
                f['why_tag'] = f"Airline baseline ({sign}{delta*100:.1f} pp" + (", no flight history)" if n_flight == 0 else ")")
            verb = 'raises' if delta > 0 else 'lowers'
            change_summary = (f"{labels[driver]} {verb} the delay estimate by {abs(delta)*100:.1f} percentage points "
                              f"at that pooling step (using {counts[driver]:,} historical departures).")
        else:
            f['why_tag'] = f"Similar to RDU baseline ({rates[i, 0]*100:.1f}% delay risk)"
            change_summary = f"Airline, seasonal, and flight-number patterns make little difference from the RDU airport baseline ({rates[i, 0]*100:.1f}% delay risk)."
        p_ot = f['p_on_time']
        p_del = f['p_delay15']
        p_c = f['p_cancelled']
        pct_ot = round(p_ot * 100, 1)
        pct_del = round(p_del * 100, 1)
        pct_c = round(p_c * 100, 1)
        prediction = (f"Estimated odds: {pct_ot:.1f}% under 15 minutes late, "
                      f"{pct_del:.1f}% delayed 15+ minutes, "
                      f"{pct_c:.1f}% cancelled.")
        outcomes = [('on-time (<15m)', pct_ot),
                    ('delayed 15+ minutes', pct_del),
                    ('cancelled', pct_c)]
        outcomes.sort(key=lambda x: x[1], reverse=True)
        top_name, top_pct = outcomes[0]
        second_name, second_pct = outcomes[1]

        if outcomes[0][1] == outcomes[1][1] == outcomes[2][1]:
            prediction += f" Note: All three outcomes are equally likely ({top_pct:.1f}% each)."
        elif top_pct == second_pct:
            prediction += f" Note: {top_name} and {second_name} are equally likely ({top_pct:.1f}% each)."
        elif abs(top_pct - second_pct) <= 1.0:
            prediction += f" Note: {top_name} ({top_pct:.1f}%) and {second_name} ({second_pct:.1f}%) are approximately equally likely."
        elif top_name == 'on-time (<15m)':
            if pct_del >= 28.0:
                prediction += f" Note: While the {pct_del:.1f}% delay risk is elevated, on-time departure remains the most likely single outcome ({pct_ot:.1f}%)."
            elif pct_c >= 8.0:
                prediction += f" Note: On-time departure is the most likely outcome ({pct_ot:.1f}%), though cancellation risk is elevated at {pct_c:.1f}%."
        elif top_name == 'delayed 15+ minutes':
            prediction += f" Note: A delay of 15+ minutes is the most likely single outcome ({pct_del:.1f}%), compared to {pct_ot:.1f}% on-time."
        else:
            prediction += f" Note: Cancellation is the most likely single outcome ({pct_c:.1f}%), compared to {pct_ot:.1f}% on-time."
        if n_flight == 0:
            evidence = ('No matching historical departures exist for this airline, flight number, and month (n=0). '
                        'The model relies on broader airline and seasonal patterns at RDU.')
        elif n_flight < 20:
            evidence = (f"{n_flight} historical departures match this airline, flight number, and month (sparse record, n<20). "
                        'Because this history is sparse, the model shrinks the estimate heavily toward broader airline and seasonal patterns.')
        else:
            evidence = (f"{n_flight} historical departures match this airline, flight number, and month. "
                        'Their outcomes are pooled with broader airline and seasonal patterns at RDU.')
        carrier_comp = (f"{f.get('carrier_name', f['carrier'])} departures at RDU average {rates[i, 1]*100:.1f}% delay "
                        f"({abs(changes[0])*100:.1f} pp {'above' if changes[0]>0 else 'below'} RDU airport average of {rates[i, 0]*100:.1f}%, n={n_airline:,}).")
        seasonal_comp = (f"In this month, {f.get('carrier_name', f['carrier'])} delay rate shifts to {rates[i, 2]*100:.1f}% "
                         f"({abs(changes[1])*100:.1f} pp {'higher' if changes[1]>0 else 'lower'} than annual airline baseline, n={n_seasonal:,}).")
        f['why_summary'] = change_summary + (f" ({n_flight} matching flight records)." if n_flight else " (No matching flight records).")
        f['why_detail'] = (f"{prediction} {change_summary} {carrier_comp} {seasonal_comp} {evidence} "
                           "These figures reflect sequential partial-pooling across model levels, not causal or operational factors. "
                           "Weather and live aircraft rotations are not used.")
        f['factor'] = f['why_detail']


def baseline_predictions(flights, training, cutoff):
    # Do not consume the locked holdout or reconstruct post-cutoff outcomes as training data.
    train=training[(training['forecast_date']<HOLDOUT_START)&(training['outcome_time_utc']<cutoff)&
                   (training['direction']=='dep')&(training['bucket']>=0)].copy()
    if train.empty: raise ValueError('No eligible baseline training history')
    model=B2FlightNumber(); hist=History(pd.DataFrame(),train); model.fit(hist)
    P=model.predict(pd.DataFrame(flights),pd.DataFrame(),hist)
    if not np.isfinite(P).all() or (P<0).any() or not np.allclose(P.sum(1),1):
        raise ValueError('Invalid probability distribution')
    summary=summaries(P).to_dict('records')
    months=pd.to_datetime(train['forecast_date']).dt.month
    sample_counts=train.assign(month=months).groupby(['carrier','flight_number','month']).size().to_dict()
    for f,p,distribution in zip(flights,summary,P):
        f['distribution'] = distribution.tolist()
        f.update({k:float(p[k]) for k in PROB_KEYS+['p_delay15']})
        f.update({q:int(p[q]) for q in ['q05','q10','q25','q50','q75','q90','q95']})
        f['exp_delay']=round(p['expected_delay_capped_min'],1)
        f['typical_if_delayed_min']=round(p['typical_if_delayed_min'],1)
        f['risk'],f['risk_badge']=risk(f['p_delay15'],f['p_cancelled'])
        matching_count=sample_counts.get((f['carrier'],f['flight_number'],f['forecast_date'].month),0)
        f['why_tag']='Historical flight-number baseline'
        f['why_detail']=f"B2 uses past outcomes for this airline and flight number in the same month ({matching_count} matching records), pooled toward airline and overall frequencies. Weather and aircraft rotations are not used."
        f['factor']=f['why_detail']; f['model_version']=MODEL_VERSION; f['champion_id']=None
        f['low_confidence_flags']=['schedule_only','weather_not_integrated','training_frozen_before_2025_07_holdout']+(['sparse_flight_history'] if matching_count<20 else [])
        f['training_end_date']=str(train['forecast_date'].max())
    explain_baseline(flights, model, P)
    return flights

def build_live_day(target, training, root=ROOT, now=None):
    realtime = now is None
    now=now or datetime.now(timezone.utc)
    cutoff=cutoff_utc(target).to_pydatetime()
    folder=root/'data'/'live'/'aeroapi'/str(target)
    archive=root/'data'/'forecasts'/f'{target}.json'
    # No input collected at or after the cutoff can become a night-before feature.
    snapshot=latest_snapshot(folder,'scheduled_departures', min(now,cutoff))
    latest=latest_snapshot(folder,'scheduled_departures',now)
    if archive.exists():
        frozen=read_json(archive)
        if frozen.get('model_version')!=MODEL_VERSION: raise ValueError('Unexpected archived model version')
        flights=frozen['flights']; metadata=frozen['metadata']
    elif snapshot:
        flights=live_records(snapshot,target)
        if not flights: return None
        flights=baseline_predictions(flights,training,cutoff)
        if realtime: now=datetime.now(timezone.utc)
        status='provisional' if now<cutoff else 'reconstructed'
        # A collection must be complete and fresh in the nightly window to be frozen.
        freeze_start=cutoff-timedelta(minutes=30)
        if freeze_start<=now<cutoff and snapshot['retrieved_at']>=freeze_start and snapshot['complete']:
            status='published'
        metadata={'model_version':MODEL_VERSION,'champion_id':None,'forecast_status':status,
                  'inputs_cutoff_utc':iso(cutoff),'schedule_retrieved_at_utc':iso(snapshot['retrieved_at']),
                  'schedule_complete':snapshot['complete'],'published_at_utc':iso(now) if status=='published' else None,
                  'generated_at_utc':iso(now),'input_files':snapshot['files'],
                  'training_end_date':flights[0]['training_end_date']}
        for f in flights:
            f.update({k:metadata[k] for k in ['forecast_status','inputs_cutoff_utc','published_at_utc']})
            f['forecast_date']=str(f['forecast_date'])
        if status=='published':
            archive.parent.mkdir(parents=True,exist_ok=True)
            # Exclusive creation guarantees a later process cannot revise an existing forecast.
            payload={'model_version':MODEL_VERSION,'metadata':metadata,'flights':flights}
            with archive.open('x',encoding='utf-8') as out: json.dump(payload,out,allow_nan=False)
    else:
        # Do not silently use a post-cutoff schedule to fabricate an earlier forecast.
        if not latest: return None
        flights=live_records(latest,target)
        metadata={'model_version':None,'forecast_status':'unavailable','inputs_cutoff_utc':iso(cutoff),
                  'schedule_retrieved_at_utc':iso(latest['retrieved_at']),'schedule_complete':latest['complete'],
                  'published_at_utc':None,'generated_at_utc':iso(now)}
        for f in flights:
            f.update({k:None for k in PROB_KEYS+['p_delay15','exp_delay','q10','q50','q90']})
            f.update(forecast_status='unavailable',risk='Forecast unavailable',risk_badge='badge-mod',
                     why_tag='Forecast unavailable',
                     why_summary='No eligible schedule was saved before the 11:00 PM cutoff; contributing factors unavailable.',
                     why_detail='No eligible schedule was saved before the 11:00 PM cutoff; individual contributing factors are unavailable.',
                     factor='No eligible schedule was saved before the 11:00 PM cutoff; individual contributing factors are unavailable.')
            f['forecast_date']=str(target)
    outcomes={}; outcome_time=None
    all_runs=snapshot_runs(folder,'scheduled_departures',now)+snapshot_runs(folder,'departures',now)
    for run in sorted(all_runs,key=lambda r:r['retrieved_at']):
        outcomes.update(run['records']); outcome_time=run['retrieved_at']
    for f in flights:
        record = {**outcomes.get(f['id'],{}), 'scheduled_out':f['scheduled_out_utc']}
        f.update(outcome_fields(record,now,f))
    metadata={**metadata,'outcomes_updated_at_utc':iso(outcome_time),
              'schedule_age_hours':round((now-utc(metadata['schedule_retrieved_at_utc'])).total_seconds()/3600,1)}
    return {'date':str(target),'label':target.strftime('%a, %b %d'),'weekday':target.strftime('%A'),
            'metadata':metadata,'stats':daily_stats(flights),'flights':flights}

def historical_weeks(training, root=ROOT):
    """Render saved B4v2 walk-forward predictions, with matching model explanations."""
    path=root/'experiments'/'results'/'B4v2-full'/'predictions.parquet'
    if not path.exists(): return []
    preds=pd.read_parquet(path).set_index('flight_id')
    configs=[('spring_2024','Spring baseline',date(2024,3,18)),('summer_2024','Summer operations',date(2024,6,3)),
             ('storms_2024','July operations',date(2024,7,15)),('winter_2023','Winter operations',date(2023,1,2))]
    data=training[training['direction']=='dep'].copy()
    data['flight_id']=data['leg_index'].astype(str)+data['direction']
    data=data[data['flight_id'].isin(preds.index)]
    weeks=[]
    for id,name,start in configs:
        days=[]
        for offset in range(7):
            target=start+timedelta(days=offset)
            flights=[]
            for _,r in data[data['forecast_date']==target].sort_values('sched_dep_utc').iterrows():
                p=preds.loc[r['flight_id']]
                carrier=str(r['carrier']); dest=str(r['dest']); airport=AIRPORTS.get(dest,{})
                sched=r['sched_dep_utc'].tz_convert(ET)
                f={'id':r['flight_id'],'flight_no':f"{carrier} {r['flight_number']}",'carrier':carrier,
                   'carrier_name':CARRIERS.get(carrier,carrier),'tail':str(r['tail']) if pd.notna(r['tail']) else 'Unknown',
                   'gate':'--','terminal':'--','dest':dest,'dest_city':airport.get('city',dest),'dest_name':airport.get('name',dest),
                   'sched_dep':sched.strftime('%H:%M'),'sched_dep_ampm':sched.strftime('%I:%M %p').lstrip('0'),
                   'sched_arr':'--:--','sched_arr_ampm':'--','is_live':False,'forecast_status':'historical_backtest',
                   'model_version':'B4v2-full','forecast_date':str(target),'exp_delay':round(float(p['expected_delay_capped_min']),1),
                   'why_tag':'B4v2 Markov baseline',
                   'why_summary':'B4v2 Markov baseline forecast; individual contributing factors were not saved.',
                   'why_detail':'Saved walk-forward B4v2 Markov forecast. The model propagates delay distributions through scheduled aircraft rotations. Individual contributing factor attributions were not saved for this historical run.',
                   'factor':'Saved walk-forward B4v2 Markov forecast. The model propagates delay distributions through scheduled aircraft rotations. Individual contributing factor attributions were not saved for this historical run.'}
                for k in PROB_KEYS+['p_delay15']: f[k]=float(p[k])
                for k in ['q10','q50','q90']: f[k]=int(p[k])
                for k,src in [('leg','leg_number'),('n_legs','n_legs_day'),('turn_min','sched_turn_min')]:
                    f[k]=int(r[src]) if pd.notna(r[src]) else None
                f['prev_origin']=str(r['prev_origin']) if pd.notna(r['prev_origin']) else None
                f['risk'],f['risk_badge']=risk(f['p_delay15'],f['p_cancelled'])
                f.update(outcome_fields({'scheduled_out':r['sched_dep_utc'].isoformat(),
                                         'actual_out':r['actual_dep_utc'].isoformat() if pd.notna(r['actual_dep_utc']) else None,
                                         'cancelled':bool(r['cancelled']),'diverted':bool(r['diverted'])},
                                        datetime.now(timezone.utc),f))
                flights.append(f)
            if flights:
                days.append({'date':str(target),'label':target.strftime('%a, %b %d'),'weekday':target.strftime('%A'),
                             'metadata':{'model_version':'B4v2-full','forecast_status':'historical_backtest'},
                             'stats':daily_stats(flights),'flights':flights})
        weeks.append({'id':id,'name':name,'date_range':f'{start} to {start+timedelta(days=6)}',
                      'description':'Historical B4v2 walk-forward forecasts.','is_live':False,'days':days})
    return weeks

def _publish(root=ROOT, now=None):
    realtime = now is None
    now=now or datetime.now(timezone.utc)
    today=now.astimezone(ET).date()
    clean_path = root / 'data' / 'clean' / 'rdu_flights.parquet'
    res_path = Path(__file__).resolve().parent / 'resources' / 'b2_training.parquet'
    if clean_path.exists():
        training = pd.read_parquet(clean_path)
    elif res_path.exists():
        training = pd.read_parquet(res_path)
    else:
        raise FileNotFoundError(f'Neither {clean_path} nor {res_path} exists')
    # Filter before any use of labels: the locked holdout is not used for training or evaluation here.
    training = training[training['forecast_date'] < HOLDOUT_START]
    days = [build_live_day(today + timedelta(days=i), training, root, None if realtime else now) for i in [-1, 0, 1]]
    days = [d for d in days if d]
    for day in days:
        scored = []
        for f in day['flights']:
            if f.get('forecast_status') != 'published' or not f.get('outcome_known') or 'distribution' not in f: continue
            y = CANCEL if f['cancelled'] else int(round(max(0, min(180, f['act_delay']))))
            bucket = 4 if f['cancelled'] else 0 if f['act_delay'] < 15 else 1 if f['act_delay'] < 45 else 2 if f['act_delay'] < 90 else 3
            result = score(np.array([f['distribution']]), np.array([y]), np.array([bucket]))
            scored.append(result)
        day['scores'] = summarize(pd.concat(scored, ignore_index=True)) if scored else None
        if day['scores']:
            atomic_json(root / 'data' / 'scores' / f"{day['date']}.json", {'date': day['date'], 'model_version': MODEL_VERSION,
                        'updated_at_utc': iso(now), 'pending_outcomes': day['stats']['pending_outcomes'], **day['scores']})
        # Full distributions remain in immutable archives, not the browser bundle.
        for f in day['flights']: f.pop('distribution', None)
    weeks = []
    if days:
        weeks.append({'id': 'live_week', 'name': 'Current departures', 'is_live': True,
                      'date_range': f"{days[0]['date']} to {days[-1]['date']}",
                      'description': 'Timestamped schedule snapshots with a B2 historical flight-number baseline.', 'days': days})
    hist = historical_weeks(training, root)
    if not hist and (root / 'web' / 'departures_data.json').exists():
        try:
            prev = read_json(root / 'web' / 'departures_data.json')
            hist = [w for w in prev.get('weeks', []) if not w.get('is_live')]
        except Exception:
            hist = []
    weeks += hist
    payload={'schema_version':2,'generated_at_utc':iso(now),'timezone':'America/New_York',
             'live_model':MODEL_VERSION,'live_model_status':'schedule_only_fallback',
             'limitations':['Weather and live aircraft rotations are not integrated into the live model.',
                            'Research LGBM_FULL results do not describe live forecast accuracy.'], 'weeks':weeks}
    for week in weeks:
        for day in week['days']:
            for f in day['flights']:
                if f.get('p_delay15') is not None:
                    assert math.isclose(sum(f[k] for k in PROB_KEYS),1,abs_tol=1e-6)
                    assert math.isclose(sum(f[k] for k in PROB_KEYS[1:4]),f['p_delay15'],abs_tol=1e-6)
    atomic_json(root/'web'/'departures_data.json',payload)
    atomic_json(root/'web'/'departures_data.js',payload,js=True)
    return {'generated_at_utc':iso(now),'model':MODEL_VERSION,'live_days':[
        {'date':d['date'],'flights':len(d['flights']),'status':d['metadata']['forecast_status'],
         'complete':d['metadata']['schedule_complete'],'known_outcomes':d['stats']['known_outcomes']} for d in days]}

def publish(root=ROOT, now=None):
    lock=root/'data'/'.publication.lock'
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('x') as handle:
        handle.write(str(os.getpid()))
    try:
        return _publish(root,now)
    finally:
        lock.unlink(missing_ok=True)

if __name__=='__main__':
    print(json.dumps(publish(),indent=2))

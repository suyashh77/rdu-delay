from datetime import date, datetime, timedelta, timezone
import json

import numpy as np
import pandas as pd
import pytest

from rdu import publish as P
from rdu.ingest import live

NOW=datetime(2026,10,26,15,tzinfo=timezone.utc)

def record(**kw):
    return {'fa_flight_id':'flight-1','ident_iata':'NK123','operator_iata':'NK','flight_number':'123',
            'scheduled_out':'2026-10-27T13:00:00Z','actual_out':None,'departure_delay':0,
            'destination':{'code_iata':'MCO'}, **kw}

def page(folder,stamp,records,number=1,next=None,endpoint='scheduled_departures'):
    folder.mkdir(parents=True,exist_ok=True)
    (folder/f'{endpoint}_{stamp}_p{number:02d}.json').write_text(json.dumps({endpoint:records,'links':{'next':next}}))

def train():
    return pd.DataFrame({'forecast_date':[date(2024,10,1),date(2024,10,2),date(2024,10,3),date(2025,8,1)],
                         'outcome_time_utc':pd.to_datetime(['2024-10-01T15:00Z','2024-10-02T15:00Z','2024-10-03T15:00Z','2025-08-01T15:00Z']),
                         'direction':['dep']*4,'bucket':[0,1,4,3],'y_capped':[0,25,180,180],
                         'carrier':['NK']*4,'flight_number':[123]*4})

def test_pending_is_neither_on_time_nor_scoreable():
    r=P.outcome_fields(record(),NOW,{'forecast_status':'published','q10':0,'q90':30,'exp_delay':10})
    assert r['act_delay'] is None and r['in80'] is None and not r['outcome_known']
    s=P.daily_stats([{**r,'p_delay15':.3,'exp_delay':10}])
    assert s['on_time_flights']==0 and s['observed_delay_pct'] is None
    assert s['interval_coverage_pct'] is None

def test_actual_gate_out_drives_delay_not_provider_estimate():
    r=P.outcome_fields(record(scheduled_out='2026-10-26T13:00:00Z',actual_out='2026-10-26T13:23:00Z',departure_delay=0),NOW)
    assert r['act_delay']==23 and r['act_dep']=='09:23' and r['outcome_known']

def test_future_actual_time_is_rejected():
    assert not P.outcome_fields(record(actual_out='2026-10-27T13:00:00Z'),NOW)['outcome_known']

def test_cancellation_not_zero_delay():
    r=P.outcome_fields(record(cancelled=True),NOW)
    assert r['cancelled'] and r['act_delay'] is None and r['act_status']=='Cancelled'
    s=P.daily_stats([{**r,'p_delay15':.3,'exp_delay':10}])
    assert s['cancelled_flights']==1 and s['on_time_flights']==0 and s['avg_actual_delay_min'] is None

def test_reconstruction_is_not_scored():
    r=P.outcome_fields(record(scheduled_out='2026-10-26T13:00:00Z',actual_out='2026-10-26T13:20:00Z'),NOW,
                       {'forecast_status':'reconstructed','q10':0,'q90':40,'exp_delay':10})
    assert r['outcome_known'] and r['in80'] is None and r['err_min'] is None

def test_early_departure_uses_capped_target_for_interval():
    r=P.outcome_fields(record(scheduled_out='2026-10-26T13:00:00Z',actual_out='2026-10-26T12:55:00Z'),NOW,
                       {'forecast_status':'published','q10':0,'q90':40,'exp_delay':10})
    assert r['act_delay']==-5 and r['in80'] and r['err_min']==10

def test_snapshots_keep_newest_run_and_enforce_cutoff(tmp_path):
    page(tmp_path,'20261026T120000Z',[record(gate_origin='A')])
    page(tmp_path,'20261026T130000Z',[record(gate_origin='B')],next='/aeroapi/next')
    page(tmp_path,'20261027T040000Z',[record(gate_origin='C')])
    snap=P.latest_snapshot(tmp_path,'scheduled_departures',datetime(2026,10,27,3,tzinfo=timezone.utc))
    assert snap['records']['flight-1']['gate_origin']=='B' and not snap['complete']

def test_missing_page_cannot_claim_complete(tmp_path):
    page(tmp_path,'20261026T120000Z',[record()],number=1,next='/aeroapi/next')
    page(tmp_path,'20261026T120020Z',[record()],number=3)
    assert not P.latest_snapshot(tmp_path,'scheduled_departures')['complete']

def test_spirit_airline_and_rdu_local_day_filter():
    snap={'records':{'a':record(),'b':record(fa_flight_id='b',scheduled_out='2026-10-27T02:00:00Z')}}
    rows=P.live_records(snap,date(2026,10,27))
    assert len(rows)==1 and rows[0]['carrier']=='NK'
    assert rows[0]['leg'] is None and rows[0]['terminal']=='--'

def test_baseline_is_coherent_and_excludes_holdout():
    rows=P.live_records({'records':{'a':record()}},date(2026,10,27))
    f=P.baseline_predictions(rows,train(),datetime(2026,10,27,3,tzinfo=timezone.utc))[0]
    assert np.isclose(sum(f[k] for k in P.PROB_KEYS),1)
    assert np.isclose(f['p_delay15'],sum(f[k] for k in P.PROB_KEYS[1:4]))
    assert f['q10']<=f['q50']<=f['q90']
    assert f['training_end_date']=='2024-10-03'
    assert f['champion_id'] is None

@pytest.mark.parametrize('target,hours',[(date(2026,3,8),23),(date(2026,11,1),25),(date(2026,10,27),24)])
def test_dst_day_window(target,hours):
    start,end=live.day_window(target)
    assert (end-start).total_seconds()==hours*3600

def test_pagination_rejects_other_host():
    with pytest.raises(ValueError): live.next_url('https://attacker.invalid/aeroapi/next')
    assert live.next_url('/aeroapi/next')=='https://aeroapi.flightaware.com/aeroapi/next'

def test_budget_skips_without_paid_request(tmp_path,monkeypatch):
    monkeypatch.setattr(live,'LIVE',tmp_path)
    monkeypatch.setattr(live,'_api_key',lambda:'not-a-real-key')
    monkeypatch.setattr(live,'month_spend',lambda key:4.49)
    monkeypatch.setattr(live,'_get',lambda *a,**k:pytest.fail('Should not issue paid request'))
    assert live.collect(date(2026,10,27),max_pages=10)['skipped']
    assert not (tmp_path/'aeroapi'/'.collection.lock').exists()

def test_failed_request_is_accounted_and_marked_incomplete(tmp_path,monkeypatch):
    monkeypatch.setattr(live,'LIVE',tmp_path)
    monkeypatch.setattr(live,'_api_key',lambda:'not-a-real-key')
    monkeypatch.setattr(live,'month_spend',lambda key:0.)
    def fail(*a,**k): raise TimeoutError()
    monkeypatch.setattr(live,'_get',fail)
    result=live.collect(date(2026,10,27),max_pages=1)
    assert not result['scheduled_departures']['complete']
    assert result['scheduled_departures']['error']=='TimeoutError'
    assert live._ledger_month_cost(datetime.now(timezone.utc).strftime('%Y-%m'))==.005

def test_frozen_forecast_survives_outcome_refresh(tmp_path):
    folder=tmp_path/'data'/'live'/'aeroapi'/'2026-10-27'
    page(folder,'20261027T024000Z',[record()])
    before=datetime(2026,10,27,2,45,tzinfo=timezone.utc)
    first=P.build_live_day(date(2026,10,27),train(),tmp_path,before)
    assert first['metadata']['forecast_status']=='published'
    path=tmp_path/'data'/'forecasts'/'2026-10-27.json'
    original=path.read_bytes()
    page(folder,'20261027T150000Z',[record(actual_out='2026-10-27T13:20:00Z')],endpoint='departures')
    later=P.build_live_day(date(2026,10,27),train(),tmp_path,datetime(2026,10,27,16,tzinfo=timezone.utc))
    assert later['flights'][0]['act_delay']==20
    assert path.read_bytes()==original
    assert later['flights'][0]['p_delay15']==first['flights'][0]['p_delay15']

def test_late_first_publication_is_reconstruction(tmp_path):
    folder=tmp_path/'data'/'live'/'aeroapi'/'2026-10-27'
    page(folder,'20261027T024000Z',[record()])
    day=P.build_live_day(date(2026,10,27),train(),tmp_path,datetime(2026,10,27,4,tzinfo=timezone.utc))
    assert day['metadata']['forecast_status']=='reconstructed'
    assert not (tmp_path/'data'/'forecasts'/'2026-10-27.json').exists()

def test_cutoff_without_snapshot_is_unavailable(tmp_path):
    folder=tmp_path/'data'/'live'/'aeroapi'/'2026-10-27'
    page(folder,'20261027T040000Z',[record()])
    day=P.build_live_day(date(2026,10,27),train(),tmp_path,datetime(2026,10,27,5,tzinfo=timezone.utc))
    assert day['metadata']['forecast_status']=='unavailable'
    assert day['flights'][0]['p_delay15'] is None

def test_provider_relative_pagination_link():
    assert live.next_url('/airports/KRDU/flights/departures?cursor=abc') == 'https://aeroapi.flightaware.com/aeroapi/airports/KRDU/flights/departures?cursor=abc'

def test_position_last_row_does_not_inherit_previous_flight_actual():
    from rdu.features.position import build
    from rdu.eval.walkforward import Data
    legs=pd.DataFrame({'flight_date':[date(2024,1,1)]*2,'tail':['N1','N1'],'dest':['RDU','ATL'],
                       'sched_dep_utc':pd.to_datetime(['2024-01-01T12:00Z','2024-01-02T03:00Z']),
                       'sched_arr_utc':pd.to_datetime(['2024-01-01T13:00Z','2024-01-02T05:00Z']),
                       'outcome_time_utc':pd.to_datetime(['2024-01-01T13:20Z',None]),'arr_delay':[20.,float('nan')]})
    flights=pd.DataFrame({'flight_id':['x'],'forecast_date':[date(2024,1,2)],'tail':['N1'],
                         'sched_dep_utc':pd.to_datetime(['2024-01-02T14:00Z']),'direction':['dep']})
    r=build(Data(legs,flights)).loc['x']
    assert r['prev_day_completed_before_cutoff']==0
    assert pd.isna(r['prev_day_last_arr_delay'])

def test_airline_without_iata_code_is_kept():
    rows=P.live_records({'records':{'a':record(operator_iata=None,operator='MTN',ident_iata=None,ident='MTN8508',flight_number='8508')}},date(2026,10,27))
    assert len(rows)==1 and rows[0]['carrier']=='MTN'

def test_weather_validation_checks_daily_and_spc_issue_time():
    from rdu.features.weather import validate_issue_times
    valid=pd.DataFrame({'forecast_date':[date(2024,3,2)],'issue_utc':pd.to_datetime(['2024-03-02T03:29Z'])})
    validate_issue_times(valid)
    valid['issue_utc']=pd.to_datetime(['2024-03-02T03:30Z'])
    with pytest.raises(ValueError): validate_issue_times(valid)

def test_publisher_scores_only_frozen_forecasts(tmp_path):
    folder=tmp_path/'data'/'live'/'aeroapi'/'2026-10-27'
    page(folder,'20261027T024000Z',[record()])
    P.build_live_day(date(2026,10,27),train(),tmp_path,datetime(2026,10,27,2,45,tzinfo=timezone.utc))
    page(folder,'20261027T150000Z',[record(actual_out='2026-10-27T13:20:00Z',scheduled_out='2026-10-27T13:10:00Z')],endpoint='departures')
    (tmp_path/'data'/'clean').mkdir()
    train().to_parquet(tmp_path/'data'/'clean'/'rdu_flights.parquet')
    P.publish(tmp_path,datetime(2026,10,27,16,tzinfo=timezone.utc))
    scores=json.loads((tmp_path/'data'/'scores'/'2026-10-27.json').read_text())
    assert scores['n']==1 and np.isfinite(scores['brier']) and np.isfinite(scores['crps'])
    bundle=json.loads((tmp_path/'web'/'departures_data.json').read_text())
    flight=bundle['weeks'][0]['days'][0]['flights'][0]
    assert flight['act_delay']==20  # retains the original forecast's schedule
    assert 'distribution' not in flight

def test_partial_schedule_is_never_frozen(tmp_path):
    folder=tmp_path/'data'/'live'/'aeroapi'/'2026-10-27'
    page(folder,'20261027T024000Z',[record()],next='/airports/KRDU/flights/scheduled_departures?cursor=x')
    day=P.build_live_day(date(2026,10,27),train(),tmp_path,datetime(2026,10,27,2,45,tzinfo=timezone.utc))
    assert day['metadata']['forecast_status']=='provisional'
    assert not (tmp_path/'data'/'forecasts'/'2026-10-27.json').exists()

def test_b2_pooling_attribution_reconciliation_and_notes():
    synth_train = pd.DataFrame({
        'forecast_date': [date(2024, 10, 1)] * 400,
        'outcome_time_utc': pd.to_datetime(['2024-10-01T15:00Z'] * 400),
        'direction': ['dep'] * 400,
        'carrier': ['DL'] * 100 + ['NK'] * 100 + ['F9'] * 100 + ['AA'] * 100,
        'flight_number': [100] * 50 + [101] * 5 + [102] * 45 + [200] * 100 + [300] * 100 + [400] * 100,
        # DL: mostly on-time (80 on-time, 15 delay, 5 cancel)
        # NK: delay dominant (10 on-time, 80 delay, 10 cancel)
        # F9: cancel dominant (10 on-time, 10 delay, 80 cancel)
        # AA: tied on-time & delay (45 on-time, 45 delay, 10 cancel)
        'bucket': ([0]*45 + [1]*5 + [0]*5 + [0]*40 + [2]*5) + ([0]*10 + [1]*80 + [4]*10) + ([0]*10 + [1]*10 + [4]*80) + ([0]*45 + [1]*45 + [4]*10),
        'y_capped': ([0]*45 + [20]*5 + [0]*5 + [0]*40 + [60]*5) + ([0]*10 + [25]*80 + [180]*10) + ([0]*10 + [20]*10 + [180]*80) + ([0]*45 + [20]*45 + [180]*10)
    })
    cutoff = pd.Timestamp('2026-10-27T03:00:00Z')
    flights = [
        {'id':'f1','direction':'dep','carrier':'DL','flight_number':100,'forecast_date':date(2026,10,27),'scheduled_out_utc':'2026-10-27T12:00Z'},
        {'id':'f2','direction':'dep','carrier':'DL','flight_number':101,'forecast_date':date(2026,10,27),'scheduled_out_utc':'2026-10-27T13:00Z'},
        {'id':'f3','direction':'dep','carrier':'DL','flight_number':999,'forecast_date':date(2026,10,27),'scheduled_out_utc':'2026-10-27T14:00Z'},
        {'id':'f4','direction':'dep','carrier':'NK','flight_number':200,'forecast_date':date(2026,10,27),'scheduled_out_utc':'2026-10-27T15:00Z'},
        {'id':'f5','direction':'dep','carrier':'F9','flight_number':300,'forecast_date':date(2026,10,27),'scheduled_out_utc':'2026-10-27T16:00Z'},
        {'id':'f6','direction':'dep','carrier':'AA','flight_number':400,'forecast_date':date(2026,10,27),'scheduled_out_utc':'2026-10-27T17:00Z'},
        # B6 500: Near-tie (e.g. ~45% vs ~44%)
        {'id':'f7','direction':'dep','carrier':'DL','flight_number':102,'forecast_date':date(2026,10,27),'scheduled_out_utc':'2026-10-27T18:00Z'},
    ]
    pred = P.baseline_predictions(flights, synth_train, cutoff)
    for f in pred:
        ev = f['risk_evidence']
        p0 = ev['rdu_delay_probability']
        total_change = sum(s['change_percentage_points'] for s in ev['stages']) / 100.0
        assert abs((p0 + total_change) - f['p_delay15']) < 1e-9

    # Sparse history warning
    assert 'sparse record, n<20' in pred[1]['why_detail']
    # Unseen flight warning
    assert 'No matching historical departures exist' in pred[2]['why_detail']
    # Delay dominant: strictly asserts delay is most likely outcome, NOT on-time
    assert 'A delay of 15+ minutes is the most likely single outcome' in pred[3]['why_detail']
    assert 'on-time departure remains the most likely' not in pred[3]['why_detail']
    # Cancellation dominant: strictly asserts cancellation is most likely outcome
    assert 'Cancellation is the most likely single outcome' in pred[4]['why_detail']
    # Exact tie probabilities: strictly asserts outcomes are equally likely with exact percentage
    assert 'are equally likely (45.0% each)' in pred[5]['why_detail']

    # Near-tie (e.g. 46.0% vs 45.0%): strictly asserts approximately equally likely with distinct percentages
    near_train = pd.DataFrame({
        'forecast_date': [date(2024, 10, 1)] * 100,
        'outcome_time_utc': pd.to_datetime(['2024-10-01T15:00Z'] * 100),
        'direction': ['dep'] * 100, 'carrier': ['WN'] * 100, 'flight_number': [888] * 100,
        'bucket': [0]*46 + [1]*45 + [4]*9, 'y_capped': [0]*46 + [20]*45 + [180]*9
    })
    pred_near = P.baseline_predictions([{'id':'fn','direction':'dep','carrier':'WN','flight_number':888,'forecast_date':date(2026,10,27),'scheduled_out_utc':'2026-10-27T18:00Z'}],
                                       near_train, cutoff)
    assert 'are approximately equally likely' in pred_near[0]['why_detail']
    assert '46.0%' in pred_near[0]['why_detail'] and '45.0%' in pred_near[0]['why_detail']

    # Three-way tie
    three_train = pd.DataFrame({
        'forecast_date': [date(2024, 10, 1)] * 300,
        'outcome_time_utc': pd.to_datetime(['2024-10-01T15:00Z'] * 300),
        'direction': ['dep'] * 300, 'carrier': ['UA'] * 300, 'flight_number': [777] * 300,
        'bucket': [0]*100 + [1]*100 + [4]*100, 'y_capped': [0]*100 + [20]*100 + [180]*100
    })
    pred_three = P.baseline_predictions([{'id':'ft','direction':'dep','carrier':'UA','flight_number':777,'forecast_date':date(2026,10,27),'scheduled_out_utc':'2026-10-27T18:00Z'}],
                                        three_train, cutoff)
    assert 'All three outcomes are equally likely (33.3% each)' in pred_three[0]['why_detail']

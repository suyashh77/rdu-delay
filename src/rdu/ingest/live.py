"""Timestamped, budget-limited FlightAware collection and nightly publication."""
import csv
import json
import logging
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
LIVE = ROOT / 'data' / 'live'
ET = ZoneInfo('America/New_York')
FAA_URL = 'https://nasstatus.faa.gov/api/airport-status-information'
AEROAPI = 'https://aeroapi.flightaware.com/aeroapi'
PRICE_PER_PAGE = {'scheduled_departures': .005, 'scheduled_arrivals': .005, 'departures': .005}
MAX_PAGES = 25
MONTHLY_CAP = 5.00
RATE_SLEEP_S = 7
log = logging.getLogger('rdu.live')

def _setup_logging():
    (LIVE / 'logs').mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=LIVE / 'logs' / 'live.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

def _now_utc():
    return datetime.now(timezone.utc)

def _get(url, headers=None, timeout=30):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=timeout) as r:
        return r.read()

def _provenance(source, path, url, retrieved, note=''):
    p = ROOT / 'data' / 'PROVENANCE.csv'
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with p.open('a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if new: writer.writerow(['source', 'path', 'url', 'retrieved_at_utc', 'note'])
        writer.writerow([source, path.relative_to(ROOT).as_posix(), url.split('?')[0], retrieved.isoformat(), note])

def faa():
    body = _get(FAA_URL)
    t = _now_utc()
    path = LIVE / 'faa' / t.strftime('%Y-%m-%d') / f"{t.strftime('%H%M%S')}Z.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    _provenance('faa_nas_status', path, FAA_URL, t)
    return path

def _api_key():
    key = os.environ.get('AEROAPI_KEY', '').strip()
    env = ROOT / '.env'
    if not key and env.exists():
        for line in env.read_text(encoding='utf-8-sig').splitlines():
            if line.strip().startswith('#') or '=' not in line: continue
            name, value = line.split('=', 1)
            if name.strip() == 'AEROAPI_KEY': key = value.strip().strip('\"\'')
    if not key: raise RuntimeError('AEROAPI_KEY not set (environment or .env)')
    return key

def _ledger_month_cost(month):
    p = LIVE / 'aeroapi' / 'ledger.csv'
    if not p.exists(): return 0.
    with p.open(encoding='utf-8') as f:
        return sum(float(r['cost_usd']) for r in csv.DictReader(f) if r['retrieved_at_utc'].startswith(month))

def _ledger_add(retrieved, endpoint, target, pages, cost):
    p = LIVE / 'aeroapi' / 'ledger.csv'
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with p.open('a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if new: writer.writerow(['retrieved_at_utc','endpoint','target_date','pages','cost_usd'])
        writer.writerow([retrieved.isoformat(), endpoint, target, pages, f'{cost:.4f}'])

def month_spend(key):
    t = _now_utc()
    start = t.strftime('%Y-%m-01T00:00:00Z')
    reported = json.loads(_get(f'{AEROAPI}/account/usage?start={start}', {'x-apikey':key}))['total_cost']
    return max(float(reported), _ledger_month_cost(t.strftime('%Y-%m')))

def day_window(target):
    """Calendar-day boundaries in RDU time, including 23/25-hour DST days."""
    return (datetime.combine(target, datetime.min.time(), ET).astimezone(timezone.utc),
            datetime.combine(target + timedelta(days=1), datetime.min.time(), ET).astimezone(timezone.utc))

def next_url(link):
    if not link: return None
    if link.startswith('/') and not link.startswith('/aeroapi/') and not link.startswith('//'):
        link = '/aeroapi' + link
    url = urllib.parse.urljoin(AEROAPI + '/', link)
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != 'https' or parts.netloc != 'aeroapi.flightaware.com' or not parts.path.startswith('/aeroapi/'):
        raise ValueError('Refusing pagination outside the FlightAware API')
    return url

def collect(target, endpoints=('scheduled_departures',), max_pages=MAX_PAGES):
    if not 1 <= max_pages <= 100: raise ValueError('max_pages must be between 1 and 100')
    if any(e not in PRICE_PER_PAGE for e in endpoints): raise ValueError('Unsupported endpoint')
    base = LIVE / 'aeroapi'
    base.mkdir(parents=True, exist_ok=True)
    lock = base / '.collection.lock'
    # Serialize collectors so two invocations cannot spend the same remaining budget.
    with lock.open('x') as lockfile:
        lockfile.write(str(os.getpid()))
    try:
        key = _api_key()
        spent = month_spend(key)
        worst = sum(PRICE_PER_PAGE[e] for e in endpoints) * max_pages
        if spent + worst > MONTHLY_CAP:
            log.warning('Budget prevents collection: %.3f + %.3f > %.2f', spent, worst, MONTHLY_CAP)
            return {'skipped':True, 'target':str(target), 'reason':'monthly_budget', 'month_spend':spent}
        start, end = day_window(target)
        query = urllib.parse.urlencode({'type':'Airline', 'start':start.strftime('%Y-%m-%dT%H:%M:%SZ'),
                                       'end':end.strftime('%Y-%m-%dT%H:%M:%SZ'), 'max_pages':1})
        folder = base / str(target)
        folder.mkdir(exist_ok=True)
        result = {'target':str(target), 'month_spend_before':spent}
        for endpoint in endpoints:
            url = f'{AEROAPI}/airports/KRDU/flights/{endpoint}?{query}'
            manifest = {'endpoint':endpoint, 'target_date':str(target), 'started_at_utc':_now_utc().isoformat(),
                        'files':[], 'complete':False, 'records':0, 'error':None}
            manifest_path = folder / f"manifest_{endpoint}_{_now_utc():%Y%m%dT%H%M%S%fZ}.json"
            try:
                for page in range(1, max_pages + 1):
                    # Reserve before request: network errors and malformed responses may still be billed.
                    requested = _now_utc()
                    _ledger_add(requested, endpoint, target, 1, PRICE_PER_PAGE[endpoint])
                    body = _get(url, {'x-apikey':key})
                    retrieved = _now_utc()
                    path = folder / f'{endpoint}_{retrieved:%Y%m%dT%H%M%S%fZ}_p{page:02d}.json'
                    path.write_bytes(body)
                    _provenance('aeroapi', path, url, retrieved, f'target={target}')
                    response = json.loads(body)
                    if not isinstance(response.get(endpoint), list): raise ValueError('Unexpected response shape')
                    manifest['files'].append(path.name)
                    manifest['records'] += len(response[endpoint])
                    url = next_url((response.get('links') or {}).get('next'))
                    time.sleep(RATE_SLEEP_S)
                    if not url:
                        manifest['complete'] = True
                        break
            except Exception as exc:
                # Error text may contain provider URLs but never request headers or the API key.
                manifest['error'] = type(exc).__name__
                log.error('Collection failed for %s %s: %s', target, endpoint, type(exc).__name__)
            finally:
                manifest['retrieved_at_utc'] = _now_utc().isoformat()
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            result[endpoint] = {k:v for k,v in manifest.items() if k != 'files'}
        log.info('collection %s', json.dumps(result))
        return result
    finally:
        lock.unlink(missing_ok=True)

def nightly(max_pages=MAX_PAGES):
    return collect(_now_utc().astimezone(ET).date() + timedelta(days=1),
                   ('scheduled_departures','scheduled_arrivals'), max_pages)

def pipeline(max_pages=MAX_PAGES):
    """Existing nightly schedule: collect, publish before cutoff, then collect outcomes."""
    from rdu.publish import publish
    target = _now_utc().astimezone(ET).date() + timedelta(days=1)
    # B2 needs only departures; omit unused arrival calls to preserve the monthly budget.
    results = {'schedule':collect(target, ('scheduled_departures',), max_pages)}
    results['publication'] = publish()
    today = _now_utc().astimezone(ET).date()
    results['outcomes'] = collect(today - timedelta(days=1), ('departures',), max_pages)
    results['publication'] = publish()
    published = any(d['date'] == str(target) and d['status'] == 'published' for d in results['publication']['live_days'])
    results['ok'] = (published and results['schedule'].get('scheduled_departures',{}).get('complete',False) and results['outcomes'].get('departures',{}).get('complete',False))
    return results

if __name__ == '__main__':
    _setup_logging()
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    try:
        if cmd == 'faa': result = {'saved':str(faa())}
        elif cmd in ('nightly','pipeline'):
            result = globals()[cmd](int(sys.argv[2]) if len(sys.argv)>2 else MAX_PAGES)
        elif cmd == 'outcomes':
            target = date.fromisoformat(sys.argv[2]) if len(sys.argv)>2 else _now_utc().astimezone(ET).date()-timedelta(days=1)
            result = collect(target, ('departures',))
        else: raise SystemExit('usage: python -m rdu.ingest.live faa|nightly|pipeline [max_pages]|outcomes [YYYY-MM-DD]')
        print(json.dumps(result, default=str))
        if result.get('skipped') or result.get('ok') is False or any(isinstance(v,dict) and v.get('complete') is False for v in result.values()):
            raise SystemExit(2)
    except Exception:
        log.exception('live %s failed', cmd)
        raise

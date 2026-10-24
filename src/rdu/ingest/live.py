"""Start-now loggers (HANDOFF §3). Raw responses saved untouched, with retrieval time.

    python -m rdu.ingest.live faa                 # FAA NAS status XML (free), every 15 min
    python -m rdu.ingest.live nightly [max_pages] # AeroAPI: tomorrow's KRDU schedule, ~22:40 ET

AeroAPI budget (Personal tier, $5/month free): before any paid call we take the larger of
FlightAware's reported month-to-date cost (free endpoint) and our own ledger, and skip the
run if it plus this run's worst case would pass MONTHLY_CAP. Prices from FlightAware's
AeroAPI page, checked 2026-10-24.
"""
import json
import logging
import os
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
LIVE = ROOT / "data" / "live"
ET = ZoneInfo("America/New_York")
FAA_URL = "https://nasstatus.faa.gov/api/airport-status-information"
AEROAPI = "https://aeroapi.flightaware.com/aeroapi"
PRICE_PER_PAGE = {"scheduled_departures": 0.005, "scheduled_arrivals": 0.005}
MAX_PAGES = 15          # per endpoint per night; RDU needs ~11
MONTHLY_CAP = 4.50      # stop short of the $5 free credit
RATE_SLEEP_S = 7        # Personal tier allows 10 result sets per minute

log = logging.getLogger("rdu.live")


def _setup_logging() -> None:
    (LIVE / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=LIVE / "logs" / "live.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _provenance(source: str, path: Path, url: str, retrieved: datetime, note: str = "") -> None:
    p = ROOT / "data" / "PROVENANCE.csv"
    new = not p.exists()
    with p.open("a", encoding="utf-8") as fh:
        if new:
            fh.write("source,path,url,retrieved_at_utc,note\n")
        fh.write(f"{source},{path.relative_to(ROOT).as_posix()},{url.split('?')[0]},{retrieved.isoformat()},{note}\n")


def _get(url: str, headers: dict | None = None, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def faa() -> Path:
    t = _now_utc()
    body = _get(FAA_URL)
    path = LIVE / "faa" / t.strftime("%Y-%m-%d") / f"{t.strftime('%H%M%S')}Z.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    _provenance("faa_nas_status", path, FAA_URL, t)
    return path


def _api_key() -> str:
    key = os.environ.get("AEROAPI_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        pairs = dict(line.split("=", 1) for line in env.read_text().splitlines() if "=" in line)
        key = pairs.get("AEROAPI_KEY", "").strip()
    if not key:
        raise RuntimeError("AEROAPI_KEY not set (environment or .env)")
    return key


def _ledger_month_cost(month: str) -> float:
    p = LIVE / "aeroapi" / "ledger.csv"
    if not p.exists():
        return 0.0
    rows = [line.split(",") for line in p.read_text().splitlines()[1:]]
    return sum(float(r[4]) for r in rows if r[0].startswith(month))


def _ledger_add(retrieved: datetime, endpoint: str, target: date, pages: int, cost: float) -> None:
    p = LIVE / "aeroapi" / "ledger.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with p.open("a", encoding="utf-8") as fh:
        if new:
            fh.write("retrieved_at_utc,endpoint,target_date,pages,cost_usd\n")
        fh.write(f"{retrieved.isoformat()},{endpoint},{target},{pages},{cost:.4f}\n")


def month_spend(key: str) -> float:
    """Larger of FlightAware's reported month-to-date cost and our ledger."""
    t = _now_utc()
    start = t.strftime("%Y-%m-01T00:00:00Z")
    reported = json.loads(_get(f"{AEROAPI}/account/usage?start={start}", {"x-apikey": key}))["total_cost"]
    return max(float(reported), _ledger_month_cost(t.strftime("%Y-%m")))


def nightly(max_pages: int = MAX_PAGES) -> dict:
    key = _api_key()
    worst_case = sum(PRICE_PER_PAGE.values()) * max_pages
    spent = month_spend(key)
    if spent + worst_case > MONTHLY_CAP:
        log.warning("SKIP nightly: month spend %.3f + worst case %.3f > cap %.2f", spent, worst_case, MONTHLY_CAP)
        return {"skipped": True, "month_spend": spent}

    target = datetime.now(ET).date() + timedelta(days=1)
    start = datetime.combine(target, datetime.min.time(), ET).astimezone(timezone.utc)
    end = start + timedelta(days=1)
    q = f"?type=Airline&start={start:%Y-%m-%dT%H:%M:%SZ}&end={end:%Y-%m-%dT%H:%M:%SZ}&max_pages=1"
    out_dir = LIVE / "aeroapi" / str(target)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"target": str(target), "month_spend_before": spent}
    for endpoint, price in PRICE_PER_PAGE.items():
        url, pages, records = f"{AEROAPI}/airports/KRDU/flights/{endpoint}{q}", 0, 0
        while url and pages < max_pages:
            t = _now_utc()
            body = _get(url, {"x-apikey": key})
            pages += 1
            path = out_dir / f"{endpoint}_{t.strftime('%Y%m%dT%H%M%SZ')}_p{pages:02d}.json"
            path.write_bytes(body)
            _provenance("aeroapi", path, url, t, f"target={target}")
            data = json.loads(body)
            records += len(data.get(endpoint, []))
            nxt = (data.get("links") or {}).get("next")
            url = f"{AEROAPI}{nxt}" if nxt else None
            if url:
                time.sleep(RATE_SLEEP_S)
        _ledger_add(_now_utc(), endpoint, target, pages, pages * price)
        result[endpoint] = {"pages": pages, "records": records, "complete": url is None}
        if url:
            log.warning("%s truncated at %d pages", endpoint, pages)
        time.sleep(RATE_SLEEP_S)
    log.info("nightly %s", json.dumps(result))
    return result


if __name__ == "__main__":
    _setup_logging()
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        if cmd == "faa":
            log.info("faa saved %s", faa().name)
        elif cmd == "nightly":
            print(json.dumps(nightly(int(sys.argv[2]) if len(sys.argv) > 2 else MAX_PAGES)))
        else:
            raise SystemExit("usage: python -m rdu.ingest.live faa|nightly [max_pages]")
    except Exception:
        log.exception("live %s failed", cmd)
        raise

"""Rebuild the website from timestamped live snapshots and saved backtest results."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rdu.publish import publish

if __name__ == '__main__':
    print(json.dumps(publish(), indent=2))

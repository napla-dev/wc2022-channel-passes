"""Stage 1 — spatial channel-pass candidate generator.

Runs the label-free geometric detector (src/channel.py) over every WC2022 match
that has event data (all 64), writing the candidate table to
outputs/channel_passes.csv. This is the high-recall CANDIDATE set; Stage 2
(scripts/channel_classifier.py) scores each candidate with the supervised model.

Usage:
    python scripts/extract_channels.py --workers 8
    python scripts/extract_channels.py --matches 3835 10503   (subset)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

import pandas as pd

from src import data_loader as dl
from src import channel as ch

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--matches", nargs="*", default=None)
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    ids = args.matches or dl.available_matches_event_only(DATA_DIR)
    print(f"Extracting channel-pass candidates from {len(ids)} matches...")

    t0 = time.time()
    rows, n_pass, n_D = [], 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(ch.extract_match, m, str(DATA_DIR)): m for m in ids}
        for fut in as_completed(futs):
            res = fut.result()
            if "error" in res:
                print(f"  !! {res['match_id']}: {res['error']}"); continue
            rows.extend(res["rows"]); n_pass += res["n_pass_eval"]; n_D += res["n_D_total"]

    df = pd.DataFrame(rows)
    print(f"Done in {time.time()-t0:.0f}s. evaluated {n_pass} passes -> "
          f"{len(df)} candidates ({100*len(df)/max(n_pass,1):.1f}%).")
    if df.empty:
        print("No candidates — check data/ path."); return
    df.to_csv(OUT_DIR / "channel_passes.csv", index=False)

    print(f"\nchannel_type split:\n{df['channel_type'].value_counts().to_string()}")
    print(f"\nmedians: depth_behind={df['depth_behind'].median():.1f}  "
          f"iso_dist={df['iso_dist'].median():.1f}  "
          f"end_x={df['end_x'].median():.1f}  length={df['length'].median():.1f}")
    print(f"\nWrote {OUT_DIR/'channel_passes.csv'}")


if __name__ == "__main__":
    main()

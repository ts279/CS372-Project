"""
Download MLB Statcast pitch-level data from Baseball Savant, one month at a time, resumably.

Pipeline stage 1 of 7 (fetch). Stages: fetch -> features -> train -> evaluate -> comps report -> app tables -> app.

Inputs:  Baseball Savant (via pybaseball.statcast), for 2023-2026.
Outputs: data/raw/statcast_YYYY_MM.parquet, one file per calendar month.

Each month is written as soon as it lands, so an interrupted run loses at most
one month. Re-running skips months already on disk. The season in progress is
clipped to yesterday, so a month file is never written from a half-played day.
Its last month is therefore partial; re-pull it later with --refresh.

Data (D8): MLB regular season, 2023-26. Splits (D5): 2023-24 train, 2025 validation, 2026 test.

Usage (from the repo root):
    python -m src.data.fetch_statcast --sample [YYYY-MM-DD]   # one day, nothing written
    python -m src.data.fetch_statcast                         # 2023-2025
    python -m src.data.fetch_statcast --seasons 2026          # test season
    python -m src.data.fetch_statcast --seasons 2026 --refresh 2026_09
    python -m src.data.fetch_statcast --seasons 2024 2025 --refresh 2024_03 2025_03   # openers

AI assistance: drafted with Claude Code (Anthropic); reviewed, run, and modified by
Toma Shigaki-Than. See ATTRIBUTION.md.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

import pybaseball
from pybaseball import statcast

pybaseball.cache.enable()  # a re-run reads pybaseball's local cache instead of re-downloading

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

# Regular season windows, padded a few days on each end. Statcast simply
# returns nothing for dates with no games, so padding is free. Savant also
# returns spring-training and postseason games inside a window, so
# build_arsenal filters to game_type == "R".
SEASON_WINDOWS = {
    2023: ("2023-03-30", "2023-10-02"),
    # why: start March 15 to catch the Seoul Series (2024-03-20/21) and the Tokyo Series
    # (2025-03-18/19), which are regular-season games played before the domestic Opening Day.
    2024: ("2024-03-15", "2024-10-01"),
    2025: ("2025-03-15", "2025-10-01"),
    # why: 2026 is the held-out test season. The start is padded early to catch
    # any opening series before the domestic Opening Day.
    2026: ("2026-03-18", "2026-10-01"),
}

SAMPLE_DAY = "2026-06-02"  # default --sample day: a normal mid-season day


def clip_to_yesterday(end: str) -> str:
    """Clip a window end so the pull never includes today's unfinished games."""
    yesterday = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    return min(pd.Timestamp(end), yesterday).strftime("%Y-%m-%d")


def month_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """Split a season into calendar-month [start, end] string pairs."""
    starts = pd.date_range(start=start, end=end, freq="MS").tolist()
    first = pd.Timestamp(start)
    if not starts or starts[0] > first:
        starts.insert(0, first)
    out = []
    for i, s in enumerate(starts):
        e = (starts[i + 1] - pd.Timedelta(days=1)) if i + 1 < len(starts) else pd.Timestamp(end)
        out.append((s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
    return out


def fetch_month(season: int, start: str, end: str, retries: int = 3, refresh: bool = False) -> int:
    """Fetch one month to parquet. Returns rows written (0 if skipped/empty).

    With refresh=True an existing file is re-downloaded. It is replaced only
    after the new pull succeeds.
    """
    tag = f"{season}_{start[5:7]}"
    dest = RAW / f"statcast_{tag}.parquet"
    if dest.exists() and not refresh:
        print(f"  [skip] {tag} already on disk", flush=True)
        return 0

    for attempt in range(1, retries + 1):
        try:
            df = statcast(start_dt=start, end_dt=end, verbose=False)
            break
        except Exception as exc:  # network flakiness is the norm here
            wait = 10 * attempt
            print(f"  [retry {attempt}/{retries}] {tag}: {type(exc).__name__}: {exc}", flush=True)
            if attempt == retries:
                print(f"  [FAIL] {tag} — rerun the script later to pick it up", flush=True)
                return 0
            time.sleep(wait)

    if df is None or df.empty:
        print(f"  [empty] {tag}", flush=True)
        return 0

    df.to_parquet(dest, index=False)
    print(f"  [ok]   {tag}: {len(df):,} pitches, dates {start}..{end} -> {dest.name}", flush=True)
    return len(df)


def run_sample(day: str) -> int:
    """Pull one day into memory and compare its schema with 2025. Writes nothing."""
    print(f"[sample] fetching {day} (nothing is written to disk)", flush=True)
    df = statcast(start_dt=day, end_dt=day, verbose=False)
    print(f"[sample] shape {df.shape}", flush=True)
    print(f"[sample] game_type counts: {df['game_type'].value_counts().to_dict()}", flush=True)
    print(f"[sample] game_year: {sorted(df['game_year'].unique().tolist())}", flush=True)
    ref = sorted(RAW.glob("statcast_2025_*.parquet"))
    if ref:
        ref_cols = set(pq.read_schema(ref[0]).names)
        new_cols = set(df.columns)
        print(f"[sample] columns: sample day={len(new_cols)}, 2025 file={len(ref_cols)}", flush=True)
        print(f"[sample] only in sample day: {sorted(new_cols - ref_cols)}", flush=True)
        print(f"[sample] only in 2025: {sorted(ref_cols - new_cols)}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", type=int, default=[2023, 2024, 2025])
    ap.add_argument("--sample", nargs="?", const=SAMPLE_DAY, default=None, metavar="YYYY-MM-DD",
                    help=f"fetch one day (default {SAMPLE_DAY}) and check its schema; write nothing")
    ap.add_argument("--refresh", nargs="*", default=[], metavar="YYYY_MM",
                    help="re-download these months even if on disk (e.g. the partial current month)")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    if args.sample:
        return run_sample(args.sample)

    total = 0
    t0 = time.time()

    for season in args.seasons:
        if season not in SEASON_WINDOWS:
            print(f"No window defined for {season}, skipping", flush=True)
            continue
        start, end = SEASON_WINDOWS[season]
        end = clip_to_yesterday(end)
        if pd.Timestamp(end) < pd.Timestamp(start):
            print(f"{season} hasn't started yet, skipping", flush=True)
            continue
        print(f"\n=== {season} ({start}..{end}) ===", flush=True)
        for m_start, m_end in month_chunks(start, end):
            tag = f"{season}_{m_start[5:7]}"
            total += fetch_month(season, m_start, m_end, refresh=tag in args.refresh)

    mins = (time.time() - t0) / 60
    print(f"\nDone. {total:,} new pitches in {mins:.1f} min.", flush=True)
    print(f"Raw files: {RAW}", flush=True)

    files = sorted(RAW.glob("statcast_*.parquet"))
    print(f"{len(files)} month files on disk.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

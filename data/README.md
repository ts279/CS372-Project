# data/

Data files are not committed because the raw Statcast pull is about 325 MB. Recreate them with the access script. See [SETUP.md](../SETUP.md) for the full steps.

- `raw/`: monthly Statcast parquet files downloaded by `src/data/fetch_statcast.py`.
  - 2023–2025: 24 files, 2.1M pitches (train and validation).
  - 2026: March onward (test).
- `processed/`: built by `src/features/build_arsenal.py`.
  - `pitches.parquet`: the cleaned pitch table with split and label.
  - `arsenal_pitch_type.parquet`: one row per pitcher-season × pitch type.
  - `pitcher_seasons.parquet`: one row per pitcher-season.
  - `player_positions.parquet`: the statsapi position cache.

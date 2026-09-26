# data/reference

## tjstats_site_2025.csv
Hand-transcribed by Toma Shigaki-Than from two screenshots of the TJStats Pitch Similarity tool
(T. Nestico, https://tjstats.ca/pitch-similarity/). These are the site's own
outputs. `src/eval/tjstats_fidelity.py` uses them to fit the tool's unpublished kernel width
(sigma) and to check how closely our TJStats rebuild reproduces the site.

- **Settings (both screenshots):** MLB, 2025 Regular Season, Full Arsenal, Same Hand ON,
  Min Pitches 10, Show 20, default metrics (Velocity, iVB, HB, Extension, Release Height,
  Arm Angle). Every other option was left at its default.
- **Sets:** `fit` = Tarik Skubal (LHP), used to fit sigma and pick the variant. `holdout` = Zack
  Wheeler (RHP), scored once with the frozen choice.
- **Columns:**
  - `set`, `target`, `target_hand`
  - `row_kind`: `usage` = the target's displayed usage % per pitch type; `comp` = one comp row
  - `rank`: 1-20 (0 for usage rows)
  - `comp`: the name as shown on the site ("First Last")
  - `team`, `matched`: the "k/6" matched-types badge
  - `pitch_type`: the raw Statcast code, or `OVERALL`
  - `site_pct`: an integer %; empty = "-" (not matched)
- **Known gap:** on the Wheeler screenshot, a thumbnail covers the rank-1 name (team PIT,
  matched 4/6). Two 2025 PIT right-handers fit its pitch-type pattern (Mitch Keller, Yohan
  Ramírez). The fidelity script therefore leaves that row out of the error metrics and counts
  the top-20 overlap over the 19 named rows.
- **Display rounding:** the site's Overall column equals the sum over the target's pitch types
  of (displayed usage × per-pitch %), with unmatched types counting 0. That holds to within
  about 1 point, because the displayed usage is rounded.

## tjstats_site_2026.csv
The site's **single-pitch** list for Cam Schlittler's 2026 four-seam fastball, 20 rows (the target plus 19
named comps), read off a screenshot the author supplied and typed in by Claude Code; the author provided the
screenshot and its settings.

- **Settings:** MLB, 2026 Regular Season, Pitch = 4-Seam Fastball (not Full Arsenal), Same Hand ON,
  Min Pitches 10, Show 20, default metrics. No role filter and no season-pitch floor — those are ours, not
  the site's.
- **Why a second page:** the 2025 pages fitted sigma and checked it once on a held-out pitcher. This page is
  from a season sigma was never fitted to, and it isolates the per-pitch kernel from the arsenal weighting,
  because single-pitch mode applies no usage weights.
- **Columns:** as in `tjstats_site_2025.csv`; `row_kind` is `target` for the highlighted row and `comp` for
  the ranked rows, `site_pct` is the SIMILARITY column, and `matched` is empty (single-pitch mode has no
  matched-types badge).
- **Scored by:** `python -m src.eval.tjstats_fidelity --single-pitch` → `docs/results/tjstats_fidelity_2026.csv`.
  Nothing is fitted there: sigma and the variant stay frozen at their 2025 values.

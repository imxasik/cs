# Data files

Put your cyclone track files (`.txt`) here. Sub-folders are allowed and a
nice way to organise by year (e.g. `data/2025/ditwah.txt`) — the file
picker shows all of them, newest first.

## Format v2 (recommended — the least typing)

```
# optional comment lines start with #
NAME 06B                 <- optional; file name is used otherwise
KIND INVEST              <- optional; INVEST or CYCLONE (auto-guessed)

OBS
2026-09-19 00:00   13.00   95.00   10   1007
...

FORECAST
2026-09-22 12:00   17.50   85.50   30   1.3    -      -
...
```

Fixed column order, separated by spaces (or commas/tabs):

| Section    | Columns                                              |
|------------|------------------------------------------------------|
| `OBS`      | time (UTC `YYYY-MM-DD HH:MM`), lat, lon, wind (kt), pressure (hPa) |
| `FORECAST` | time (UTC), lat, lon, wind (kt), r24, r34, r64 (degrees of radius) |

Rules: one fix per line · `-` (or `00`, `--`, empty) means "none" ·
`#` comments and blank lines are skipped · pressure / radii columns may be
left out entirely · `NAME`/`KIND` may be omitted (a file named `06B.txt`
becomes invest 06B automatically).

## Legacy format (still accepted)

Older files keep working untouched: a `Cyclone Name:` / `Invest Name:`
header, `Synoptic Time, Latitude, ...` column headers in any order/case,
and a `=== FORECAST ===` divider. The loader also accepts any separator,
synonym column names and missing-value tokens — but for new files, v2 is
the clean way.

## Notes

- The observed section must come first; at least 2 forecast points are
  needed for the uncertainty cone.
- Keep the reference CSVs (`climo.csv`, `climo_bias.csv`) in `assets/` —
  the AI features read them from there.
- Coastline and country boundaries are vector data in `assets/geo/`
  (`land.geojson`, `borders.geojson` — public domain, pre-built);
  rebuild with `scripts/make_coastline.py` / `scripts/make_borders.py`
  if ever needed.

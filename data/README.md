# Data files

Put your cyclone track files (`.txt`) here. Sub-folders are allowed and a
nice way to organise by year (e.g. `data/2025/ditwah.txt`) — the file
picker shows all of them, newest first.

## Format v2 (recommended) — titles + vertical-line columns

Every section starts with a **TITLE row** so you always know which number
is what, and the columns are separated by a **`|` vertical line** — no
wide padding to line up, nothing to guess when you type a new fix:

```
# optional comment lines start with #
NAME 06B                 <- optional; file name is used otherwise
KIND INVEST              <- optional; INVEST or CYCLONE (auto-guessed)

OBS
TIME (UTC) | LAT | LON | WIND (KT) | PRESSURE (HPA)
2026-09-19 00:00 | 13.00 | 95.00 | 10 | 1007
2026-09-19 06:00 | 13.10 | 94.00 | 10 | 1007

FORECAST
TIME (UTC) | LAT | LON | WIND (KT) | R24 | R34 | R64
2026-09-22 12:00 | 17.50 | 85.50 | 30 | 1.3 | - | -
2026-09-23 00:00 | 17.80 | 85.00 | 35 | 1.5 | 1.0 | -
```

| Section    | Columns (after `TIME (UTC)`)                          |
|------------|-------------------------------------------------------|
| `OBS`      | LAT, LON, WIND (KT), PRESSURE (HPA)                   |
| `FORECAST` | LAT, LON, WIND (KT), R24, R34, R64 (wind radii, degrees) |

Rules: one fix per line · `-` (or `00`, `--`, empty) means "none" ·
`#` comments and blank lines are skipped · the title row is optional (the
fixed column order above is assumed without it) · units in parentheses are
only hints for the eye and are ignored by the loader · `NAME`/`KIND` may
be omitted (a file named `06B.txt` becomes invest 06B automatically).

Plain spaces or commas also work as separators — but `|` keeps hand-edited
rows tidy without extra spacing, so it is the style to use in new files.

## Legacy format (still accepted)

Older files keep working untouched: a `Cyclone Name:` / `Invest Name:`
header, `Synoptic Time, Latitude, ...` column headers in any order/case,
and a `=== FORECAST ===` divider. The loader also accepts any separator,
synonym column names and missing-value tokens — but for new files, v2 is
the clean way.

## Notes

- The observed section must come first; at least 2 forecast points are
  needed for the uncertainty cone.
- Coastline and country boundaries are vector data in `assets/geo/`
  (`land.geojson`, `borders.geojson` — public domain, pre-built);
  rebuild with `scripts/make_coastline.py` / `scripts/make_borders.py`
  if ever needed.

# Data files

Put your cyclone track files (`.txt`) here. Sub-folders are allowed and are a
nice way to organise by year (e.g. `data/2025/ditwah.txt`) — the file picker
shows all of them, newest first.

## File format

```
Cyclone Name: DITWAH              <- or:  Invest Name: 95B
Synoptic Time, Latitude, Longitude, Intensity, Pressure
2025-11-25 00:00, 05.20, 78.90, 15, 1009
...one row per 6-h observation...
=== FORECAST ===
Synoptic Time, Latitude, Longitude, Intensity, WindR24, WindR34, WindR64
2025-12-03 06:00, 11.9, 79.2, 20, 00, 00, 00
...one row per 6-h forecast...
```

| Field      | Unit / format                         |
|------------|----------------------------------------|
| Synoptic Time | `YYYY-MM-DD HH:MM` (UTC)            |
| Latitude / Longitude | decimal degrees (N/E positive) |
| Intensity  | 1-minute sustained wind, **knots**     |
| Pressure   | hPa (observed section only)            |
| WindR24 / WindR34 / WindR64 | wind-radius in degrees of radius for the 24/34/64-kt isotachs (forecast only, `00` = none) |

Notes:

- The first line decides the title: `Cyclone Name: X` or `Invest Name: X`.
- The observed section must come first; `=== FORECAST ===` starts the forecast section.
- At least 2 forecast points are needed for the uncertainty cone.
- Keep the reference CSVs (`climo.csv`, `climo_bias.csv`) that live in `assets/` —
  the AI features read them from there.

# Cyclone Track & Cone Plotter — Professional Edition

Reads a combined observed/forecast cyclone file and produces an
**agency-style track & uncertainty-cone graphic**: a clean five-zone
dashboard (header · map · sidebar · forecast band · footer) where every
panel is generated from the data itself, so nothing is ever hard-coded,
crowded or overlapping.

You only edit **config.ini**, **data/** and (optionally) **features/** —
normally you never touch a `.py` file.

## The layout

```
+-----------------------------------------------------------------+
| HEADER  brand/logo · storm title · validity · issued card       |
+---------------------------------------------+-------------------+
|                                             | SIDEBAR           |
| MAP                                         | STORM SUMMARY     |
| observed + forecast track, cone,            | MAP KEY           |
| wind radii, ports, landfall,                | NEAREST 4 PORTS   |
| collision-free label chips,                 |                   |
| scale bar, north arrow                      |                   |
+---------------------------------------------+                   |
| BAND  map key strip · forecast table        |                   |
+---------------------------------------------+-------------------+
| FOOTER  wind · pressure · updated                    © brand    |
+-----------------------------------------------------------------+
```

Why it stays clean, whatever the data:

* **Zones are separate axes** — cards live outside the map, so a table can
  never cover a wind ring and a legend can never cover the coast.
* **Everything is dynamic** — a card, legend section, table column or label
  appears only when the matching datum exists (no forecast → no cone card;
  a post-landfall track → an **OverLand** status above the forecast table; 12 forecast steps →
  the table thins itself to `forecast_table_max_cols` columns; …).
* **The map window is solved from what is drawn** — every wind-radius ring
  and the cone fit fully inside the frame, clipped to the basemap
  coverage, so nothing is cut off and no empty ocean is wasted.
* **The coast is vector, not a raster image** — GSHHS high-res shoreline
  drawn as theme-coloured land with a soft shallow-water halo, razor-sharp
  at every zoom and DPI, plus thin dashed **country boundaries** at
  1:50 m scale (Natural Earth 50 m class) so the political geography reads
  at a glance.
* **Labels are collision-solved** — every chip (forecast points, NOW,
  landfall, ports) is placed in the first free slot around its marker;
  if no slot is free the *label* is dropped, never drawn on top of
  something else.
* **Type sizes are measured, not guessed** — glyph widths come from the
  real font; the sidebar solves one font scale that fits all its cards,
  tables shrink (then ellipsise) before they could ever overflow.

## Structure

- `main.py` – entry point (interactive wizard or one-shot CLI).
- `config.ini` – every toggle/number, no Python needed.
- `run.sh` – launcher (installs dependencies on first run).
- `cyclone/`
  - `cli.py` – wizard + command-line interface.
  - `config.py` – reads `config.ini` (all values optional, safe defaults).
  - `data_loader.py` – forgiving reader for track files (see below).
  - `theme.py` – **design tokens**: every colour/line/type size of the
    graphic in one place; `[style] theme` picks a theme.
  - `plotting.py` – the layout engine described above.
  - `basemap.py` – vector coastline renderer (crisp land at any DPI).
  - `cone.py`, `geo.py`, `ace.py`, `landfall.py`, `ports.py` – math & data.
  - `feature_manager.py` – auto-runs plugins from `features/`.
- `features/` – pluggable add-ons (`_TEMPLATE.py` to copy).
- `assets/geo/land.geojson` – GSHHS high-res coastline (public domain,
  simplified & clipped to the region; rebuild via
  `scripts/make_coastline.py`).
- `assets/geo/borders.geojson` – country boundaries, 1:50 m scale (public
  domain; rebuild via `scripts/make_borders.py`).
- `data/` – your track files. `output/plots/` – generated PNGs.

## Mobile (Pydroid 3)

The whole graphic is generated offline on the phone: no network calls, no
extra dependencies beyond `matplotlib / numpy / pandas / scipy`, and both
geo assets ship inside the repo (~1.1 MB total).  The `scripts/` folder is
only needed on a laptop to *rebuild* those assets from the public-domain
source databases — normal use never touches it.

## Quick start

```bash
./run.sh                        # interactive wizard
./run.sh data/06B.txt           # one-shot
python main.py data/06B.txt --dpi 200 -n my_name
```

## Data files — format v2, titles + `|` columns

```
NAME 06B                 <- optional; the file name works too
KIND INVEST              <- optional; auto-guessed otherwise

OBS
TIME (UTC) | LAT | LON | WIND (KT) | PRESSURE (HPA)
2026-09-19 00:00 | 13.00 | 95.00 | 10 | 1007
FORECAST
TIME (UTC) | LAT | LON | WIND (KT) | R24 | R34 | R64
2026-09-22 12:00 | 17.50 | 85.50 | 30 | 1.3 | - | -
```

Every section can start with a column-TITLE row (so each number is
labelled) and the columns are split by a `|` vertical line — tidy to type,
no wide padding to line up.  Fixed column order
(`time lat lon wind [pressure | r24 r34 r64]`), spaces or commas work too,
`-` for "none", `#` comments allowed — see `data/README.md`.
Older legacy files (name header + column headers + `=== FORECAST ===`)
still load unmodified, in any separator/case/synonym style.

## Config (config.ini)

`[plot]` numbers: `buffer`, `ucr`, `minlat_offset`, `maxlat_offset`,
`output_dpi`, `approach_radius`, `approach_ports`, `forecast_tz_offset`,
`forecast_table_max_cols`, `forecast_table_min_fontsize`,
`wind_radius_pad`, and the map coverage `min_lat/max_lat/min_lon/max_lon` (the bundled regional coverage is 20–130°E and 25°S–45°N).

Toggles (1/0): `show_cone`, `show_legend` (MAP KEY card), `show_ports`,
`show_approach_table` (NEAREST 4 PORTS card),
`show_movement_table` + `show_max_wind_boxes` + `show_ace_box`
(STORM SUMMARY stats), `show_forecast_table` (bottom band table),
`show_forecast_key` (key strip above it), `show_landfall`, `show_footer`,
`show_grid`, `show_scale_bar`,
`organize_by_year`, `full_track_extent`, `wind_radius_extent`.

`[style]`: `theme` (design tokens, `cyclone/theme.py`), `brand_logo`
(local PNG/JPG path), `brand_name` (fallback text), `date_format`,
`footer_text`, `forecast_time_label`,
`forecast_speed_label`, `forecast_speed_unit`, `forecast_speed_mode`
(`wind` = forecast wind km/h, `motion` = translation speed), `output_name`
(`{name}` placeholder supported).

Every key is optional — delete anything you do not use and the defaults
apply.

### Header logo and timestamps

Use your own logo by placing it in `assets/` and setting:

```ini
[style]
brand_logo = assets/xp_weather_logo.png
brand_name = XP WEATHER
```

PNG (including transparency) and JPG are supported. Relative paths resolve
from the project root, not the current working directory; absolute paths
also work. The logo keeps its original aspect ratio in a compact top-left
slot. No logo is bundled: until you supply one, an empty or unreadable
`brand_logo` falls back to `brand_name` (a bad path also prints a warning).
No network access or additional runtime dependency is needed.

The issue card uses the **last observation's time**, not the time you run
the program, with separate, spaced lines such as:

```text
ISSUED 06Z, 23 SEP 2026 UTC
LOCAL 12PM, 23 SEP 2026 (+6H)
```

`forecast_tz_offset` controls local time, including date/year rollover.
Whole hours omit `:00`; nonzero minutes remain visible, e.g. `11:30AM`.

### Readable sidebar and map labels

- **STORM SUMMARY** replaces AT A GLANCE.
- **MAP KEY** uses larger, dark labels and measured line wrapping. The
  duplicate **TRACK & AREAS** section is removed; the forecast key strip
  remains controlled by `show_forecast_key`.
- **NEAREST 4 PORTS** shows the actual row count in its title (so changing
  `approach_ports` updates the number). All three cards are packed above
  the footer, including their gaps and shadows.
- The footer and observed/forecast time range are bold. Latitude tick
  labels are bold too, with a tightly measured left gutter to avoid clipping.
- The compact kilometre scale is sized from its labelled distance at the
  map's centre latitude, rather than using a fixed, oversized bar.
- The base coastline is near-black; short shoreline sections inherit the
  exact risk colour of their nearby port marker and label (High = magenta,
  Medium = amber). The extra map legend text is omitted to keep the map clean.

## Pluggable features (features/)

Copy `features/_TEMPLATE.py` to a new name and implement
`run_feature(context)`; it auto-runs after each plot with the context:
`cyclone_name`, `track_obs`, `track_for`, `is_invest`, `landfall`,
`approaches`, `map_file`, `output_image`, `plots_dir`, `outputs_dir`.
A crashing feature is reported and the rest continue.

## Tests

```bash
python -m pip install -r requirements.txt pytest
MPLBACKEND=Agg python -m pytest -q
```

Tests cover issue/local time formatting, logo fitting and fallback, map-key
readability, sidebar containment, scale geometry, and unclipped latitude labels.

## License

MIT — see [LICENSE](LICENSE).

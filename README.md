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
+--------------------------------------------------------------+
|  HEADER   brand chip · storm title · validity · issued card  |
+---------------------------------------------+----------------+
|                                             |  SIDEBAR       |
|  MAP                                        |  MAP KEY       |
|  observed + forecast track, cone,           |  AT A GLANCE   |
|  wind radii, ports, landfall,               |  NEAREST PORTS |
|  collision-free label chips,                |  AI OVERLAYS   |
|  scale bar, north arrow                     |                |
+---------------------------------------------+----------------+
|  BAND   map key strip · forecast table (TIME / WIND rows)    |
+--------------------------------------------------------------+
|  FOOTER   wind · pressure · updated          © brand         |
+--------------------------------------------------------------+
```

Why it stays clean, whatever the data:

* **Zones are separate axes** — cards live outside the map, so a table can
  never cover a wind ring and a legend can never cover the coast.
* **Everything is dynamic** — a card, legend section, table column or label
  appears only when the matching datum exists (no forecast → no cone card;
  no landfall → an "Unknown · no landfall" risk row; 12 forecast steps →
  the table thins itself to `forecast_table_max_cols` columns; …).
* **The map window is solved from what is drawn** — every wind-radius ring
  and the cone fit fully inside the frame, clipped to the basemap
  coverage, so nothing is cut off and no empty ocean is wasted.
* **The coast is vector, not a raster image** — GSHHS high-res shoreline
  drawn as theme-coloured land with a soft shallow-water halo, razor-sharp
  at every zoom and DPI.
* **Labels are collision-solved** — every chip (forecast points, NOW,
  landfall, ports, AI) is placed in the first free slot around its marker;
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
- `features/` – pluggable add-ons (`_TEMPLATE.py` to copy); `ailoc.py`
  (k-NN landfall "AI Position"), `aibc.py` (climatology bias track).
- `assets/geo/land.geojson` – GSHHS high-res coastline (public domain,
  simplified & clipped to the region; rebuild via
  `scripts/make_coastline.py`). `assets/*.csv` – AI reference data.
- `data/` – your track files. `output/plots/` – generated PNGs.

## Quick start

```bash
./run.sh                        # interactive wizard
./run.sh data/06B.txt           # one-shot
python main.py data/06B.txt --show-ai --dpi 200 -n my_name
```

## Data files — format v2, the least typing possible

```
NAME 06B                 <- optional; the file name works too
KIND INVEST              <- optional; auto-guessed otherwise

OBS
2026-09-19 00:00   13.00   95.00   10   1007
FORECAST
2026-09-22 12:00   17.50   85.50   30   1.3    -      -
```

Fixed column order (`time lat lon wind [pressure | r24 r34 r64]`), spaces
or commas, `-` for "none", `#` comments allowed — see `data/README.md`.
Older legacy files (name header + column headers + `=== FORECAST ===`)
still load unmodified, in any separator/case/synonym style.

## Config (config.ini)

`[plot]` numbers: `buffer`, `ucr`, `minlat_offset`, `maxlat_offset`,
`output_dpi`, `approach_radius`, `approach_ports`, `forecast_tz_offset`,
`forecast_table_max_cols`, `forecast_table_min_fontsize`,
`wind_radius_pad`, and the map coverage `min_lat/max_lat/min_lon/max_lon`.

Toggles (1/0): `show_cone`, `show_legend` (MAP KEY card), `show_ports`,
`show_approach_table` (NEAREST PORTS card),
`show_movement_table` + `show_max_wind_boxes` + `show_ace_box` (AT A
GLANCE stats), `show_forecast_table` (bottom band table),
`show_forecast_key` (key strip above it), `show_landfall`, `show_footer`,
`show_ai_position`, `show_bias_track`, `show_grid`, `show_scale_bar`,
`organize_by_year`, `full_track_extent`, `wind_radius_extent`.

`[style]`: `theme` (design tokens, `cyclone/theme.py`), `brand_name`
(header chip), `date_format`, `footer_text`, `forecast_time_label`,
`forecast_speed_label`, `forecast_speed_unit`, `forecast_speed_mode`
(`wind` = forecast wind km/h, `motion` = translation speed), `output_name`
(`{name}` placeholder supported).

Every key is optional — delete anything you do not use and the defaults
apply.

## Pluggable features (features/)

Copy `features/_TEMPLATE.py` to a new name and implement
`run_feature(context)`; it auto-runs after each plot with the context:
`cyclone_name`, `track_obs`, `track_for`, `is_invest`, `landfall`,
`approaches`, `map_file`, `output_image`, `plots_dir`, `outputs_dir`.
A crashing feature is reported and the rest continue.

## License

MIT — see [LICENSE](LICENSE).

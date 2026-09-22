# Cyclone Track & Cone Plotter (Pydroid3-ready, Pluggable Features)

This project reads combined observed/forecast cyclone files and produces
an NHC-style track and uncertainty cone graphic.

It is designed so that:
- You only change **config.ini** and files in **data/** and **features/**.
- You normally never need to edit any `.py` file.

## Structure

- `main.py` – entry point. Run with no arguments for the interactive wizard,
  or pass a data file for a one-shot run (see *Usage*).
- `config.ini` – editable configuration (no need to touch Python files).
- `run.sh` – convenience launcher (installs dependencies on first run).
- `cyclone/` – core logic:
  - `cli.py` – interactive wizard + command-line interface.
  - `config.py` – reads settings from `config.ini`.
  - `data_loader.py` – reads the combined text file.
  - `cone.py` – builds the smooth NHC-style cone polygon.
  - `geo.py` – distance, bearing & direction helpers.
  - `ace.py` – ACE (Accumulated Cyclone Energy) calculator.
  - `landfall.py` – landfall estimation, per-port closest approach and the
    port table (current distance & direction of the centre from each port).
  - `plotting.py` – all Matplotlib plotting code (uses toggles from `config.ini`).
  - `ports.py` – port list used for the map markers and the port table
    (Bay of Bengal, Sri Lanka, Myanmar, Thailand/Malaysia, Sumatra — edit
    as you like; add a `"Name": (lat, lon)` line and it just works).
  - `feature_manager.py` – automatically loads feature modules from `features/`.
- `features/` – pluggable feature modules:
  - `_TEMPLATE.py` – copy-and-rename starter for your own feature.
  - `summary.py` – generates a PDF summary report per cyclone.
  - `ailoc.py` – k-NN landfall location prediction ("AI Position" star).
  - `aibc.py` – climatology-guided bias correction ("AI-BC Track").
  - `ai.py` – simple static bias fallback.
- `assets/` – background map (`Map.png`) and AI training CSVs
  (`climo.csv`, `climo_bias.csv`).
- `data/` – put all your cyclone data text files here (sub-folders OK,
  e.g. `data/2025/ditwah.txt`). See `data/README.md` for the file format.
- `output/` – generated results: `output/plots/` (PNG) and `output/files/` (PDF).

## Quick start

Desktop / Termux (installs dependencies automatically on first run):

```bash
./run.sh                      # interactive wizard
./run.sh data/2025/ditwah.txt # one-shot: no questions asked
```

Or manually:

```bash
pip install -r requirements.txt
python main.py
```

Pydroid 3 (Android): copy this folder to your device, open `main.py` in
Pydroid 3 and run it. You'll see a numbered list of data files (newest
first) — type the number and press Enter. Pressing Enter at any prompt
keeps the `config.ini` value.

## Command-line options

```
python main.py data/2025/montha.txt --buffer 2.5 --ucr 0.25
python main.py data/95B.txt --show-ai          # force AI overlays on
python main.py data/95B.txt --no-features      # skip the plugins
python main.py data/95B.txt --dpi 200 -o /tmp  # custom DPI / output folder
python main.py data/95B.txt -n 95B_ForecastTable   # custom PNG name
python main.py data/95B.txt --full-track   # keep the whole observed
                                           # track in frame as well
python main.py --list                          # list data files and exit
```

Every option defaults to whatever `config.ini` says, so the CLI is only
needed for per-run overrides and automation.

## Config toggles (config.ini)

Under `[plot]` you can set:

- `buffer`, `ucr` – padding and cone growth.
- `min_lat`, `max_lat`, `min_lon`, `max_lon` – map extent.
- `minlat_offset`, `maxlat_offset` – fine tuning vertical zoom.
- `output_dpi` – PNG resolution.
- `organize_by_year` – save plots into `output/plots/<year>/`.
- `full_track_extent` – `0` (default) zooms the map to the forecast track;
  `1` zooms out far enough to keep the whole **observed** track in frame
  too (the same as the `--full-track` command-line flag).

Toggles (1 = ON, 0 = OFF):

- `show_cone` – draw the uncertainty cone or not.
- `show_legend` – show/hide legend.
- `show_ports` – draw port markers. When an estimated landfall is available,
  each port marker is risk-coloured by distance to that landfall: red/high
  for `<100 km`, orange/medium for `100–300 km`, and green/low for `>300 km`.
  The `PORT RISK` key is placed horizontally at the top of the map.
- `show_port_table` / `show_approach_table` – both toggles show the *same*
  single port table (they are aliases, kept so older config.ini files keep
  working); set both to 0 to hide it.
- `show_movement_table` – show/hide movement info.
- `show_ace_box` – show/hide ACE box.
- `show_max_wind_boxes` – show/hide max observed/forecast boxes.
- `show_footer` – show/hide footer text.
- `show_ai_position` – show/hide the AI landfall position star.
- `show_bias_track` – show/hide the AI bias-corrected track.
- `show_landfall` – estimate & mark landfall (where/when the forecast
  track first crosses the coastline) with a red ✕ marker + a compact
  `LF-DD/HHZ` label placed bottom-left of the marker (it falls back to
  left / up-left / above / bottom-right if that would collide with a
  forecast time label).
- `show_approach_table` – the port table. Ports are *selected* by
  distance to the **landfall point** — the `approach_ports` ports nearest
  to where the storm is expected to come ashore — and the table then shows
  the situation **right now**: how far the current storm centre is from
  each of those ports and in which direction it lies, e.g.

  ```
  PORT        DIS      DIR
  Puri        450 km   SSE
  ```

  i.e. the centre is currently 450 km to the south-south-east of Puri.

- `show_forecast_table` – the bottom-centre **forecast table**: one column
  per forecast step between the port table and the movement table,

  ```
  Time (BST)   18/21SEP   06/22SEP   18/22SEP   ...
  Speed (KM)   46KM/H     56KM/H     56KM/H     ...
  ```

  `Time` is the synoptic time shifted by `forecast_tz_offset` hours
  (default **6** → BST; set it to `0` for UTC) in `DD/HHMMM` form. `Speed`
  is the **forecast wind intensity in km/h** (1 kt = 1.852 km/h), so the
  row always matches the knots printed on the forecast points:
  `25KT → 46KM/H`, `30KT → 56KM/H`, `35KT → 65KM/H`. Set
  `forecast_speed_mode = motion` in `[style]` if you would rather have the
  storm's translation speed (great-circle distance from the previous track
  point, the last observed fix for the first column, divided by the hours
  between the two) shown there instead. If a file has more steps than
  `forecast_table_max_cols` (default **8**), the steps in between are
  dropped evenly so the columns stay readable (first and last are always
  kept). The label texts and the unit come from `[style]`
  (`forecast_time_label`, `forecast_speed_label`, `forecast_speed_unit`).

- `show_forecast_key` – the key strip drawn directly on top of the
  forecast table: `Uncertainty Cone | Forecast Track | Landfall Est.`,
  left-aligned with the table and right-aligned with the movement table.
  While it is on, those three entries are left out of the `INTENSITY
  SCALE` legend on the right so nothing is listed twice; set it to `0` and
  they move back into the legend.

Numbers:

- `approach_radius` – ports farther than this many km from the track are
  skipped in the port table (default 800).
- `approach_ports` – how many of the landfall's nearest ports the table
  lists (default 4). When the forecast never reaches land there is no
  landfall to measure from, so the `approach_ports` ports the track comes
  closest to are listed instead.
- `forecast_tz_offset` – hours added to the UTC synoptic times before the
  forecast table prints them (default 6 = BST).
- `forecast_table_max_cols` – most columns the forecast table may use
  (default 8); extra steps are thinned out evenly.
- `forecast_table_min_fontsize` – smallest font the forecast table and its
  key strip may shrink to when the middle gap is narrow (default 6.8).

Under `[style]`:

- `date_format` – strftime format for title dates (default `%HZ, %d %b %Y`).
- `footer_text` – right-hand footer text on the map.
- `forecast_time_label`, `forecast_speed_label`, `forecast_speed_unit` –
  first cells of the two forecast-table rows and the speed unit
  (defaults `Time (BST)`, `Speed (KM)`, `KM/H`).
- `forecast_speed_mode` – `wind` (default) puts the forecast wind in km/h
  in the Speed row; `motion` puts the storm's translation speed there.
- `output_name` – name of the output PNG, **without** `.png`. Empty (the
  default) keeps the usual `<Name>_Track`. `{name}` is replaced by the
  cyclone/invest name, so `output_name = {name}_Track_v2` saves 95B as
  `95B_Track_v2.png`. The `-n/--name` command-line flag overrides it for a
  single run:

  ```bash
  python main.py data/95B.txt -n 95B_ForecastTable
  ```

## Pluggable features (features/ folder)

To add a new feature (for example `landfall.py`):

1. Go to the `features/` folder.
2. Copy `_TEMPLATE.py` and rename the copy, e.g. `landfall.py`.
3. Edit only that file and implement your logic inside:

   ```python
   from pathlib import Path

   def run_feature(context: dict):
       cyclone_name = context["cyclone_name"]
       track_obs = context["track_obs"]
       track_for = context["track_for"]
       output_dir = Path(context["output_dir"])
       # do calculations, save files, etc.
   ```

When you run `main.py`, **all** `.py` files in `features/` with a
`run_feature(context)` function will be auto-detected and executed
after the main PNG has been created (files starting with `_` are skipped).

You do **not** need to edit `main.py` or any core file to add new features.
If a feature crashes, it is reported and the rest continue.

### Port table sizing

The table is built from the measured text extents, not from a fixed size:
column widths come from the glyph widths of the actual port names, the box
is placed inside the map, and the font is shrunk (down to ~6.5 pt) if
needed. If a name were still too long it is shortened with `…`. So adding
a long name such as `Krishnapatnam` can never push text outside the table
box.

### Bottom row: port table · forecast table · movement table

The whole bottom row is aligned: the port table starts at the left edge,
the forecast table fills the gap in the middle (its left edge follows the
port table, its right edge follows the movement table) and its `Time` row
lines up with the movement table's rows. The forecast table is built from
the measured text extents too — it shrinks the font (down to
`forecast_table_min_fontsize`) and stretches its columns to fill exactly
the space between its two neighbours, so it can never run into them.

## Data format

See `data/README.md`. Short version: a `Cyclone Name:` (or `Invest Name:`)
header, observed rows, a `=== FORECAST ===` divider, then forecast rows.

## License

MIT — see [LICENSE](LICENSE).

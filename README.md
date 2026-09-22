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

Toggles (1 = ON, 0 = OFF):

- `show_cone` – draw the uncertainty cone or not.
- `show_legend` – show/hide legend.
- `show_ports` – draw port markers.
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

Numbers:

- `approach_radius` – ports farther than this many km from the track are
  skipped in the port table (default 800).
- `approach_ports` – how many of the landfall's nearest ports the table
  lists (default 4). When the forecast never reaches land there is no
  landfall to measure from, so the `approach_ports` ports the track comes
  closest to are listed instead.

Under `[style]`:

- `date_format` – strftime format for title dates (default `%HZ, %d %b %Y`).
- `footer_text` – right-hand footer text on the map.

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

## Data format

See `data/README.md`. Short version: a `Cyclone Name:` (or `Invest Name:`)
header, observed rows, a `=== FORECAST ===` divider, then forecast rows.

## License

MIT — see [LICENSE](LICENSE).

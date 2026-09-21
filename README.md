# Cyclone Track & Cone Plotter (Pydroid3-ready, Pluggable Features)

This project reads combined observed/forecast cyclone files and produces
an NHC-style track and uncertainty cone graphic.

It is designed so that:
- You only change **config.ini** and files in **data/** and **features/**.
- You normally never need to edit any `.py` file.

## Structure

- `main.py` – entry point. Run this from Pydroid3.
  - Shows all `.txt` files in `data/` and asks you to choose one.
- `config.ini` – editable configuration (no need to touch Python files).
- `cyclone/` – core logic:
  - `config.py` – reads settings from `config.ini`.
  - `data_loader.py` – reads the combined text file.
  - `cone.py` – builds the smooth NHC-style cone polygon.
  - `geo.py` – distance, bearing & direction helpers.
  - `ace.py` – ACE (Accumulated Cyclone Energy) calculator.
  - `plotting.py` – all Matplotlib plotting code (uses toggles from `config.ini`).
  - `feature_manager.py` – automatically loads feature modules from `features/`.
- `features/` – pluggable feature modules:
  - `__init__.py` – small helper docstring.
  - `example_feature.py` – example feature that writes a text summary.
- `TcCites.py` – sample Bay of Bengal ports (edit as you like).
- `data/` – put all your cyclone data text files here.
- `assets/Map.png` – background map image (replace with your own).

## Config toggles (config.ini)

Under `[plot]` you can set:

- `buffer`, `ucr` – padding and cone growth.
- `min_lat`, `max_lat`, `min_lon`, `max_lon` – map extent.
- `minlat_offset`, `maxlat_offset` – fine tuning vertical zoom.
- `output_dpi` – PNG resolution.

Toggles (1 = ON, 0 = OFF):

- `show_cone` – draw the uncertainty cone or not.
- `show_legend` – show/hide legend.
- `show_ports` – draw port markers.
- `show_port_table` – show/hide port distance table.
- `show_movement_table` – show/hide movement info.
- `show_ace_box` – show/hide ACE box.
- `show_max_wind_boxes` – show/hide max observed/forecast boxes.
- `show_footer` – show/hide footer text.

## Pluggable features (features/ folder)

To add a new feature (for example `landfall.py`):

1. Go to the `features/` folder.
2. Copy `example_feature.py` and rename the copy, e.g. `landfall.py`.
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
after the main PNG has been created.

You do **not** need to edit `main.py` or any core file to add new features.

## Usage in Pydroid3

1. Copy this entire folder to your device (e.g. under `Pydroid3/scripts`).
2. Open Pydroid3, browse to `main.py` and open it.
3. Run `main.py`.
4. You'll see a numbered list of data files from `data/`.
   Type the number of the file you want and press Enter.
5. The script will:
   - Generate `<CYCLONE_NAME>_Track.png`
   - Run all feature modules from `features/` (e.g. `example_feature.py`).

To change behaviour:
- Edit `config.ini` for plotting options and toggles.
- Add or edit `.py` files in `features/` for new features.

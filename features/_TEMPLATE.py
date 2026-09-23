"""
TEMPLATE — copy this file, rename it (e.g. my_feature.py), and edit run_feature().

Every .py file in features/ (except names starting with "_") is auto-discovered.
If it defines run_feature(context), it runs automatically after each plot.

The output is image-only: save any extra images next to the main plot in
plots_dir (do not create new output sub-folders).

The context dict contains:
    cyclone_name : str
    track_obs    : pandas.DataFrame  (tnd, Latitude, Longitude, Intensity, Pressure)
    track_for    : pandas.DataFrame  (tnd, Latitude, Longitude, Intensity, WindR24/34/64)
    is_invest    : bool
    landfall     : dict | None  (landfall estimate returned by the plotter)
    approaches   : list  (port-approach rows returned by the plotter)
    map_file     : str  -> assets/geo/land.geojson (vector coast)
    output_image : str  -> the PNG that was just generated
    plots_dir    : str  -> output/plots  (save any extra images here)
    outputs_dir  : str  -> output/
"""
from pathlib import Path


def run_feature(context: dict):
    cyclone_name = context["cyclone_name"]
    track_obs = context["track_obs"]
    track_for = context["track_for"]
    plots_dir = Path(context["plots_dir"])

    # Example: write an extra image next to the main plot.
    out_file = plots_dir / f"{cyclone_name}_example.png"
    print(f"[EXAMPLE] would write {out_file}")
    print(f"[EXAMPLE] obs={len(track_obs)} fcst={len(track_for)} points")

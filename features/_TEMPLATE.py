"""
TEMPLATE — copy this file, rename it (e.g. my_feature.py), and edit run_feature().

Every .py file in features/ (except names starting with "_") is auto-discovered.
If it defines run_feature(context), it runs automatically after each plot.

The context dict contains:
    cyclone_name : str
    track_obs    : pandas.DataFrame  (tnd, Latitude, Longitude, Intensity, Pressure)
    track_for    : pandas.DataFrame  (tnd, Latitude, Longitude, Intensity, WindR24/34/64)
    is_invest    : bool
    output_dir   : str  -> output/files   (put generated text/PDF files here)
    map_file     : str  -> assets/Map.png
    output_image : str  -> the PNG that was just generated
"""
from pathlib import Path


def run_feature(context: dict):
    cyclone_name = context["cyclone_name"]
    track_obs = context["track_obs"]
    track_for = context["track_for"]
    output_dir = Path(context["output_dir"])

    out_file = output_dir / f"{cyclone_name}_example.txt"
    with open(out_file, "w") as f:
        f.write(f"Example feature output for {cyclone_name}\n")
        f.write(f"Observed points : {len(track_obs)}\n")
        f.write(f"Forecast points : {len(track_for)}\n")

    print(f"[EXAMPLE] wrote {out_file}")

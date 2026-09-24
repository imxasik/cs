#!/usr/bin/env python3
"""
Generate web-friendly JSON for the interactive viewer.
Reads data/ files and outputs web_data.json + geojson for Leaflet.
"""
import json
import math
from pathlib import Path
from cyclone.data_loader import process_combined_cyclone_data
from cyclone.cone import create_nhc_cone
from cyclone import config as cfg

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output" / "web"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def to_geojson_track(df, name):
    if df is None or len(df)==0:
        return {"type": "Feature", "properties": {"name": name}, "geometry": {"type": "LineString", "coordinates": []}}
    coords = [[float(lon), float(lat)] for lat, lon in zip(df["Latitude"], df["Longitude"])]
    return {
        "type": "Feature",
        "properties": {"name": name},
        "geometry": {"type": "LineString", "coordinates": coords}
    }

def process_file(fpath: Path):
    cname, obs, for_, is_invest = process_combined_cyclone_data(str(fpath))
    # cone
    cone_coords = []
    try:
        if len(obs)>0 and len(for_)>=2:
            ext_lon = [float(obs["Longitude"].iloc[-1])] + list(for_["Longitude"].values)
            ext_lat = [float(obs["Latitude"].iloc[-1])] + list(for_["Latitude"].values)
            cone_pts, slon, slat = create_nhc_cone(ext_lon, ext_lat, 0.0, cfg.UCR)
            cone_coords = [[float(lon), float(lat)] for lon, lat in cone_pts]
    except Exception as e:
        print(f"Cone failed for {fpath}: {e}")
    
    # wind radii as circles approximated by polygons
    wind_radii = []
    if for_ is not None and len(for_)>0:
        for i in range(len(for_)):
            lat = float(for_["Latitude"].iloc[i])
            lon = float(for_["Longitude"].iloc[i])
            for col in ["WindR34", "WindR64"]:
                try:
                    r = float(for_[col].iloc[i])
                except Exception:
                    continue
                if not math.isfinite(r) or r<=0:
                    continue
                # approximate circle with 32 points
                pts = []
                for ang in range(0, 360, 12):
                    rad = math.radians(ang)
                    # simple equirectangular approx for small radii
                    dlat = r * math.cos(rad)
                    dlon = r * math.sin(rad) / max(0.3, math.cos(math.radians(lat)))
                    pts.append([lon+dlon, lat+dlat])
                pts.append(pts[0])
                wind_radii.append({
                    "type": "Feature",
                    "properties": {"radius_col": col, "radius_deg": r, "wind": float(for_["Intensity"].iloc[i]),
                                   "time": str(for_["tnd"].iloc[i])},
                    "geometry": {"type": "Polygon", "coordinates": [pts]}
                })
    
    # obs points
    obs_features = []
    if obs is not None and len(obs)>0:
        for _, row in obs.iterrows():
            obs_features.append({
                "type": "Feature",
                "properties": {"type": "obs", "wind": float(row["Intensity"]), "time": str(row["tnd"]),
                               "pressure": float(row["Pressure"]) if not (row["Pressure"] is None or str(row["Pressure"])=='nan') else None},
                "geometry": {"type": "Point", "coordinates": [float(row["Longitude"]), float(row["Latitude"])]}
            })
    for_features = []
    if for_ is not None and len(for_)>0:
        for _, row in for_.iterrows():
            for_features.append({
                "type": "Feature",
                "properties": {"type": "forecast", "wind": float(row["Intensity"]), "time": str(row["tnd"])},
                "geometry": {"type": "Point", "coordinates": [float(row["Longitude"]), float(row["Latitude"])]}
            })
    
    return {
        "name": cname,
        "is_invest": bool(is_invest),
        "file": fpath.name,
        "obs_track": to_geojson_track(obs, "obs"),
        "for_track": to_geojson_track(for_, "forecast"),
        "cone": {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [cone_coords]}} if cone_coords else None,
        "wind_radii": {"type": "FeatureCollection", "features": wind_radii},
        "obs_points": {"type": "FeatureCollection", "features": obs_features},
        "for_points": {"type": "FeatureCollection", "features": for_features},
        "bounds": {
            "min_lat": float(min(list(obs["Latitude"])+list(for_["Latitude"]))) if (obs is not None and len(obs) and for_ is not None and len(for_)) else 0,
            "max_lat": float(max(list(obs["Latitude"])+list(for_["Latitude"]))) if (obs is not None and len(obs) and for_ is not None and len(for_)) else 0,
            "min_lon": float(min(list(obs["Longitude"])+list(for_["Longitude"]))) if (obs is not None and len(obs) and for_ is not None and len(for_)) else 0,
            "max_lon": float(max(list(obs["Longitude"])+list(for_["Longitude"]))) if (obs is not None and len(obs) and for_ is not None and len(for_)) else 0,
        }
    }

all_data = []
for txt in sorted(DATA_DIR.glob("*.txt")):
    try:
        d = process_file(txt)
        all_data.append(d)
        # write individual
        (OUTPUT_DIR / f"{txt.stem}.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
        print(f"Generated {txt.stem}.json")
    except Exception as e:
        print(f"Failed {txt}: {e}")

(OUTPUT_DIR / "all.json").write_text(json.dumps(all_data, indent=2), encoding="utf-8")
print(f"Generated all.json with {len(all_data)} storms")

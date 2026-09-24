#!/usr/bin/env python3
"""
Rebuild assets/geo/states.geojson — extracts polygon boundary rings as LineStrings
for border rendering, while keeping name and centroid properties for labels.
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from pathlib import Path
import shapefile

REGION = (55.0, 125.0, -25.0, 45.0)  # lon_min, lon_max, lat_min, lat_max
OUT = Path(__file__).resolve().parent.parent / "assets" / "geo" / "states.geojson"

URL = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip"

def main():
    print("Downloading Natural Earth Admin-1 data...")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        content = resp.read()

    print("Processing shapes into border lines and centroids...")
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        shp = io.BytesIO(zf.read([n for n in zf.namelist() if n.endswith(".shp")][0]))
        dbf = io.BytesIO(zf.read([n for n in zf.namelist() if n.endswith(".dbf")][0]))
        shx = io.BytesIO(zf.read([n for n in zf.namelist() if n.endswith(".shx")][0]))

        reader = shapefile.Reader(shp=shp, dbf=dbf, shx=shx)
        fields = [f[0] for f in reader.fields[1:]]

        features = []
        for sr in reader.shapeRecords():
            rec = dict(zip(fields, sr.record))
            name = rec.get("name") or rec.get("NAME") or rec.get("woe_name")
            if not name:
                continue

            shape = sr.shape
            bbox = shape.bbox  # [xmin, ymin, xmax, ymax]
            
            if bbox[2] < REGION[0] or bbox[0] > REGION[1] or bbox[3] < REGION[2] or bbox[1] > REGION[3]:
                continue

            points = shape.points
            if not points:
                continue
            
            # সেন্ট্রয়েড ক্যালকুলেশন
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)

            if not (REGION[0] <= cx <= REGION[1] and REGION[2] <= cy <= REGION[3]):
                continue

            # পলিগণের অংশগুলোকে LineString বাউন্ডারিতে রূপান্তর করা
            parts = list(shape.parts) + [len(points)]
            for i in range(len(parts) - 1):
                ring = points[parts[i]:parts[i+1]]
                if len(ring) < 2:
                    continue
                
                line_coords = [[round(p[0], 3), round(p[1], 3)] for p in ring]
                
                features.append({
                    "type": "Feature",
                    "properties": {
                        "name": name,
                        "longitude": round(cx, 3),
                        "latitude": round(cy, 3)
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": line_coords
                    }
                })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({
            "type": "FeatureCollection",
            "source": "Natural Earth Admin-1 Borders & Labels",
            "features": features
        }, fh, ensure_ascii=False, separators=(",", ":"))

    print(f"Successfully wrote {OUT}: {len(features)} boundary lines with labels.")

if __name__ == "__main__":
    main()

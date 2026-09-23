#!/usr/bin/env python3
"""
Rebuild assets/geo/land.geojson from the GSHHS high-resolution coastline.

GSHHS (Global Self-consistent Hierarchical High-resolution Shoreline) is
public domain (NOAA / Univ. of Hawaii).  The binary ships in the PyPI
package ``basemap-data-hires`` (mpl_toolkits/basemap_data/gshhs_h.dat plus
its text meta file), so this script works fully offline once that package
is installed:

    pip install basemap-data-hires
    python scripts/make_coastline.py

What it does
------------
1. reads the meta file (level, area, npts, south, north, byte-offset,
   byte-count, dateline flag) and the little-endian float32 (lon, lat)
   pairs it points at,
2. keeps level-1 (land) polygons whose bounding box touches the project
   map area (lon 55..125 E, lat -25..45 N),
3. simplifies them radially (0.008 deg ~ half a pixel at every zoom this
   project uses) and rounds to 5 decimals,
4. writes a small GeoJSON FeatureCollection that cyclone/basemap.py draws
   as crisp vector land at any DPI.

The result is a few hundred kB instead of the ~15 MB regional slice (and
~90 MB global) database, and replaces the old hand-made raster Map.png.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

try:
    from mpl_toolkits import basemap_data
    DATA = Path(list(basemap_data.__path__)[0])
except ImportError:
    raise SystemExit("pip install basemap-data-hires first")

REGION = (55.0, 125.0, -25.0, 45.0)      # lon_min, lon_max, lat_min, lat_max
TOL = 0.008                              # radial simplification, degrees
MIN_SPAN = 0.05                          # drop islets smaller than this (deg)
MIN_AREA = 10.0                            # drop islets smaller than this (km2)
OUT = Path(__file__).resolve().parent.parent / "assets" / "geo" / "land.geojson"


def clip_ring(ring, x0, x1, y0, y1):
    """Sutherland-Hodgman clip of a ring to the project region: continent
    scale polygons keep only the part this project can ever show."""
    def clip_edge(pts, inside, intersect):
        out = []
        for a, b in zip(pts, pts[1:] + pts[:1]):
            ia, ib = inside(a), inside(b)
            if ia:
                out.append(a)
            if ia != ib:
                out.append(intersect(a, b))
        return out

    def lerp(a, b, t):
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    for inside, inter in (
        (lambda p: p[0] >= x0, lambda a, b: lerp(a, b, (x0 - a[0]) / (b[0] - a[0]))),
        (lambda p: p[0] <= x1, lambda a, b: lerp(a, b, (x1 - a[0]) / (b[0] - a[0]))),
        (lambda p: p[1] >= y0, lambda a, b: lerp(a, b, (y0 - a[1]) / (b[1] - a[1]))),
        (lambda p: p[1] <= y1, lambda a, b: lerp(a, b, (y1 - a[1]) / (b[1] - a[1]))),
    ):
        ring = clip_edge(ring, inside, inter)
        if len(ring) < 3:
            return []
    return ring


def tol_for(area_km2):
    """Coarser simplification for continent-scale polygons: at the zooms
    this project uses their extra detail is sub-pixel anyway."""
    if area_km2 < 2000:
        return 0.006
    if area_km2 < 50000:
        return 0.010
    return 0.016


def simplify(pts, tol):
    out = [pts[0]]
    for p in pts[1:]:
        if abs(p[0] - out[-1][0]) >= tol or abs(p[1] - out[-1][1]) >= tol:
            out.append(p)
    if out[-1] != pts[-1]:
        out.append(pts[-1])
    return out


def main():
    dat = (DATA / "gshhs_h.dat").read_bytes()
    features = []
    for line in (DATA / "gshhsmeta_h.dat").read_text().splitlines():
        f = line.split()
        if len(f) < 8:
            continue
        level = int(f[0])
        npts = int(f[2])
        south, north = float(f[3]), float(f[4])
        offset, bytecount = int(f[5]), int(f[6])
        if level != 1:                       # land only (no lakes/ponds)
            continue
        if north < REGION[2] or south > REGION[3]:
            continue
        ring = np.frombuffer(dat[offset:offset + bytecount],
                             dtype="<f4").reshape(-1, 2).astype(np.float64)
        w, e = float(ring[:, 0].min()), float(ring[:, 0].max())
        if e < REGION[0] or w > REGION[1]:
            continue
        area = float(f[1])
        if area < MIN_AREA or ((e - w) < MIN_SPAN
                             and (north - south) < MIN_SPAN):
            continue
        ring = clip_ring([tuple(p) for p in ring], *REGION)
        if len(ring) < 4:
            continue
        pts = simplify(ring, tol_for(area))
        if len(pts) < 4:
            continue
        coords = [[round(lon, 3), round(lat, 3)] for lon, lat in pts]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        features.append({
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({"type": "FeatureCollection",
                   "source": "GSHHS high-res (public domain), simplified",
                   "features": features}, fh, separators=(",", ":"))
    n = sum(len(f["geometry"]["coordinates"][0]) for f in features)
    print(f"wrote {OUT}: {len(features)} polygons, {n} points, "
          f"{OUT.stat().st_size / 1024:.0f} kB")


if __name__ == "__main__":
    main()

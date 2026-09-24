#!/usr/bin/env python3
"""
Rebuild assets/geo/borders.geojson — country boundaries at 1:50 million
scale (the same scale class as Natural Earth 50m admin-0).

Source: the GMT/WDBII political boundary set shipped in the PyPI package
``basemap-data`` (mpl_toolkits/basemap_data/countries_i.dat, "intermediate"
= 1:50m scale, public domain).  It is used because it can be bundled
fully offline — important for Pydroid 3 phones, where runtime downloads
are not an option.

    pip install basemap-data
    python scripts/make_borders.py

What it does
------------
1. reads the meta file (npts, south, north, byte-offset, byte-count) and
   the little-endian float32 (lon, lat) pairs it points at,
2. keeps boundary lines whose bounding box touches the project map area
   (lon 20..130 E, lat -25..45 N) and clips them to that rectangle,
3. simplifies radially (0.006 deg) and rounds to 3 decimals,
4. writes a tiny GeoJSON FeatureCollection of LineStrings that
   cyclone/basemap.py draws as a thin dashed administrative boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import os

if os.environ.get("BORDERS_DATA_DIR"):
    DATA = Path(os.environ["BORDERS_DATA_DIR"])
else:
    try:
        from mpl_toolkits import basemap_data
        DATA = Path(list(basemap_data.__path__)[0])
    except ImportError:
        raise SystemExit("pip install basemap-data first "
                         "(or set BORDERS_DATA_DIR)")

REGION = (20.0, 130.0, -25.0, 45.0)      # lon_min, lon_max, lat_min, lat_max
TOL = 0.006                              # radial simplification, degrees
OUT = Path(__file__).resolve().parent.parent / "assets" / "geo" / "borders.geojson"


def _code(p, x0, x1, y0, y1):
    c = 0
    if p[0] < x0:
        c |= 1
    elif p[0] > x1:
        c |= 2
    if p[1] < y0:
        c |= 4
    elif p[1] > y1:
        c |= 8
    return c


def clip_line(pts, x0, x1, y0, y1):
    """Cohen-Sutherland clip of a polyline to a rectangle -> list of runs."""
    runs, cur = [], []
    for a, b in zip(pts, pts[1:]):
        ca, cb = _code(a, x0, x1, y0, y1), _code(b, x0, x1, y0, y1)
        p, q = a, b
        while True:
            if not (ca | cb):
                break
            if ca & cb:
                p = q = None
                break
            c = ca or cb
            if c & 8:
                t = ((y1 - p[1]) / (q[1] - p[1])) if q[1] != p[1] else 0
                x = p[0] + (q[0] - p[0]) * t
                new = (x, y1)
            elif c & 4:
                t = ((y0 - p[1]) / (q[1] - p[1])) if q[1] != p[1] else 0
                x = p[0] + (q[0] - p[0]) * t
                new = (x, y0)
            elif c & 2:
                t = ((x1 - p[0]) / (q[0] - p[0])) if q[0] != p[0] else 0
                y = p[1] + (q[1] - p[1]) * t
                new = (x1, y)
            else:
                t = ((x0 - p[0]) / (q[0] - p[0])) if q[0] != p[0] else 0
                y = p[1] + (q[1] - p[1]) * t
                new = (x0, y)
            if c == ca:
                p, ca = new, _code(new, x0, x1, y0, y1)
            else:
                q, cb = new, _code(new, x0, x1, y0, y1)
        if p is None:
            if cur:
                runs.append(cur)
                cur = []
            continue
        if not cur:
            cur = [p]
        elif abs(cur[-1][0] - p[0]) > 1e-9 or abs(cur[-1][1] - p[1]) > 1e-9:
            runs.append(cur)
            cur = [p]
        cur.append(q)
    if cur:
        runs.append(cur)
    return [r for r in runs if len(r) >= 2]


def simplify(pts, tol):
    out = [pts[0]]
    for p in pts[1:]:
        if abs(p[0] - out[-1][0]) >= tol or abs(p[1] - out[-1][1]) >= tol:
            out.append(p)
    if out[-1] != pts[-1]:
        out.append(pts[-1])
    return out


def main():
    dat = (DATA / "countries_i.dat").read_bytes()
    features = []
    for line in (DATA / "countriesmeta_i.dat").read_text().splitlines():
        f = line.split()
        if len(f) < 7:
            continue
        npts = int(f[2])
        south, north = float(f[3]), float(f[4])
        offset, bytecount = int(f[5]), int(f[6])
        if npts < 2 or north < REGION[2] or south > REGION[3]:
            continue
        ring = np.frombuffer(dat[offset:offset + bytecount],
                             dtype="<f4").reshape(-1, 2).astype(np.float64)
        w, e = float(ring[:, 0].min()), float(ring[:, 0].max())
        if e < REGION[0] or w > REGION[1]:
            continue
        for run in clip_line([tuple(p) for p in ring], *REGION):
            pts = simplify(run, TOL)
            if len(pts) < 2:
                continue
            coords = [[round(lon, 3), round(lat, 3)] for lon, lat in pts]
            features.append({
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "LineString", "coordinates": coords},
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({"type": "FeatureCollection",
                   "source": "GMT/WDBII political boundaries, 1:50m scale "
                             "(public domain), simplified",
                   "features": features}, fh, separators=(",", ":"))
    n = sum(len(f["geometry"]["coordinates"]) for f in features)
    print(f"wrote {OUT}: {len(features)} lines, {n} points, "
          f"{OUT.stat().st_size / 1024:.0f} kB")


if __name__ == "__main__":
    main()

"""
Vector basemap: crisp land polygons instead of the old raster Map.png.

The coastline lives in ``assets/geo/land.geojson`` (GSHHS high-res, public
domain, simplified & clipped to the project region — rebuild with
``scripts/make_coastline.py``).  Because it is vector, the coast stays
razor-sharp at every zoom level and every DPI, and the land/sea colours
come from the theme like everything else.

Drawing is cheap: polygons whose bounding box does not touch the current
map window are skipped, and the rest are drawn as two stacked PathPatches
(a soft shallow-water halo under a filled land patch with a thin coast
stroke).
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


@lru_cache(maxsize=4)
def load_land(geojson_path):
    """((Nx2 arrays), (bbox arrays)) for every land polygon, cached."""
    rings, boxes = [], []
    try:
        with open(geojson_path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return (), ()
    for feat in data.get("features", ()):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Polygon":
            continue
        coords = geom.get("coordinates") or []
        if not coords:
            continue
        ring = np.asarray(coords[0], dtype=float)
        if ring.ndim != 2 or len(ring) < 4:
            continue
        rings.append(ring)
        boxes.append((ring[:, 0].min(), ring[:, 0].max(),
                      ring[:, 1].min(), ring[:, 1].max()))
    return tuple(rings), tuple(boxes)


def draw_land(ax, theme, bounds, geojson_path):
    """
    Paint sea + vector land into `ax` for the window
    `bounds = (lon_min, lon_max, lat_min, lat_max)`.
    """
    ax.set_facecolor(theme["sea"])
    rings, boxes = load_land(geojson_path)
    if not rings:
        return 0
    lon_min, lon_max, lat_min, lat_max = bounds
    drawn = 0
    for ring, (w, e, s, n) in zip(rings, boxes):
        if e < lon_min or w > lon_max or n < lat_min or s > lat_max:
            continue
        path = MplPath(ring)
        # two-stage shallow-water shelf: a wide faint glow under a tighter
        # band, so the coast reads like a real chart's depth contours
        ax.add_patch(PathPatch(
            path, transform=ax.transData, fill=False,
            edgecolor=theme["coast_halo2"], linewidth=7.0, alpha=0.55,
            zorder=0.3, clip_on=True, joinstyle="round"))
        ax.add_patch(PathPatch(
            path, transform=ax.transData, fill=False,
            edgecolor=theme["coast_halo"], linewidth=2.8, alpha=0.75,
            zorder=0.4, clip_on=True, joinstyle="round"))
        # land fill + thin coast stroke
        ax.add_patch(PathPatch(
            path, transform=ax.transData, facecolor=theme["land"],
            edgecolor=theme["coast"], linewidth=0.8, alpha=1.0,
            zorder=0.5, clip_on=True, joinstyle="round"))
        drawn += 1
    return drawn


@lru_cache(maxsize=4)
def load_borders(geojson_path):
    """((Nx2 line arrays), (bbox tuples)) for every boundary, cached."""
    lines, boxes = [], []
    try:
        with open(geojson_path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return (), ()
    for feat in data.get("features", ()):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "LineString":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        line = np.asarray(coords, dtype=float)
        lines.append(line)
        boxes.append((line[:, 0].min(), line[:, 0].max(),
                      line[:, 1].min(), line[:, 1].max()))
    return tuple(lines), tuple(boxes)


def has_borders(geojson_path):
    return bool(load_borders(geojson_path)[0])


def draw_borders(ax, theme, bounds, geojson_path):
    """
    Thin dashed country-boundary lines (1:50m scale) for the window
    `bounds = (lon_min, lon_max, lat_min, lat_max)`.  Drawn as a single
    LineCollection so it stays cheap on phones.
    """
    lines, boxes = load_borders(geojson_path)
    if not lines:
        return 0
    lon_min, lon_max, lat_min, lat_max = bounds
    segs = [ln for ln, (w, e, s, n) in zip(lines, boxes)
            if not (e < lon_min or w > lon_max or n < lat_min or s > lat_max)]
    if not segs:
        return 0
    ax.add_collection(LineCollection(
        segs, colors=theme["border"], linewidths=0.85,
        linestyles=(0, (3.2, 2.2)), alpha=0.9, zorder=0.8,
        capstyle="round", joinstyle="round"))
    return len(segs)

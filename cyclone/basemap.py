"""
Vector basemap — gorgeous, crisp, super dynamic.

- Sea + land with soft halo, modern colors from theme
- Dynamic linewidths based on zoom factor
- Country + state borders with modern styling
- Cheap bbox culling for mobile performance
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

from .theme import dynamic_linewidth


@lru_cache(maxsize=8)
def load_land(geojson_path):
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


def draw_land(ax, theme, bounds, geojson_path, zf=1.0):
    ax.set_facecolor(theme["sea"])
    rings, boxes = load_land(geojson_path)
    if not rings:
        return 0
    lon_min, lon_max, lat_min, lat_max = bounds
    drawn = 0
    # dynamic line widths
    # A fine black primary coast stays crisp without turning dense delta
    # islands into a heavy dotted band at high DPI.
    halo_lw = 1.8 + 0.35 * zf
    coast_lw = 1.15 + 0.20 * zf
    for ring, (w, e, s, n) in zip(rings, boxes):
        if e < lon_min or w > lon_max or n < lat_min or s > lat_max:
            continue
        path = MplPath(ring)
        # double halo for gorgeous depth
        ax.add_patch(PathPatch(
            path, transform=ax.transData, fill=False,
            edgecolor=theme["coast_halo"], linewidth=halo_lw + 1.2, alpha=0.35,
            zorder=0.35, clip_on=True, joinstyle="round", capstyle="round"))
        ax.add_patch(PathPatch(
            path, transform=ax.transData, fill=False,
            edgecolor=theme["coast_halo"], linewidth=halo_lw, alpha=0.55,
            zorder=0.4, clip_on=True, joinstyle="round", capstyle="round"))
        # land fill + coast stroke — more modern, slightly transparent edge
        ax.add_patch(PathPatch(
            path, transform=ax.transData, facecolor=theme["land"],
            edgecolor=theme["coast"], linewidth=coast_lw, alpha=1.0,
            zorder=0.55, clip_on=True, joinstyle="round", capstyle="round"))
        drawn += 1
    return drawn


def _project_to_segment_km(point, a, b):
    """Return local-km distance and t for a lon/lat segment projection."""
    lat0 = np.radians((point[1] + a[1] + b[1]) / 3.0)
    sx, sy = 111.320 * np.cos(lat0), 110.574
    p = np.array([point[0] * sx, point[1] * sy])
    aa = np.array([a[0] * sx, a[1] * sy])
    bb = np.array([b[0] * sx, b[1] * sy])
    vec = bb - aa
    denom = float(np.dot(vec, vec))
    t = 0.0 if denom == 0 else float(np.dot(p - aa, vec) / denom)
    t = max(0.0, min(1.0, t))
    nearest = aa + t * vec
    return float(np.linalg.norm(p - nearest)), t


def draw_risk_coastline(ax, theme, bounds, geojson_path, ports, zf=1.0):
    """Highlight short coastline sections using the matching port-risk colour.

    The full coastline remains a crisp near-black reference.  Around each
    visible port, a restrained segment is overlaid with the same colour used
    by its marker and label.  This makes the risk relationship legible
    without turning an entire country shoreline into a loud colour band.
    """
    rings, boxes = load_land(geojson_path)
    if not rings or not ports:
        return 0
    lon_min, lon_max, lat_min, lat_max = bounds
    # Keep sections short and proportional to risk importance.
    half_km = {"high": 82.0, "medium": 62.0, "low": 44.0,
               "norisk": 30.0, "unknown": 24.0}
    drawn = 0
    for _name, plat, plon, risk in ports:
        if not (lon_min <= plon <= lon_max and lat_min <= plat <= lat_max):
            continue
        best = None
        for ring, (west, east, south, north) in zip(rings, boxes):
            if east < plon - 3.0 or west > plon + 3.0 or north < plat - 3.0 or south > plat + 3.0:
                continue
            for idx in range(len(ring) - 1):
                a, b = ring[idx], ring[idx + 1]
                distance, fraction = _project_to_segment_km((plon, plat), a, b)
                if best is None or distance < best[0]:
                    best = (distance, ring, idx, fraction)
        if best is None:
            continue
        _distance, ring, index, _fraction = best
        # A port can be a little inland/offshore; still colour its nearest
        # coast if it is within a sensible coastal association distance.
        if _distance > 90.0:
            continue
        colour = risk.get("color", theme["coast"])
        reach = half_km.get(risk.get("key", "unknown"), 24.0)
        start, end = index, index + 1
        distance = 0.0
        while start > 0 and distance < reach:
            a, b = ring[start - 1], ring[start]
            lat0 = np.radians((a[1] + b[1]) / 2.0)
            distance += float(np.hypot((b[0] - a[0]) * 111.320 * np.cos(lat0),
                                       (b[1] - a[1]) * 110.574))
            start -= 1
        distance = 0.0
        while end < len(ring) - 1 and distance < reach:
            a, b = ring[end], ring[end + 1]
            lat0 = np.radians((a[1] + b[1]) / 2.0)
            distance += float(np.hypot((b[0] - a[0]) * 111.320 * np.cos(lat0),
                                       (b[1] - a[1]) * 110.574))
            end += 1
        segment = ring[start:end + 1]
        if len(segment) < 2:
            continue
        line_width = dynamic_linewidth(3.0, zf)
        # White keyline separates risk colour from the black coast and the
        # warm land fill at every DPI.
        ax.plot(segment[:, 0], segment[:, 1], color="#ffffff",
                linewidth=line_width + 2.2, alpha=0.92, zorder=2.55,
                solid_capstyle="round", solid_joinstyle="round", clip_on=True)
        ax.plot(segment[:, 0], segment[:, 1], color=colour,
                linewidth=line_width, alpha=0.98, zorder=2.6,
                solid_capstyle="round", solid_joinstyle="round", clip_on=True)
        drawn += 1
    return drawn


@lru_cache(maxsize=8)
def load_borders(geojson_path):
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


def draw_borders(ax, theme, bounds, geojson_path, zf=1.0):
    lines, boxes = load_borders(geojson_path)
    if not lines:
        return 0
    lon_min, lon_max, lat_min, lat_max = bounds
    segs = [ln for ln, (w, e, s, n) in zip(lines, boxes)
            if not (e < lon_min or w > lon_max or n < lat_min or s > lat_max)]
    if not segs:
        return 0
    lw = theme.get("state_width", 0.8) + 0.3 * zf
    # country borders slightly bolder and more visible
    ax.add_collection(LineCollection(
        segs, colors=theme["border"], linewidths=max(0.9, lw),
        linestyles=(0, (4.0, 2.6)), alpha=0.92, zorder=2.1,
        capstyle="round", joinstyle="round"))
    return len(segs)


@lru_cache(maxsize=8)
def load_states(geojson_path):
    lines, boxes = [], []
    try:
        with open(geojson_path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return (), ()
    for feat in data.get("features", ()):
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        if gtype == "LineString":
            coords_list = [geom.get("coordinates") or []]
        elif gtype == "MultiLineString":
            coords_list = geom.get("coordinates") or []
        else:
            continue
        for coords in coords_list:
            if len(coords) < 2:
                continue
            line = np.asarray(coords, dtype=float)
            lines.append(line)
            boxes.append((line[:, 0].min(), line[:, 0].max(),
                          line[:, 1].min(), line[:, 1].max()))
    return tuple(lines), tuple(boxes)


def draw_states(ax, theme, bounds, geojson_path, zf=1.0):
    lines, boxes = load_states(geojson_path)
    if not lines:
        return 0
    lon_min, lon_max, lat_min, lat_max = bounds
    segs = [ln for ln, (w, e, s, n) in zip(lines, boxes)
            if not (e < lon_min or w > lon_max or n < lat_min or s > lat_max)]
    if not segs:
        return 0
    color = theme.get("state_border", "#cbd5e1")
    base_lw = theme.get("state_width", 0.6)
    lw = base_lw + 0.15 * zf
    ls = theme.get("state_style", ":")
    ax.add_collection(LineCollection(
        segs, colors=color, linewidths=lw,
        linestyles=ls, alpha=0.82, zorder=1.95,
        capstyle="round", joinstyle="round"))
    return len(segs)

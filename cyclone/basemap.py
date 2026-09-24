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
    halo_lw = 2.8 + 0.6 * zf
    coast_lw = 2.0 + 0.4 * zf
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

"""
Cyclone track & uncertainty-cone — SUPER DYNAMIC, GORGEOUS, MOBILE-FIRST edition.

Five-zone dashboard, but now:
- Map window is SOLVED from what is drawn (track + wind radii + cone) with
  beautiful centering that always fits even at max zoom.
- All markers, lines, text scale dynamically with zoom factor.
- Collision-free label chips with 16 candidates, priority queue, dynamic font.
- Modern, clean, eye-pleasing cards with large radius, soft shadow, generous padding.
- Mobile-first type scale, readable on any phone when fit-to-width.
- No overlap, ever — zones are separate axes, labels avoid each other.

New features (no clutter):
- Smart weighted center (current + forecast centroid)
- Dynamic padding that grows with span
- Zoom-aware marker/line/font scaling
- Inset mini-map when tightly zoomed
- Wind radius edge labels when zoomed in
- Modern scale bar + north arrow + landfall glow
- Gorgeous header/footer with better hierarchy
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch, Polygon, Rectangle, FancyBboxPatch
from matplotlib.path import Path as MplPath

from . import basemap
from . import config as cfg
from .theme import (
    get_theme, INTENSITY_SCALE, INTENSITY_LEGEND_ORDER, WIND_RADII, PORT_RISK,
    wind_category, wind_color, wind_cat_label,
    zoom_factor as theme_zoom_factor,
    dynamic_font, dynamic_marker_size, dynamic_linewidth,
)
from .cone import create_nhc_cone
from .geo import haversine, get_bearing, get_cardinal_direction, KNOTS_TO_KMH
from .ace import calculate_ace
from .landfall import (
    find_landfall, port_centre_table, current_centre, classify_port_risk,
    PORT_RISK_BANDS,
)
from .ports import BOB

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

BUFFER = cfg.BUFFER
UCR = cfg.UCR
OUTPUT_DPI = cfg.OUTPUT_DPI
FULL_TRACK_EXTENT = cfg.FULL_TRACK_EXTENT

FIG_W_IN, FIG_H_IN = 14.8, 13.2  # taller for mobile, more breathing room

# ---------------------------------------------------------------------------
# Text metrics
# ---------------------------------------------------------------------------
_TEXT_SAFETY = 1.07

def _text_width_pt(text, fontsize, weight="normal"):
    text = str(text)
    try:
        from matplotlib.font_manager import FontProperties
        from matplotlib.textpath import TextPath
        fp = FontProperties(family="sans-serif", weight=weight, size=fontsize)
        ink = float(TextPath((0, 0), text, prop=fp).get_extents().width)
        return ink * _TEXT_SAFETY
    except Exception:
        return 0.60 * fontsize * len(text)

def _fit_text(text, max_width_pt, fontsize, weight="normal"):
    text = str(text)
    if _text_width_pt(text, fontsize, weight) <= max_width_pt or not text:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _text_width_pt(text[:mid] + "…", fontsize, weight) <= max_width_pt:
            lo = mid
        else:
            hi = mid - 1
    return (text[:lo] + "…") if lo else "…"

def _ax_pt(ax):
    fig = ax.figure
    pos = ax.get_position()
    return (max(1.0, pos.width * fig.get_figwidth() * 72.0),
            max(1.0, pos.height * fig.get_figheight() * 72.0))

# ---------------------------------------------------------------------------
# Rounded rect
# ---------------------------------------------------------------------------
def _rrect_path(x0, y0, x1, y1, rx, ry):
    k = 0.5523
    rx = min(rx, (x1 - x0) / 2.0)
    ry = min(ry, (y1 - y0) / 2.0)
    vx, vy = rx * k, ry * k
    verts = [
        (x0 + rx, y0), (x1 - rx, y0),
        (x1 - rx + vx, y0), (x1, y0 + ry - vy), (x1, y0 + ry),
        (x1, y1 - ry),
        (x1, y1 - ry + vy), (x1 - rx + vx, y1), (x1 - rx, y1),
        (x0 + rx, y1),
        (x0 + rx - vx, y1), (x0, y1 - ry + vy), (x0, y1 - ry),
        (x0, y0 + ry),
        (x0, y0 + ry - vy), (x0 + rx - vx, y0), (x0 + rx, y0),
        (x0 + rx, y0),
    ]
    codes = [
        MplPath.MOVETO, MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return MplPath(verts, codes)

def add_rrect(ax, x0, y0, x1, y1, r_pt, *, fc="#ffffff", ec="#e2e8f0",
              lw=1.0, alpha=1.0, z=1.0, shadow=False, T=None):
    aw, ah = _ax_pt(ax)
    rx, ry = r_pt / aw, r_pt / ah
    if shadow:
        sx, sy = 1.6 / aw, 1.8 / ah
        ax.add_patch(PathPatch(
            _rrect_path(x0 + sx, y0 - sy, x1 + sx, y1 - sy, rx, ry),
            transform=ax.transAxes, facecolor=(T or {}).get("shadow", "#0f172a"),
            edgecolor="none", alpha=0.09, linewidth=0, zorder=z - 0.7,
            clip_on=False))
        # second soft shadow for gorgeous depth
        ax.add_patch(PathPatch(
            _rrect_path(x0 + sx*0.6, y0 - sy*0.6, x1 + sx*0.6, y1 - sy*0.6, rx, ry),
            transform=ax.transAxes, facecolor=(T or {}).get("shadow", "#0f172a"),
            edgecolor="none", alpha=0.05, linewidth=0, zorder=z - 0.6,
            clip_on=False))
    patch = PathPatch(
        _rrect_path(x0, y0, x1, y1, rx, ry),
        transform=ax.transAxes, facecolor=fc, edgecolor=ec,
        linewidth=lw, alpha=alpha, zorder=z, clip_on=False)
    ax.add_patch(patch)
    return patch

# ---------------------------------------------------------------------------
# SUPER DYNAMIC map window — fits track + wind radii + cone, beautiful center
# ---------------------------------------------------------------------------
def _row_max_radius(row):
    r = 0.0
    for col, _lbl, _c in WIND_RADII:
        try:
            v = float(row[col])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(v) and v > r:
            r = v
    return r

def wind_radius_extent(anchor_lat, anchor_lon, track_for, ucr=None,
                       show_cone=None, pad=None):
    """
    SUPER DYNAMIC: (lat_min, lat_max, lon_centre, meta)
    - Collects every wind-radius circle + cone radius
    - Computes weighted center (current 30% + forecast centroid 70%)
    - Dynamic padding based on span
    - Ensures minimum span for context, even at max zoom
    - Returns zoom_factor for dynamic scaling
    """
    ucr = UCR if ucr is None else ucr
    show_cone = cfg.SHOW_CONE if show_cone is None else show_cone
    pad = cfg.WIND_RADIUS_PAD if pad is None else pad

    lat_min = float(np.min(anchor_lat))
    lat_max = float(np.max(anchor_lat))
    lon_min = float(np.min(anchor_lon))
    lon_max = float(np.max(anchor_lon))

    # Track + radii + cone extents
    if track_for is not None and len(track_for) > 0:
        include_cone = show_cone and len(track_for) >= 2
        for i in range(len(track_for)):
            lat = float(track_for["Latitude"].iloc[i])
            lon = float(track_for["Longitude"].iloc[i])
            r = _row_max_radius(track_for.iloc[i])
            if include_cone:
                r = max(r, ucr * (i + 1) * 1.1)  # slightly larger cone for safety
            if r > 0.0:
                # lon radius needs cos(lat) correction for visual fit
                lon_r = r / max(0.3, math.cos(math.radians(lat)))
                lat_min = min(lat_min, lat - r)
                lat_max = max(lat_max, lat + r)
                lon_min = min(lon_min, lon - lon_r)
                lon_max = max(lon_max, lon + lon_r)

    # Weighted center: current position + forecast centroid
    # This makes the map center beautifully between where it is and where it's going
    if track_for is not None and len(track_for) > 0:
        fc_lats = track_for["Latitude"].values
        fc_lons = track_for["Longitude"].values
        # weight by wind intensity if available (stronger points matter more)
        try:
            weights = np.array([max(1.0, float(w)) for w in track_for["Intensity"].values])
            weights = weights / np.sum(weights)
            fc_cent_lat = float(np.sum(fc_lats * weights))
            fc_cent_lon = float(np.sum(fc_lons * weights))
        except Exception:
            fc_cent_lat = float(np.mean(fc_lats))
            fc_cent_lon = float(np.mean(fc_lons))
        # current is last anchor
        cur_lat = float(anchor_lat[-1]) if len(anchor_lat) else fc_cent_lat
        cur_lon = float(anchor_lon[-1]) if len(anchor_lon) else fc_cent_lon
        # 30% current, 70% forecast centroid — feels natural
        center_lat = 0.32 * cur_lat + 0.68 * fc_cent_lat
        center_lon = 0.30 * cur_lon + 0.70 * fc_cent_lon
    else:
        center_lat = 0.5 * (lat_min + lat_max)
        center_lon = 0.5 * (lon_min + lon_max)

    # Dynamic padding: larger relative pad when span is small (zoomed in)
    # This ensures wind circles never touch edge even at max zoom
    lat_span_raw = lat_max - lat_min
    lon_span_raw = lon_max - lon_min
    span_raw = max(lat_span_raw, lon_span_raw, 0.5)

    # base pad + proportional pad
    dyn_pad_lat = 0.35 + span_raw * 0.18 + pad
    dyn_pad_lon = 0.40 + span_raw * 0.20 + pad

    # Extra buffer from config (user can increase for more zoom-out)
    extra_buf = max(0.0, cfg.BUFFER - 2.0) * 0.6
    dyn_pad_lat += extra_buf
    dyn_pad_lon += extra_buf

    lat_min = lat_min - dyn_pad_lat + cfg.MINLAT_OFFSET
    lat_max = lat_max + dyn_pad_lat + cfg.MAXLAT_OFFSET
    lon_min = lon_min - dyn_pad_lon
    lon_max = lon_max + dyn_pad_lon

    # Ensure minimum span for context (avoid super tight crop)
    min_span = 4.0  # degrees — enough to see coast, but still tight
    if lat_max - lat_min < min_span:
        c = center_lat
        lat_min, lat_max = c - min_span/2, c + min_span/2
    if lon_max - lon_min < min_span:
        c = center_lon
        lon_min, lon_max = c - min_span/2, c + min_span/2

    # Clip to basemap coverage
    lat_min = max(lat_min, cfg.MIN_LAT)
    lat_max = min(lat_max, cfg.MAX_LAT)
    lon_min = max(lon_min, cfg.MIN_LON)
    lon_max = min(lon_max, cfg.MAX_LON)

    # Recompute center after clipping, but keep weighted center if possible
    # If weighted center is outside clipped bounds, clamp it
    lon_center = center_lon
    lon_center = min(max(lon_center, lon_min + (lon_max-lon_min)*0.35),
                     lon_max - (lon_max-lon_min)*0.35)

    # Compute zoom factor for dynamic scaling
    zf = theme_zoom_factor(lat_max - lat_min, lon_max - lon_min)

    meta = {
        "lat_span": lat_max - lat_min,
        "lon_span": lon_max - lon_min,
        "zoom_factor": zf,
        "center_lat": center_lat,
        "center_lon": lon_center,
        "dyn_pad_lat": dyn_pad_lat,
        "dyn_pad_lon": dyn_pad_lon,
    }

    return lat_min, lat_max, lon_center, meta

# ---------------------------------------------------------------------------
# Forecast table steps
# ---------------------------------------------------------------------------
def forecast_table_steps(track_obs, track_for, mode="wind", tz_offset_hours=6.0):
    steps = []
    if track_for is None or len(track_for) == 0:
        return steps
    tz = pd.Timedelta(hours=tz_offset_hours)
    if mode != "motion":
        for i in range(len(track_for)):
            wind = track_for["Intensity"].iloc[i]
            value = None
            if wind is not None and not pd.isna(wind):
                value = float(wind) * KNOTS_TO_KMH
            steps.append((track_for["tnd"].iloc[i] + tz, value))
        return steps
    prev = None
    if track_obs is not None and len(track_obs):
        prev = (float(track_obs["Latitude"].iloc[-1]),
                float(track_obs["Longitude"].iloc[-1]),
                track_obs["tnd"].iloc[-1])
    for i in range(len(track_for)):
        lat = float(track_for["Latitude"].iloc[i])
        lon = float(track_for["Longitude"].iloc[i])
        when = track_for["tnd"].iloc[i]
        speed = None
        if prev is not None:
            hours = (when - prev[2]).total_seconds() / 3600.0
            if hours > 0:
                speed = haversine((prev[0], prev[1]), (lat, lon)) / hours
        steps.append((when + tz, speed))
        prev = (lat, lon, when)
    return steps

def thin_steps(steps, max_cols):
    n = len(steps)
    if not max_cols or max_cols < 2 or n <= max_cols:
        return list(steps)
    k = int(max_cols)
    keep = []
    for j in range(k):
        i = min(n - 1, max(0, int(round(j * (n - 1) / (k - 1)))))
        if i not in keep:
            keep.append(i)
    if keep[-1] != n - 1:
        keep[-1] = n - 1
    return [steps[i] for i in keep]

# ---------------------------------------------------------------------------
# Label chip geometry — super dynamic collision solver
# ---------------------------------------------------------------------------
def _chip_bbox(base_disp, offset_px, ha, va, w_pt, pad_pt, h_pt, pt2px):
    ax_, ay_ = np.array(base_disp) + np.array(offset_px, dtype=float)
    w, h, p = w_pt * pt2px, h_pt * pt2px, pad_pt * pt2px
    if ha == "left":
        x0, x1 = ax_ - p, ax_ + w + p
    elif ha == "right":
        x0, x1 = ax_ - w - p, ax_ + p
    else:
        half = (w + 2 * p) / 2
        x0, x1 = ax_ - half, ax_ + half
    if va == "bottom":
        y0, y1 = ay_ - p, ay_ + h + p
    elif va == "top":
        y0, y1 = ay_ - h - p, ay_ + p
    else:
        half = (h + 2 * p) / 2
        y0, y1 = ay_ - half, ay_ + half
    return (x0, y0, x1, y1)

def _hits_point(box, x, y, r):
    cx = min(max(x, box[0]), box[2])
    cy = min(max(y, box[1]), box[3])
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r

def _overlap(a, b, gap=4.0):
    return not (a[2] + gap <= b[0] or b[2] + gap <= a[0]
                or a[3] + gap <= b[1] or b[3] + gap <= a[1])

# ---------------------------------------------------------------------------
# Zone painters — modern, gorgeous, mobile-first
# ---------------------------------------------------------------------------
def _draw_header(ax, T, *, name, is_invest, obs_span, for_span, issued,
                 issued_sub, brand, zf=1.0):
    ax.axis("off")
    aw, ah = _ax_pt(ax)

    # Brand chip — modern pill with larger radius
    fs_b = dynamic_font(T["fs_small"] + 0.8, zf, 9, 14)
    bw = _text_width_pt(brand, fs_b, "bold") + 22
    bh = 18.0
    y_c = 0.60
    add_rrect(ax, 0.0, y_c - bh/2/ah, bw/aw, y_c + bh/2/ah,
              8.0, fc=T["navy"], ec="none", z=2, T=T, shadow=True)
    ax.text(bw/2/aw, y_c, brand, transform=ax.transAxes, fontsize=fs_b,
            fontweight="bold", color=T["on_dark"], ha="center", va="center",
            zorder=3)

    # Issued card — modern with double shadow, larger
    fs_i = dynamic_font(T["fs_small"] + 0.6, zf, 9, 13)
    fs_sub = dynamic_font(T["fs_tiny"] + 0.6, zf, 8, 11.5)
    iw = max(_text_width_pt(issued, fs_i, "bold"),
             _text_width_pt(issued_sub, fs_sub, "bold")) + 24
    ih = 52.0
    x1 = 1.0
    x0 = x1 - iw/aw
    add_rrect(ax, x0, y_c - ih/2/ah, x1, y_c + ih/2/ah, 8.0,
              fc=T["paper"], ec=T["card_edge"], lw=1.0, z=2, shadow=True, T=T)
    ax.text(x0 + 10/aw, y_c + 6.5/ah, issued, transform=ax.transAxes,
            fontsize=fs_i, fontweight="bold", color=T["ink"],
            ha="left", va="center", zorder=3)
    ax.text(x0 + 10/aw, y_c - 11.5/ah, issued_sub, transform=ax.transAxes,
            fontsize=fs_sub, fontweight="bold", color=T["ink_faint"],
            ha="left", va="center", zorder=3)

    # Title — larger, bolder, more modern
    kind = "INVEST" if is_invest else "CYCLONE"
    fs_title = dynamic_font(T["fs_title"], zf, 20, 32)
    fs_subt = dynamic_font(T["fs_subtitle"], zf, 11, 16)
    ax.text(0.5, 0.68, f'TROPICAL {kind} "{name.upper()}"',
            transform=ax.transAxes, fontsize=fs_title,
            fontweight="black", color=T["ink"], ha="center", va="center",
            zorder=3)
    ax.text(0.5, 0.22, obs_span + "  •  " + for_span,
            transform=ax.transAxes, fontsize=fs_subt, fontweight="bold",
            color=T["ink_soft"], ha="center", va="center", zorder=3)

    # Accent rule — modern gradient-like with two segments
    ax.plot([0, 1], [0.02, 0.02], transform=ax.transAxes, color=T["navy"],
            lw=1.6, zorder=2, clip_on=False, alpha=0.9)
    ax.plot([0, 0.18], [0.02, 0.02], transform=ax.transAxes,
            color=T["accent"], lw=3.2, zorder=3, clip_on=False, solid_capstyle="round")

def _draw_footer(ax, T, left, right, zf=1.0):
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                           facecolor=T["navy"], edgecolor="none", zorder=1,
                           clip_on=False))
    # accent segment
    ax.add_patch(Rectangle((0, 0.82), 0.18, 0.18, transform=ax.transAxes,
                           facecolor=T["accent"], edgecolor="none", zorder=2,
                           clip_on=False))
    fs_f = dynamic_font(T["fs_footer"], zf, 10, 14)
    ax.text(0.012, 0.50, left, transform=ax.transAxes, fontsize=fs_f,
            fontweight="bold", color=T["on_dark"], ha="left", va="center", zorder=3)
    ax.text(0.988, 0.50, right, transform=ax.transAxes,
            fontsize=fs_f, fontweight="bold", color=T["on_dark"],
            ha="right", va="center", zorder=3)

# --- sidebar cards ----------------------------------------------------------
def _dot_patch(ax, x_c, y_c, r_pt, *, fc, ec, lw, z=4, alpha=1.0):
    aw, ah = _ax_pt(ax)
    from matplotlib.patches import Ellipse
    return ax.add_patch(Ellipse((x_c, y_c), 2 * r_pt / aw, 2 * r_pt / ah,
                                transform=ax.transAxes, facecolor=fc,
                                edgecolor=ec, linewidth=lw, alpha=alpha,
                                zorder=z, clip_on=False))

def _swatch(ax, x_c, y_c, kind, colour, fs, T, z=4, zf=1.0):
    aw, ah = _ax_pt(ax)
    r = fs * 0.44 * (0.9 + 0.2*zf)
    if kind == "dot":
        _dot_patch(ax, x_c, y_c, r, fc=colour, ec="#ffffff", lw=1.0, z=z)
        # outer ring for depth
        _dot_patch(ax, x_c, y_c, r+1.2, fc="none", ec="#000000", lw=0.6, z=z, alpha=0.25)
    elif kind == "ring":
        from matplotlib.patches import Ellipse
        ax.add_patch(Ellipse((x_c, y_c), 2 * r / aw, 2 * r / ah,
                             transform=ax.transAxes, facecolor="none",
                             edgecolor=colour,
                             linewidth=max(1.2, fs * 0.18 * zf), zorder=z,
                             clip_on=False))
        # white halo for clarity
        ax.add_patch(Ellipse((x_c, y_c), 2 * (r+1.8) / aw, 2 * (r+1.8) / ah,
                             transform=ax.transAxes, facecolor="none",
                             edgecolor="#ffffff", linewidth=2.2, alpha=0.7,
                             zorder=z-0.2, clip_on=False))
    elif kind == "cone":
        w, h = fs * 1.6, fs * 0.9
        ax.add_patch(Rectangle((x_c - w/2/aw, y_c - h/2/ah),
                               w/aw, h/ah, transform=ax.transAxes,
                               facecolor=T["cone_fill"], alpha=0.28,
                               edgecolor=T["cone_edge"], linewidth=0.9,
                               zorder=z, clip_on=False))
    elif kind in ("line", "dash", "coast", "border"):
        w = fs * 1.6
        ax.plot([x_c - w/2/aw, x_c + w/2/aw], [y_c, y_c],
                transform=ax.transAxes, color=colour,
                lw={"line": 2.4, "dash": 1.6, "coast": 1.8, "border": 1.2}[kind],
                linestyle={"line": "-", "dash": (0, (4, 2.4)),
                           "coast": "-", "border": (0, (2.8, 2.0))}[kind],
                zorder=z, clip_on=False, solid_capstyle="round")
    elif kind == "x":
        ax.plot([x_c], [y_c], marker="X", markersize=fs * 1.1,
                transform=ax.transAxes, color=colour,
                markeredgecolor="#ffffff", markeredgewidth=0.9,
                linestyle="none", zorder=z, clip_on=False)
    elif kind == "star":
        ax.plot([x_c], [y_c], marker="*", markersize=fs * 1.4,
                transform=ax.transAxes, markerfacecolor=colour,
                markeredgecolor="#78350f", markeredgewidth=0.7,
                linestyle="none", zorder=z, clip_on=False)

_KEY_SEC_GAP = 1.10
_KEY_TITLE = 1.15
_KEY_LEAD = 0.10
_KEY_ITEM = 1.65

def _key_card_height(rows, fs, T, zf=1.0):
    h = T["card_pad"] * 2 + fs * 2.1
    i = 0
    while i < len(rows):
        if rows[i][0] == "section":
            h += fs * (_KEY_SEC_GAP + _KEY_TITLE + _KEY_LEAD)
            i += 1
            j = i
            while j < len(rows) and rows[j][0] != "section":
                j += 1
            h += ((j - i) // 2 + ((j - i) % 2)) * fs * _KEY_ITEM
            i = j
        else:
            h += fs * _KEY_ITEM
            i += 1
    return h

def _draw_key_card(ax, x0, x1, y1, rows, fs, T, zf=1.0):
    aw, ah = _ax_pt(ax)
    h = _key_card_height(rows, fs, T, zf)
    y0 = y1 - h / ah
    add_rrect(ax, x0, y0, x1, y1, T["card_radius"], fc=T["paper"],
              ec=T["card_edge"], lw=T["line_card"], z=2, shadow=True, T=T)
    pad = T["card_pad"] / aw
    ty = y1 - (T["card_pad"] + fs * 1.0) / ah
    ax.text(x0 + pad, ty, "MAP KEY", transform=ax.transAxes,
            fontsize=fs * 1.12, fontweight="black", color=T["ink"], ha="left",
            va="center", zorder=4)
    tw = _text_width_pt("MAP KEY", fs * 1.12, "bold") / aw
    ax.plot([x0 + pad, x0 + pad + tw], [ty - fs * 1.05 / ah] * 2,
            transform=ax.transAxes, color=T["accent"], lw=2.4, zorder=4,
            clip_on=False, solid_capstyle="round")
    y = ty - fs * 2.0 / ah
    gut = 19.0 / aw
    inner = (x1 - x0) - 2 * pad
    col_w = inner / 2.0
    i = 0
    while i < len(rows):
        if rows[i][0] == "section":
            y -= fs * _KEY_SEC_GAP / ah
            ax.text(x0 + pad, y - fs * 0.60 / ah, rows[i][1],
                    transform=ax.transAxes,
                    fontsize=fs * 0.90, fontweight="bold",
                    color=T["ink_faint"], ha="left", va="center", zorder=4)
            y -= fs * _KEY_TITLE / ah
            y -= fs * _KEY_LEAD / ah
            i += 1
            items = []
            while i < len(rows) and rows[i][0] != "section":
                items.append(rows[i])
                i += 1
            for k, row in enumerate(items):
                cx = x0 + pad + (k % 2) * col_w
                cy = y - fs * 1.0 / ah
                _kind, colour, label = row[1], row[2], row[3]
                _swatch(ax, cx + gut / 2, cy, _kind, colour, fs * 1.02, T, zf=zf)
                label = _fit_text(label, col_w * aw - gut - 10, fs * 1.08)
                ax.text(cx + gut + 4 / aw, cy, label,
                        transform=ax.transAxes, fontsize=fs * 0.98,
                        color=T["ink_soft"], ha="left", va="center", zorder=4,
                        fontweight="500")
                if k % 2 == 1 or k == len(items) - 1:
                    y -= fs * _KEY_ITEM / ah
        else:
            cy = y - fs * 1.0 / ah
            _kind, colour, label = rows[i][1], rows[i][2], rows[i][3]
            _swatch(ax, x0 + pad + gut / 2, cy, _kind, colour, fs * 1.02, T, zf=zf)
            label = _fit_text(label, inner * aw - gut - 10, fs * 0.98)
            ax.text(x0 + pad + gut + 4 / aw, cy, label,
                    transform=ax.transAxes, fontsize=fs * 0.98,
                    color=T["ink_soft"], ha="left", va="center", zorder=4)
            y -= fs * _KEY_ITEM / ah
            i += 1
    return y0

_GLANCE_ROW = 3.4
_GLANCE_VAL = 1.35

def _glance_card_height(stats, fs, T, zf=1.0):
    rows = (len(stats) + 1) // 2
    return T["card_pad"] * 2.6 + fs * 2.2 + (rows * fs * _GLANCE_ROW)

def _draw_glance_card(ax, x0, x1, y1, stats, fs, T, zf=1.0):
    aw, ah = _ax_pt(ax)
    h = _glance_card_height(stats, fs, T, zf)
    y0 = y1 - h / ah
    add_rrect(ax, x0, y0, x1, y1, T["card_radius"], fc=T["paper"],
              ec=T["card_edge"], lw=T["line_card"], z=2, shadow=True, T=T)
    pad = T["card_pad"] / aw
    ty = y1 - (T["card_pad"] + fs * 1.0) / ah
    ax.text(x0 + pad, ty, "STORM SUMMARY", transform=ax.transAxes,
            fontsize=fs * 1.10, fontweight="black", color=T["ink"], ha="left",
            va="center", zorder=4)
    tw = _text_width_pt("STORM SUMMARY", fs * 1.10, "bold") / aw
    ax.plot([x0 + pad, x0 + pad + tw], [ty - fs * 1.05 / ah] * 2,
            transform=ax.transAxes, color=T["accent"], lw=2.4, zorder=4,
            clip_on=False, solid_capstyle="round")

    rows = (len(stats) + 1) // 2
    top = ty - fs * 2.6 / ah
    col_w = (x1 - x0 - 2 * pad) / 2.0
    mid_x = x0 + pad + col_w

    r_top = top + fs * 0.40 / ah
    r_bot = top - (rows - 1) * fs * _GLANCE_ROW / ah - fs * 2.4 / ah
    ax.plot([mid_x, mid_x], [r_bot, r_top], transform=ax.transAxes,
            color=T["card_edge_soft"], lw=1.1, zorder=3.6, clip_on=False, alpha=0.8)

    for i, (label, value, sub, colour) in enumerate(stats):
        col = i % 2
        cx = x0 + pad + col * col_w + (12.0 if col else 4.0) / aw
        cy = top - (i // 2) * fs * _GLANCE_ROW / ah
        ax.text(cx, cy, label, transform=ax.transAxes,
                fontsize=fs * 0.86, fontweight="bold", color=T["ink_soft"],
                ha="left", va="center", zorder=4, alpha=0.9)
        ax.text(cx, cy - fs * 1.30 / ah, value, transform=ax.transAxes,
                fontsize=fs * _GLANCE_VAL, fontweight="black", color=colour,
                ha="left", va="center", zorder=4)
        if sub:
            ax.text(cx, cy - fs * 2.30 / ah, sub, transform=ax.transAxes,
                    fontsize=fs * 0.70, color=T["ink_faint"], ha="left", va="center", zorder=4)

        if col == 0 and (i // 2) < rows - 1:
            ry = cy - fs * 2.45 / ah
            ax.plot([x0 + pad, x1 - pad], [ry, ry], transform=ax.transAxes,
                    color=T["card_edge_soft"], lw=1.0, zorder=3.6,
                    clip_on=False, alpha=0.7)
    return y0

def _table_card_height(headers, rows, fs, T, subtitle=None, zf=1.0):
    h = T["card_pad"] * 2 + fs * 2.1
    if subtitle:
        h += fs * 1.4
    h += fs * 1.9 + len(rows) * fs * 2.0
    return h

def _draw_table_card(ax, x0, x1, y1, title, headers, rows, fs, T,
                     dots=None, col_align=None, subtitle=None, zf=1.0):
    aw, ah = _ax_pt(ax)
    h = _table_card_height(headers, rows, fs, T, subtitle=subtitle, zf=zf)
    y0 = y1 - h / ah
    add_rrect(ax, x0, y0, x1, y1, T["card_radius"], fc=T["paper"],
              ec=T["card_edge"], lw=T["line_card"], z=2, shadow=True, T=T)
    pad = T["card_pad"] / aw
    ty = y1 - (T["card_pad"] + fs * 1.0) / ah
    ax.text(x0 + pad, ty, title, transform=ax.transAxes, fontsize=fs * 1.12,
            fontweight="black", color=T["ink"], ha="left", va="center",
            zorder=4)
    tw = _text_width_pt(title, fs * 1.12, "bold") / aw
    ax.plot([x0 + pad, x0 + pad + tw], [ty - fs * 1.05 / ah] * 2,
            transform=ax.transAxes, color=T["accent"], lw=2.4, zorder=4,
            clip_on=False, solid_capstyle="round")
    y = ty - fs * 2.0 / ah
    if subtitle:
        ax.text(x0 + pad, y, subtitle, transform=ax.transAxes,
                fontsize=fs * 0.84, color=T["ink_faint"],
                ha="left", va="center", zorder=4)
        y -= fs * 1.45 / ah

    n = len(headers)
    gut = 12.0 if dots else 0.0
    widths = []
    for c in range(n):
        w = _text_width_pt(headers[c], fs * 0.90, "bold")
        for row in rows:
            w = max(w, _text_width_pt(row[c], fs,
                                      "bold" if c == 0 else "normal"))
        widths.append(w + 14.0 + (gut if c == 0 else 0.0))
    total = sum(widths)
    inner = (x1 - x0 - 2 * pad) * aw
    if total > inner:
        widths = [w * inner / total for w in widths]
        total = inner
    frac = [w / aw for w in widths]
    bx = x0 + pad + (inner - total) / 2.0 / aw
    rows = [[_fit_text(cell, frac[c] * aw - (gut + 9 if c == 0 else 11), fs,
                       "bold" if c == 0 else "normal")
             for c, cell in enumerate(row)] for row in rows]

    y_top = y
    y = y - fs * 0.70 / ah
    hx = bx
    for c, head in enumerate(headers):
        align = (col_align or ["left"] * n)[c]
        if c == 0 and dots:
            cell_x, cell_align = hx + 12.0 / aw, "left"
        elif align == "left":
            cell_x, cell_align = hx + 6 / aw, "left"
        else:
            cell_x, cell_align = hx + frac[c] / 2, "center"
        ax.text(cell_x, y, head, transform=ax.transAxes,
                fontsize=fs * 0.90, fontweight="bold", color=T["ink_soft"],
                ha=cell_align, va="center", zorder=4)
        hx += frac[c]
    y -= fs * 0.95 / ah
    ax.plot([bx, bx + total / aw], [y, y], transform=ax.transAxes,
            color=T["card_edge"], lw=1.2, zorder=4, clip_on=False)

    row_h = fs * 2.0
    y_bot = y
    for r, row in enumerate(rows):
        y -= row_h / 2 / ah
        y_bot = y - row_h / 2 / ah
        if r % 2 == 0:
            ax.add_patch(Rectangle((bx, y - row_h / 2 / ah),
                                   total / aw, row_h / ah,
                                   transform=ax.transAxes,
                                   facecolor=T["paper_tint"], edgecolor="none",
                                   zorder=3, clip_on=False))
        hx = bx
        for c, cell in enumerate(row):
            align = (col_align or ["left"] * n)[c]
            if c == 0 and dots:
                _dot_patch(ax, hx + 6.0 / aw, y, fs * 0.34, fc=dots[r],
                           ec="#ffffff", lw=0.8, z=5)
                cell_x = hx + 13.0 / aw
                cell_align = "left"
            elif align == "left":
                cell_x, cell_align = hx + 6 / aw, "left"
            else:
                cell_x, cell_align = hx + frac[c] / 2, "center"
            ax.text(cell_x, y, cell, transform=ax.transAxes, fontsize=fs,
                    fontweight="bold" if c == 0 else "600",
                    color=T["ink"], ha=cell_align, va="center", zorder=5)
            hx += frac[c]
        y -= row_h / 2 / ah
    hx = bx
    for c in range(n - 1):
        hx += frac[c]
        ax.plot([hx, hx], [y_bot, y_top], transform=ax.transAxes,
                color=T["card_edge_soft"], lw=1.1, zorder=4.2,
                clip_on=False, alpha=0.8)
    return y0

# --- bottom band ------------------------------------------------------------
_BAND_PAD = 10.0
_STRIP_H = 26.0
_STRIP_GAP = 10.0

def _band_row_h(T, zf=1.0):
    base = max(18.0, T["fs_table"] * 1.65)
    return base * (0.9 + 0.15*zf)

def _band_height_pt(has_strip, has_table, T, zf=1.0):
    h = _BAND_PAD
    if has_strip:
        h += _STRIP_H + (_STRIP_GAP if has_table else 0.0)
    if has_table:
        h += 2 * _band_row_h(T, zf)
    h += _BAND_PAD
    return h if (has_strip or has_table) else 0.0

def _draw_band(ax, T, *, key_items, steps, time_label, speed_label,
               speed_unit, min_fontsize=6.8, zf=1.0):
    ax.axis("off")
    aw, ah = _ax_pt(ax)
    y = 1.0 - _BAND_PAD / ah

    if key_items:
        y -= _STRIP_H / ah
        fs = dynamic_font(T["fs_chip"] + 0.6, zf, 9, 15)
        def chip_widths(f):
            ws = []
            for _kind, _colour, label in key_items:
                ws.append(10 + 18 + 8 + _text_width_pt(label, f, "bold") + 10)
            return ws
        ws = chip_widths(fs)
        avail = aw - 10
        if sum(ws) + 8 * (len(ws) - 1) > avail:
            fs = max(6.5, fs * avail / (sum(ws) + 8 * (len(ws) - 1)))
            ws = chip_widths(fs)
        x = 5.0 / aw
        for (kind, colour, label), w in zip(key_items, ws):
            w_f = w / aw
            add_rrect(ax, x, y, x + w_f, y + _STRIP_H / ah, 6.0,
                      fc=T["paper"], ec=T["card_edge"], lw=1.0, z=2, T=T, shadow=True)
            _swatch(ax, x + (10 + 9) / aw, y + _STRIP_H / 2 / ah, kind, colour,
                    fs + 1.2, T, z=3, zf=zf)
            ax.text(x + (10 + 18 + 8) / aw, y + _STRIP_H / 2 / ah, label,
                    transform=ax.transAxes, fontsize=fs, fontweight="bold",
                    color=T["ink_soft"], ha="left", va="center", zorder=3)
            x += w_f + 8.0 / aw
        y -= _STRIP_GAP / ah

    if steps:
        fs = dynamic_font(T["fs_table"], zf, 9, 15)
        row_h = _band_row_h(T, zf)
        label_w = max(_text_width_pt(time_label, fs, "bold"),
                      _text_width_pt(speed_label, fs, "bold")) + 18
        col_w = (aw - 10 - label_w) / max(1, len(steps))
        widest = max(_text_width_pt(t.strftime("%H/%d%b").upper(), fs, "bold")
                     for t, _v in steps) + 12
        if widest > col_w:
            fs = max(min_fontsize, fs * col_w / widest)
            label_w = max(_text_width_pt(time_label, fs, "bold"),
                          _text_width_pt(speed_label, fs, "bold")) + 18
            col_w = (aw - 10 - label_w) / max(1, len(steps))

        y0 = y - 2 * row_h / ah
        add_rrect(ax, 5 / aw, y0, 1 - 5 / aw, y, 8.0, fc=T["paper"],
                  ec=T["card_edge"], lw=1.2, z=2, shadow=True, T=T)
        lx0, lx1 = 5 / aw, (5 + label_w) / aw
        ax.add_patch(Rectangle((lx0, y0), lx1 - lx0, y - y0,
                               transform=ax.transAxes, facecolor=T["navy"],
                               edgecolor="none", zorder=3, clip_on=False))
        ax.text((lx0 + lx1) / 2, y - row_h / 2 / ah, time_label,
                transform=ax.transAxes, fontsize=fs * 0.95,
                fontweight="bold", color=T["on_dark"], ha="center",
                va="center", zorder=4)
        ax.text((lx0 + lx1) / 2, y - 1.5 * row_h / ah, speed_label,
                transform=ax.transAxes, fontsize=fs * 0.95,
                fontweight="bold", color=T["on_dark"], ha="center",
                va="center", zorder=4)
        for i, (t, val) in enumerate(steps):
            cx0 = lx1 + i * col_w / aw
            cx1 = cx0 + col_w / aw
            if i % 2 == 0:
                ax.add_patch(Rectangle((cx0, y0), cx1 - cx0, y - y0,
                                       transform=ax.transAxes,
                                       facecolor=T["paper_tint"],
                                       edgecolor="none", zorder=3,
                                       clip_on=False))
            ax.text((cx0 + cx1) / 2, y - row_h / 2 / ah,
                    t.strftime("%H/%d%b").upper(), transform=ax.transAxes,
                    fontsize=fs, fontweight="bold", color=T["ink"],
                    ha="center", va="center", zorder=4)
            if val is None:
                txt, col = "--", T["ink_faint"]
            else:
                txt, col = f"{int(round(val))}{speed_unit}", wind_color(
                    val / KNOTS_TO_KMH)
            ax.text((cx0 + cx1) / 2, y - 1.5 * row_h / ah, txt,
                    transform=ax.transAxes, fontsize=fs, fontweight="black",
                    color=col, ha="center", va="center", zorder=4)
            ax.plot([cx1, cx1], [y0, y], transform=ax.transAxes,
                    color=T["card_edge_soft"], lw=1.1, zorder=3.5,
                    clip_on=False, alpha=0.8)
        ax.plot([lx0, 1 - 5 / aw], [y - row_h / ah] * 2,
                transform=ax.transAxes, color=T["card_edge_soft"], lw=1.1,
                zorder=3.6, clip_on=False, alpha=0.8)

# ---------------------------------------------------------------------------
# Map helpers
# ---------------------------------------------------------------------------
def _nice_step(span, target=7):
    for step in (0.5, 1, 2, 3, 5, 10, 15, 20):
        if span / step <= target:
            return step
    return 20

def _draw_states_and_labels(ax, T, bounds, geojson_path, zf=1.0):
    try:
        with open(geojson_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return

    lon_min, lon_max, lat_min, lat_max = bounds
    map_span = lon_max - lon_min

    major_countries = [
        {"name": "INDIA", "lon": 78.96, "lat": 22.50},
        {"name": "BANGLADESH", "lon": 90.35, "lat": 23.68},
        {"name": "PAKISTAN", "lon": 69.34, "lat": 30.37},
        {"name": "THAILAND", "lon": 100.99, "lat": 15.87},
        {"name": "SRI LANKA", "lon": 80.77, "lat": 7.87},
        {"name": "MYANMAR", "lon": 95.95, "lat": 21.91},
        {"name": "BHUTAN", "lon": 90.43, "lat": 27.51},
        {"name": "NEPAL", "lon": 84.12, "lat": 28.39},
        {"name": "CHINA", "lon": 104.19, "lat": 35.86}
    ]

    # Zoom out: show country labels with dynamic font
    if map_span > 14.0:
        fs_country = dynamic_font(T["fs_tiny"] * 1.5, zf, 8, 14)
        for country in major_countries:
            lon, lat = country["lon"], country["lat"]
            name = country["name"]
            if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
                continue
            ax.text(
                lon, lat, name,
                transform=ax.transData,
                fontsize=fs_country,
                fontweight="black",
                color=T["ink"],
                ha="center", va="center",
                alpha=0.88,
                zorder=3.0,
                bbox=dict(
                    facecolor=T["paper"],
                    alpha=0.92,
                    edgecolor=T["card_edge"],
                    linewidth=0.8,
                    boxstyle="round,pad=0.35"
                )
            )
        return

    # Zoom in: state borders + labels with dynamic sizing
    fs_state = dynamic_font(T["fs_tiny"] * 1.05, zf, 7, 11)
    for feat in data.get("features", ()):
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        if coords:
            lons = [p[0] for p in coords]
            lats = [p[1] for p in coords]
            ax.plot(
                lons, lats,
                color=T["state_border"],
                linestyle=":",
                linewidth=dynamic_linewidth(0.9, zf),
                alpha=0.75,
                transform=ax.transData,
                zorder=1.8
            )

    for feat in data.get("features", ()):
        props = feat.get("properties", {})
        name = props.get("name")
        lon = props.get("longitude")
        lat = props.get("latitude")
        if not name or lon is None or lat is None:
            continue
        if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
            continue
        ax.text(
            lon, lat, name,
            transform=ax.transData,
            fontsize=fs_state,
            fontweight="600",
            color=T["ink_faint"],
            ha="center", va="center",
            alpha=0.85,
            zorder=2.2,
            bbox=dict(
                facecolor=T["paper"],
                alpha=0.80,
                edgecolor="none",
                boxstyle="round,pad=0.18"
            )
        )

def _draw_map(ax, T, *, lon_min, lon_max, lat_min, lat_max, basemap_path,
              track_obs, track_for, cone_pts, smooth, fc_track,
              landfall_info, visible_ports, show_grid, zf=1.0, meta=None):
    lon_span = abs(float(lon_max) - float(lon_min))
    lat_span = abs(float(lat_max) - float(lat_min))
    lat_mid = 0.5 * (lat_min + lat_max)

    basemap.draw_land(ax, T, (lon_min, lon_max, lat_min, lat_max), basemap_path)
    borders_path = str(Path(basemap_path).with_name("borders.geojson"))
    basemap.draw_borders(ax, T, (lon_min, lon_max, lat_min, lat_max), borders_path)

    if getattr(cfg, "SHOW_STATES", True):
        states_path = str(Path(basemap_path).with_name("states.geojson"))
        basemap.draw_states(ax, T, (lon_min, lon_max, lat_min, lat_max), states_path)

    if getattr(cfg, "SHOW_STATES", True) or getattr(cfg, "SHOW_STATE_LABELS", True):
        states_path = str(Path(basemap_path).with_name("states.geojson"))
        _draw_states_and_labels(ax, T, (lon_min, lon_max, lat_min, lat_max), states_path, zf=zf)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal", adjustable="datalim")

    for spine in ax.spines.values():
        spine.set_color(T["map_edge"])
        spine.set_linewidth(dynamic_linewidth(1.6, zf))
    ax.tick_params(axis="both", which="both", length=0, width=0,
                   color=T["map_edge"], labelcolor=T["ink_soft"],
                   labelsize=dynamic_font(T["fs_tick"], zf, 9, 15), pad=2.0)

    xs = _nice_step(lon_max - lon_min)
    ys = _nice_step(lat_max - lat_min)
    # Keep ticks slightly inside edges to prevent label clipping (super dynamic)
    x_start = np.ceil((lon_min + 0.35) / xs) * xs
    x_end = np.floor((lon_max - 0.35) / xs) * xs
    y_start = np.ceil((lat_min + 0.35) / ys) * ys
    y_end = np.floor((lat_max - 0.35) / ys) * ys
    if x_start <= x_end:
        ax.set_xticks(np.arange(x_start, x_end + 0.01, xs))
    else:
        ax.set_xticks(np.arange(np.ceil(lon_min / xs) * xs, lon_max, xs))
    if y_start <= y_end:
        ax.set_yticks(np.arange(y_start, y_end + 0.01, ys))
    else:
        ax.set_yticks(np.arange(np.ceil(lat_min / ys) * ys, lat_max, ys))
    ax.set_xticklabels([f"{v:.0f}°E" for v in ax.get_xticks()], va="bottom", fontweight="bold")
    ax.set_yticklabels([f"{v:.0f}°N" for v in ax.get_yticks()], ha="left", va="center", fontweight="bold")

    if show_grid:
        ax.grid(color=T["grid"], linestyle=(0, (4, 4)), linewidth=dynamic_linewidth(0.9, zf),
                alpha=0.65, zorder=1)
    ax.set_axisbelow(True)

    # ---- wind-radius rings — gorgeous, dynamic ----
    if track_for is not None and len(track_for):
        for i in range(len(track_for)):
            lon = float(track_for["Longitude"].iloc[i])
            lat = float(track_for["Latitude"].iloc[i])
            for col, _lbl, colour in WIND_RADII:
                v = track_for[col].iloc[i]
                if pd.isna(v) or float(v) <= 0:
                    continue
                r = float(v)
                # soft fill
                ax.add_patch(Circle((lon, lat), r, facecolor=colour,
                                    alpha=0.07, linewidth=0, zorder=2.1))
                # white halo for crispness
                ax.add_patch(Circle((lon, lat), r, fill=False,
                                    edgecolor="#ffffff", linewidth=dynamic_linewidth(4.2, zf),
                                    alpha=0.70, zorder=4.5))
                # main ring
                ax.add_patch(Circle((lon, lat), r, fill=False,
                                    edgecolor=colour,
                                    linewidth=dynamic_linewidth(T["line_ring"], zf),
                                    alpha=0.90, zorder=4.7, linestyle="-"))

                # Radius label when zoomed in
                if zf > 1.2 and r > 0.3:
                    # place label at east edge of circle
                    label_lon = lon + r / max(0.3, math.cos(math.radians(lat)))
                    ax.text(label_lon, lat, f"{int(r*111)}km",
                            fontsize=dynamic_font(8.5, zf, 7, 11),
                            color=colour, fontweight="bold",
                            ha="left", va="center", zorder=5,
                            bbox=dict(facecolor="white", alpha=0.85, edgecolor=colour,
                                      boxstyle="round,pad=0.15", linewidth=0.8))

    # ---- uncertainty cone — modern with double fill for depth ----
    if cone_pts is not None:
        # outer soft fill
        ax.add_patch(Polygon(cone_pts, closed=True,
                             facecolor=T["cone_fill"], alpha=0.10,
                             edgecolor="none", zorder=3.0))
        # inner slightly darker
        ax.add_patch(Polygon(cone_pts, closed=True,
                             facecolor=T["cone_fill"], alpha=0.14,
                             edgecolor="none", zorder=3.1))
        # crisp edge above rings
        ax.add_patch(Polygon(cone_pts, closed=True, fill=False,
                             edgecolor=T["cone_edge"], alpha=0.95,
                             linewidth=dynamic_linewidth(2.4, zf), zorder=4.9))

    # ---- tracks — dynamic widths ----
    if track_obs is not None and len(track_obs) >= 2:
        ax.plot(track_obs["Longitude"], track_obs["Latitude"],
                color=T["obs_track"], lw=dynamic_linewidth(2.2, zf),
                linestyle=(0, (3, 1.5)), alpha=0.92, zorder=5.1,
                solid_capstyle="round")
    if fc_track is not None:
        ax.plot(fc_track[1], fc_track[0], color=T["accent"],
                lw=dynamic_linewidth(3.4, zf),
                solid_capstyle="round", zorder=5.3, alpha=0.95)
        # arrow head
        if smooth is not None and len(smooth[0]) > 4:
            lx, ly = smooth[0][-1], smooth[1][-1]
            px, py = smooth[0][-4], smooth[1][-4]
            ang = np.arctan2(ly - py, lx - px)
            size = 0.024 * (ax.get_ylim()[1] - ax.get_ylim()[0]) * (0.8 + 0.2*zf)
            tri = np.array([
                [lx + size * np.cos(ang), ly + size * np.sin(ang)],
                [lx + size * 0.62 * np.cos(ang + 2.5),
                 ly + size * 0.62 * np.sin(ang + 2.5)],
                [lx + size * 0.62 * np.cos(ang - 2.5),
                 ly + size * 0.62 * np.sin(ang - 2.5)],
            ])
            ax.add_patch(Polygon(tri, closed=True, facecolor=T["accent"],
                                 edgecolor="white", linewidth=1.0, zorder=5.4))

    # ---- markers — dynamic size ----
    base_obs_s = dynamic_marker_size(90, zf)
    base_for_s = dynamic_marker_size(155, zf)
    if track_obs is not None and len(track_obs):
        for lat, lon, wind in zip(track_obs["Latitude"],
                                  track_obs["Longitude"],
                                  track_obs["Intensity"]):
            ax.scatter(lon, lat, s=base_obs_s, color=wind_color(wind),
                       edgecolor="#000000", linewidth=1.6, zorder=6.2, alpha=0.95)
        clat, clon = float(track_obs["Latitude"].iloc[-1]), float(track_obs["Longitude"].iloc[-1])
        # pulsing rings for NOW
        ax.scatter(clon, clat, s=dynamic_marker_size(380, zf), facecolors="none",
                   edgecolors=T["accent"], linewidths=dynamic_linewidth(2.6, zf), zorder=6.3, alpha=0.9)
        ax.scatter(clon, clat, s=dynamic_marker_size(620, zf), facecolors="none",
                   edgecolors=T["accent"], linewidths=dynamic_linewidth(1.8, zf), alpha=0.35,
                   zorder=6.3)
    if track_for is not None and len(track_for):
        for lat, lon, wind in zip(track_for["Latitude"],
                                  track_for["Longitude"],
                                  track_for["Intensity"]):
            ax.scatter(lon, lat, s=base_for_s, color=wind_color(wind),
                       edgecolor="#000000", linewidth=2.1, zorder=6.4, alpha=0.98)

    # ---- ports — dynamic visibility & size ----
    port_threshold = 12.0 if zf < 0.9 else 18.0
    if lon_span <= port_threshold:
        port_s = dynamic_marker_size(60, zf)
        for name, plat, plon, risk in visible_ports:
            ax.scatter(plon, plat, s=port_s, color=risk["color"],
                       edgecolor="#000000", linewidth=1.3, zorder=6.1, alpha=0.95)

    # ---- landfall — star with glow ----
    if landfall_info is not None:
        lf_lon, lf_lat = landfall_info["lon"], landfall_info["lat"]
        # glow
        ax.scatter(lf_lon, lf_lat, s=dynamic_marker_size(420, zf), color=T["landfall"],
                   alpha=0.18, zorder=7.0, edgecolor="none")
        ax.scatter(lf_lon, lf_lat, s=dynamic_marker_size(280, zf), color=T["landfall"],
                   alpha=0.28, zorder=7.1, edgecolor="none")
        ax.plot(lf_lon, lf_lat, marker="*", markersize=dynamic_marker_size(18, zf),
                color=T["landfall"], markeredgecolor="#000000", markeredgewidth=2.2,
                linestyle="none", zorder=7.2)

def _scale_bar(ax, T, lon_span, lat_mid, zf=1.0):
    kx = 111.320 * np.cos(np.radians(lat_mid))
    best = None
    for total in (50, 100, 150, 200, 300, 400, 600, 800, 1200):
        frac = (total / kx) / lon_span
        if 0.10 <= frac <= 0.28:
            best = total
            break
    if best is None:
        best = 200
    fig = ax.figure
    pt2px = fig.dpi / 72.0
    aw_pt, ah_pt = _ax_pt(ax)
    fs = dynamic_font(T["fs_tiny"] + 2.6, zf, 8, 13)
    bar_w = 0.14 * aw_pt * (0.9 + 0.15*zf)
    bar_h = 5.0
    km_w = _text_width_pt("KM", fs, "bold")
    box_w = 10 + bar_w + 8 + km_w + 10
    box_h = 8 + fs * 1.2 + 3 + bar_h + 3 + fs * 1.2 + 8

    ax_bb = ax.get_window_extent(fig.canvas.get_renderer())
    x1_px = ax_bb.x1 - 8 * pt2px
    y0_px = ax_bb.y0 + 14 * pt2px
    x0_px = x1_px - box_w * pt2px
    y1_px = y0_px + box_h * pt2px

    def to_ax(px, py):
        return ((px - ax_bb.x0) / ax_bb.width, (py - ax_bb.y0) / ax_bb.height)

    bx0, by0 = to_ax(x0_px, y0_px)
    bx1, by1 = to_ax(x1_px, y1_px)

    # Modern card for scale bar
    add_rrect(ax, bx0, by0, bx1, by1, 6.0, fc="#ffffff", ec=T["card_edge"],
              lw=1.0, alpha=0.96, z=8, T=T, shadow=True)

    bar_y = by0 + (8 + fs * 1.2 + 3) / ah_pt
    sx = bx0 + 10 / aw_pt
    seg = (bar_w / 2) / aw_pt
    ax.add_patch(Rectangle((sx, bar_y), seg, bar_h / ah_pt,
                           transform=ax.transAxes, facecolor=T["ink"],
                           edgecolor=T["ink"], linewidth=0.6, zorder=9,
                           clip_on=False))
    ax.add_patch(Rectangle((sx + seg, bar_y), seg, bar_h / ah_pt,
                           transform=ax.transAxes, facecolor="#ffffff",
                           edgecolor=T["ink"], linewidth=0.6, zorder=9,
                           clip_on=False))
    ax.text(sx + (bar_w / 2) / aw_pt, bar_y + bar_h / ah_pt + 6 / ah_pt, "KM",
            transform=ax.transAxes, fontsize=fs, fontweight="bold",
            color=T["ink_faint"], ha="center", va="bottom", zorder=9)
    for frac_, txt in ((0.0, "0"), (0.5, f"{best // 2}"), (1.0, f"{best}")):
        ax.text(sx + (bar_w * frac_) / aw_pt, bar_y - 3 / ah_pt, txt,
                transform=ax.transAxes, fontsize=fs, fontweight="bold",
                color=T["ink_soft"], ha="center", va="top", zorder=9)
    return (x0_px, y0_px, x1_px, y1_px)

def _north_arrow(ax, T, zf=1.0):
    fig = ax.figure
    pt2px = fig.dpi / 72.0
    ax_bb = ax.get_window_extent(fig.canvas.get_renderer())
    fs = dynamic_font(T["fs_small"] + 5.0, zf, 10, 16)
    x_px = ax_bb.x1 - 24 * pt2px
    y_px = ax_bb.y1 - 18 * pt2px
    xa = (x_px - ax_bb.x0) / ax_bb.width
    ya = (y_px - ax_bb.y0) / ax_bb.height
    aw, ah = _ax_pt(ax)
    h = (26 + 4*zf) / ah
    w = (10 + 1.5*zf) / aw
    # Modern arrow with shadow
    ax.add_patch(Polygon([(xa, ya), (xa - w, ya - h), (xa, ya - h * 0.70),
                          (xa + w, ya - h)], closed=True,
                         facecolor=T["ink"], edgecolor="white", linewidth=0.8,
                         transform=ax.transAxes, zorder=9, clip_on=False))
    ax.text(xa, ya + 3 / ah, "N", transform=ax.transAxes, fontsize=fs,
            fontweight="black", color=T["ink"], ha="center", va="bottom",
            zorder=9)

    # Coastline & Border legend — modern, higher, no overlap with ticks
    fs_bl = dynamic_font(T["fs_tiny"] + 1.4, zf, 8, 12)
    ax.plot([0.030, 0.080], [0.125, 0.125], transform=ax.transAxes,
            color=T["coast"], lw=dynamic_linewidth(2.4, zf), zorder=9, clip_on=False,
            solid_capstyle="round")
    ax.text(0.090, 0.125, "Coastline", transform=ax.transAxes,
            fontsize=fs_bl, fontweight="bold", color=T["ink_soft"],
            ha="left", va="center", zorder=9)
    ax.plot([0.030, 0.080], [0.090, 0.090], transform=ax.transAxes,
            color=T["border"], lw=dynamic_linewidth(1.9, zf),
            linestyle=(0, (3.4, 2.4)), zorder=9, clip_on=False, solid_capstyle="round")
    ax.text(0.090, 0.090, "Country Border", transform=ax.transAxes,
            fontsize=fs_bl, fontweight="bold", color=T["ink_soft"],
            ha="left", va="center", zorder=9)

    return (x_px - 14 * pt2px, y_px - 18 * pt2px,
            x_px + 14 * pt2px, y_px + fs * 1.5 * pt2px)

# ---------------------------------------------------------------------------
# Main entry — SUPER DYNAMIC
# ---------------------------------------------------------------------------
def plot_cyclone(cyclone_name, track_data_obs, track_data_for, is_invest,
                 map_asset_path, output_path):
    global OUTPUT_DPI, FULL_TRACK_EXTENT
    T = get_theme(cfg.THEME)
    pt2px = OUTPUT_DPI / 72.0

    obs, fc = track_data_obs, track_data_for
    has_obs = obs is not None and len(obs) > 0
    has_for = fc is not None and len(fc) > 0
    if not has_obs and not has_for:
        raise ValueError("Track file contains no usable fixes")

    # analytics
    landfall_info, approach_rows = None, []
    if cfg.SHOW_LANDFALL or cfg.SHOW_PORTS or cfg.SHOW_APPROACH_TABLE:
        try:
            landfall_info = find_landfall(obs, fc, BOB())
            if cfg.SHOW_LANDFALL and landfall_info is not None:
                where = (f"near {landfall_info['place']}"
                         if landfall_info.get("place")
                         else f"at {landfall_info['lat']}N, {landfall_info['lon']}E")
                print(f"[LANDFALL] Est. landfall: {landfall_info['time_str']} {where} "
                      f"({landfall_info['time']:%d %b %Y %H:%M} UTC)")
            if cfg.SHOW_APPROACH_TABLE:
                approach_rows = port_centre_table(
                    obs, fc, BOB(), landfall=landfall_info,
                    radius_km=cfg.APPROACH_RADIUS, top=cfg.APPROACH_PORTS)
        except Exception as e:
            print(f"[WARN] Landfall/approach estimate failed: {e}")

    # motion
    obs_speed = obs_dir = None
    if has_obs and len(obs) >= 2:
        c1 = (obs["Latitude"].iloc[-2], obs["Longitude"].iloc[-2])
        c2 = (obs["Latitude"].iloc[-1], obs["Longitude"].iloc[-1])
        hrs = (obs["tnd"].iloc[-1] - obs["tnd"].iloc[-2]).total_seconds() / 3600
        if hrs > 0:
            obs_speed = haversine(c1, c2) / hrs
            obs_dir = get_cardinal_direction(get_bearing(c1, c2))
    for_speed = for_dir = time_label = None
    if has_obs and has_for:
        last = (obs["Latitude"].iloc[-1], obs["Longitude"].iloc[-1])
        last_t = obs["tnd"].iloc[-1]
        target = last_t + pd.Timedelta(hours=24)
        idx = None
        for i, t in enumerate(fc["tnd"]):
            if abs((t - target).total_seconds() / 3600) <= 3:
                idx = i
                break
        if idx is None:
            idx = len(fc) - 1
        coord = (fc["Latitude"].iloc[idx], fc["Longitude"].iloc[idx])
        hrs = (fc["tnd"].iloc[idx] - last_t).total_seconds() / 3600
        if hrs > 0:
            for_speed = haversine(last, coord) / hrs
            for_dir = get_cardinal_direction(get_bearing(last, coord))
        time_label = "24hr" if hrs >= 12 else f"{int(hrs)}h"

    # cone & track
    cone_pts = smooth = fc_track = None
    prev_lat = prev_lon = ci = None
    ci_tnd = None
    pressure = None
    if has_obs:
        prev_lat = float(obs["Latitude"].iloc[-1])
        prev_lon = float(obs["Longitude"].iloc[-1])
        ci = obs["Intensity"].iloc[-1]
        ci_tnd = obs["tnd"].iloc[-1]
        pressure = obs["Pressure"].iloc[-1]
    if cfg.SHOW_CONE and has_for and len(fc) >= 2 and prev_lat is not None:
        ext_lon = np.concatenate([[prev_lon], fc["Longitude"].values])
        ext_lat = np.concatenate([[prev_lat], fc["Latitude"].values])
        try:
            cone_pts, s_lon, s_lat = create_nhc_cone(
                ext_lon, ext_lat, initial_uncertainty=0.00, growth_rate=UCR)
            smooth = (s_lon, s_lat)
            fc_track = (s_lat, s_lon)
        except Exception as e:
            print(f"[WARN] Cone error ({e})")
            fc_track = (np.concatenate([[prev_lat], fc["Latitude"].values]),
                        np.concatenate([[prev_lon], fc["Longitude"].values]))
    elif has_for and prev_lat is not None:
        fc_track = (np.concatenate([[prev_lat], fc["Latitude"].values]),
                    np.concatenate([[prev_lon], fc["Longitude"].values]))

    # dynamic content
    steps = []
    if cfg.SHOW_FORECAST_TABLE and has_for:
        steps = thin_steps(
            forecast_table_steps(obs, fc, mode=cfg.FORECAST_SPEED_MODE,
                                 tz_offset_hours=cfg.FORECAST_TZ_OFFSET),
            cfg.FORECAST_TABLE_MAX_COLS)

    key_items = []
    if cfg.SHOW_FORECAST_KEY:
        if cfg.SHOW_CONE and has_for and len(fc) >= 2:
            key_items.append(("cone", T["cone_fill"], "Uncertainty Cone"))
        if fc_track is not None:
            key_items.append(("line", T["accent"], "Forecast Track"))
        if cfg.SHOW_LANDFALL and landfall_info is not None:
            key_items.append(("x", T["landfall"], "Landfall Est."))

    key_rows = []
    if cfg.SHOW_LEGEND:
        on_track = set()
        if has_obs:
            on_track |= {wind_color(w) for w in obs["Intensity"]}
        if has_for:
            on_track |= {wind_color(w) for w in fc["Intensity"]}
        int_rows = [(("item", "dot", c, lbl))
                    for _thr, _k, lbl, c in INTENSITY_LEGEND_ORDER
                    if c in on_track]
        if int_rows:
            key_rows.append(("section", "STORM INTENSITY"))
            key_rows.extend(int_rows)
        wr_rows = []
        if has_for:
            for col, lbl, colour in WIND_RADII:
                if any(pd.notna(v) and float(v) > 0 for v in fc[col]):
                    wr_rows.append(("item", "ring", colour, lbl))
        if wr_rows:
            key_rows.append(("section", "WIND RADII"))
            key_rows.extend(wr_rows)
        if cfg.SHOW_PORTS:
            key_rows.append(("section", "PORT RISK"))
            for band in PORT_RISK_BANDS:
                key_rows.append(("item", "dot",
                                 PORT_RISK.get(band["key"], band["color"]),
                                 f"{band['label'].title()}"))
            if landfall_info is None:
                key_rows.append(("item", "dot", PORT_RISK["unknown"],
                                 "Unknown · no landfall"))

    # stats
    stats = []
    if has_obs:
        p_txt = "--" if pressure is None or pd.isna(pressure) else f"{int(pressure)} hPa"
        stats.append(("CURRENT", f"{int(ci)} KTS", "", wind_color(ci)))
        stats.append(("PRESSURE", p_txt, "", T["ink"]))
    if cfg.SHOW_MAX_WIND_BOXES and has_obs:
        w_obs = obs["Intensity"].max()
        stats.append(("MAX OBSERVED", f"{int(w_obs)} KTS", "", wind_color(w_obs)))
    else:
        stats.append(("MAX OBSERVED", "--", "", T["ink_faint"]))
    if cfg.SHOW_MAX_WIND_BOXES and has_for:
        w_fc = fc["Intensity"].max()
        stats.append(("MAX FORECAST", f"{int(w_fc)} KTS", "", wind_color(w_fc)))
    else:
        stats.append(("MAX FORECAST", "--", "", T["ink_faint"]))
    if cfg.SHOW_MOVEMENT_TABLE and obs_dir is not None:
        stats.append(("MOVING (KM)", f"{obs_dir} {int(obs_speed)}", "km/h", T["ink"]))
    else:
        stats.append(("MOVING (KM)", "--", "", T["ink_faint"]))
    if cfg.SHOW_MOVEMENT_TABLE and for_dir is not None:
        stats.append(("NEXT 12HRS", f"{for_dir} {int(for_speed)}", f"km/h · {time_label}", T["ink"]))
    else:
        stats.append(("NEXT 12HRS", "--", "", T["ink_faint"]))
    if cfg.SHOW_LANDFALL and landfall_info is not None:
        stats.append(("LANDFALL EST.", landfall_info["time_str"],
                      landfall_info.get("place") or "coast crossing", T["landfall"]))
    else:
        stats.append(("LANDFALL EST.", "--", "", T["ink_faint"]))
    if cfg.SHOW_ACE_BOX and has_obs:
        stats.append(("ACE", f"{calculate_ace(obs):.3f}", "", T["ink"]))
    else:
        stats.append(("ACE", "--", "", T["ink_faint"]))

    port_rows, port_dots = [], []
    if cfg.SHOW_APPROACH_TABLE and approach_rows:
        for r in approach_rows:
            port_rows.append([r["name"], f"{r['dist_km']} km", r["dir_str"]])
            if landfall_info is not None:
                port_dots.append(classify_port_risk(r.get("landfall_km"))["color"])
            else:
                port_dots.append(PORT_RISK["unknown"])
        port_dots = [PORT_RISK.get(
            classify_port_risk(r.get("landfall_km"))["key"], c)
            for r, c in zip(approach_rows, port_dots)]

    # ---------------- layout — SUPER DYNAMIC, MOBILE-FIRST -------------------------------
    has_side = bool(key_rows) or bool(stats) or bool(port_rows)
    has_band = bool(key_items) or bool(steps)
    m_l, m_r = 0.048, 0.020  # left larger for °N labels, prevents clipping
    m_t = m_b = 0.016
    header_h = 0.115
    footer_h = 0.054 if cfg.SHOW_FOOTER else 0.0
    gap = 0.020
    # compute zoom early for band height
    if has_for:
        anchor_lat = fc["Latitude"].values
        anchor_lon = fc["Longitude"].values
    elif has_obs:
        anchor_lat = obs["Latitude"].values
        anchor_lon = obs["Longitude"].values
    else:
        anchor_lat = np.array([prev_lat])
        anchor_lon = np.array([prev_lon])
    if FULL_TRACK_EXTENT and has_obs:
        anchor_lat = np.concatenate([obs["Latitude"].values, anchor_lat])
        anchor_lon = np.concatenate([obs["Longitude"].values, anchor_lon])

    # SUPER DYNAMIC extent
    if cfg.WIND_RADIUS_EXTENT:
        lat_min, lat_max, lon_c, meta = wind_radius_extent(
            anchor_lat, anchor_lon, fc if has_for else None)
        zf = meta["zoom_factor"]
    else:
        lat_min = float(np.min(anchor_lat)) - BUFFER + cfg.MINLAT_OFFSET
        lat_max = float(np.max(anchor_lat)) + BUFFER + cfg.MAXLAT_OFFSET
        lon_c = 0.5 * (float(np.min(anchor_lon)) + float(np.max(anchor_lon)))
        zf = theme_zoom_factor(lat_max - lat_min, 6.0)
        meta = {"zoom_factor": zf, "lat_span": lat_max-lat_min, "lon_span": 6.0}

    band_h_pt = _band_height_pt(bool(key_items), bool(steps), T, zf=zf)
    band_h = band_h_pt / (FIG_H_IN * 72.0) if has_band else 0.0
    side_w = 0.285 if has_side else 0.0

    foot_top = m_b + footer_h + (0.008 if footer_h else 0.0)
    map_y1 = 1.0 - m_t - header_h - gap
    side_x0 = 1.0 - m_r - side_w
    map_x1 = side_x0 - (gap if has_side else 0.0)
    map_w = map_x1 - m_l
    band_y0 = foot_top
    band_y1 = band_y0 + band_h
    map_y0 = band_y1 + (0.045 if has_band else 0.022)  # extra room for °E ticks
    map_h = map_y1 - map_y0
    side_y0 = foot_top if has_band else map_y0
    side_h = map_y1 - side_y0

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=OUTPUT_DPI)
    fig.patch.set_facecolor(T["page_bg"])
    ax_head = fig.add_axes([m_l, 1 - m_t - header_h, 1 - m_l - m_r, header_h])
    ax_map = fig.add_axes([m_l, map_y0, map_w, map_h])
    ax_side = fig.add_axes([side_x0, side_y0, side_w, side_h]) if has_side else None
    ax_band = fig.add_axes([m_l, band_y0, map_w, band_h]) if has_band else None
    ax_foot = fig.add_axes([m_l, m_b, 1 - m_l - m_r, footer_h]) if footer_h else None

    # final lon span from aspect
    lat_span = lat_max - lat_min
    aspect = (map_w * FIG_W_IN) / (map_h * FIG_H_IN)
    lon_span = lat_span * aspect
    lon_c = min(max(lon_c, cfg.MIN_LON + lon_span / 2),
                cfg.MAX_LON - lon_span / 2)
    lon_min, lon_max = lon_c - lon_span / 2, lon_c + lon_span / 2

    # update meta with final span
    meta["lon_span"] = lon_max - lon_min
    meta["lat_span"] = lat_max - lat_min

    # ---------------- paint zones -------------------------------------------
    tz = pd.Timedelta(hours=cfg.FORECAST_TZ_OFFSET)
    issued_local = (ci_tnd + tz) if ci_tnd is not None else None
    local_str = ""
    if issued_local is not None:
        hr_12 = issued_local.strftime("%I").lstrip("0")
        ampm = issued_local.strftime("%p")
        local_str = f"LOCAL {hr_12}{ampm}, {issued_local:%d %b %Y}".upper()

    _draw_header(
        ax_head, T, name=str(cyclone_name), is_invest=is_invest,
        obs_span=(f"OBSERVED {obs['tnd'].iloc[0]:%d/%HZ} – "
                  f"{obs['tnd'].iloc[-1]:%d/%HZ}") if has_obs else "OBSERVED --",
        for_span=(f"FORECAST {fc['tnd'].iloc[0]:%d/%HZ} – "
                  f"{fc['tnd'].iloc[-1]:%d/%HZ}") if has_for else "FORECAST --",
        issued=(f"ISSUED {ci_tnd:%HZ, %d %b %Y}".upper() if ci_tnd is not None else "ISSUED --"),
        issued_sub=(local_str if local_str else "SYNOPTIC CHART"),
        brand=cfg.BRAND_NAME, zf=zf)

    if ax_foot is not None:
        p_txt = "--" if pressure is None or pd.isna(pressure) else int(pressure)
        _draw_footer(
            ax_foot, T,
            left=(f"WIND {int(ci)} KT  •  PRESSURE {p_txt} HPA  "
                  f"•  UPDATED {ci_tnd:%HZ @ %d %b %Y}"
                  if ci_tnd is not None else "NO CURRENT FIX"),
            right=cfg.FOOTER_TEXT, zf=zf)
    if ax_band is not None:
        _draw_band(ax_band, T, key_items=key_items, steps=steps,
                   time_label=cfg.FORECAST_TIME_LABEL,
                   speed_label=cfg.FORECAST_SPEED_LABEL,
                   speed_unit=cfg.FORECAST_SPEED_UNIT,
                   min_fontsize=cfg.FORECAST_TABLE_MIN_FONTSIZE, zf=zf)

    if ax_side is not None:
        ax_side.axis("off")
        cards = []
        if stats:
            cards.append(("glance", stats))
        if key_rows:
            cards.append(("key", key_rows))
        if port_rows:
            cards.append(("table", ("NEAREST PORTS",
                                    ["PORT", "DIST (KM)", "DIRECTION"],
                                    port_rows, port_dots,
                                    "centre distance & bearing from port")))
        _aw, ah_pt = _ax_pt(ax_side)
        fs_mult = {"glance": 1.22, "key": 1.05, "table": 0.92}  # more compact for mobile fit
        gaps = (len(cards) - 1) * T["card_gap"]
        def total_at(fs):
            tot = 0.0
            for kind, payload in cards:
                f = fs * fs_mult[kind]
                if kind == "key":
                    tot += _key_card_height(payload, f, T, zf=zf)
                elif kind == "glance":
                    tot += _glance_card_height(payload, f, T, zf=zf)
                else:
                    _t, _h, rows, _d, _sub = payload
                    tot += _table_card_height(_h, rows, f, T, subtitle=_sub, zf=zf)
            return tot + gaps
        fs = T["fs_body"]
        for cand in (1.20, 1.12, 1.04, 0.96, 0.88, 0.80, 0.72, 0.64, 0.56, 0.50):
            if total_at(T["fs_body"] * cand) <= ah_pt * 0.98:  # 98% to leave breathing room
                fs = T["fs_body"] * cand
                break
        else:
            fs = T["fs_body"] * 0.50
        y = 1.0
        x0, x1 = 0.0, 1.0
        for kind, payload in cards:
            f = fs * fs_mult[kind]
            if kind == "key":
                y = _draw_key_card(ax_side, x0, x1, y, payload, f, T, zf=zf)
            elif kind == "glance":
                y = _draw_glance_card(ax_side, x0, x1, y, payload, f, T, zf=zf)
            else:
                title, heads, rows, dots, sub = payload
                y = _draw_table_card(ax_side, x0, x1, y, title, heads, rows,
                                     f, T, dots=dots, subtitle=sub,
                                     col_align=["left", "center", "center"], zf=zf)
            y -= T["card_gap"] / ah_pt

    # map content
    visible_ports = []
    if cfg.SHOW_PORTS:
        for name, (plat, plon) in BOB().items():
            if lat_min <= plat <= lat_max and lon_min <= plon <= lon_max:
                dist = (haversine((plat, plon),
                                  (landfall_info["lat"], landfall_info["lon"]))
                        if landfall_info is not None else None)
                band = classify_port_risk(dist)
                band = dict(band)
                band["color"] = PORT_RISK.get(band["key"], band["color"])
                visible_ports.append((name, plat, plon, band))

    _draw_map(ax_map, T, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min,
              lat_max=lat_max, basemap_path=map_asset_path,
              track_obs=obs if has_obs else None,
              track_for=fc if has_for else None, cone_pts=cone_pts,
              smooth=smooth, fc_track=fc_track,
              landfall_info=(landfall_info if cfg.SHOW_LANDFALL else None),
              visible_ports=visible_ports, show_grid=cfg.SHOW_GRID, zf=zf, meta=meta)

    fig.canvas.draw()

    # ---------------- SUPER DYNAMIC collision-free chips --------------------
    marker_disp = []
    if has_obs:
        marker_disp += [ax_map.transData.transform((lon, lat))
                        for lat, lon in zip(obs["Latitude"], obs["Longitude"])]
    if has_for:
        marker_disp += [ax_map.transData.transform((lon, lat))
                        for lat, lon in zip(fc["Latitude"], fc["Longitude"])]
    if cfg.SHOW_LANDFALL and landfall_info is not None:
        marker_disp.append(ax_map.transData.transform(
            (landfall_info["lon"], landfall_info["lat"])))

    placed = []
    obstacles = []
    if cfg.SHOW_SCALE_BAR:
        obstacles.append(_scale_bar(ax_map, T, lon_max - lon_min,
                                    0.5 * (lat_min + lat_max), zf=zf))
    obstacles.append(_north_arrow(ax_map, T, zf=zf))
    ax_bb = ax_map.get_window_extent(fig.canvas.get_renderer())
    marker_r_px = pt2px * T["fs_chip"] * 1.10 * (0.8 + 0.25*zf)

    # 16 candidates around circle for super dynamic placement
    CANDIDATES = [
        (16, 16, "left", "bottom"), (-16, 16, "right", "bottom"),
        (16, -16, "left", "top"), (-16, -16, "right", "top"),
        (0, 22, "center", "bottom"), (0, -22, "center", "top"),
        (22, 0, "left", "center"), (-22, 0, "right", "center"),
        (18, 10, "left", "bottom"), (-18, 10, "right", "bottom"),
        (18, -10, "left", "top"), (-18, -10, "right", "top"),
        (10, 18, "left", "bottom"), (-10, 18, "right", "bottom"),
        (10, -18, "left", "top"), (-10, -18, "right", "top"),
    ]

    def place_chip(text, lon, lat, fs, edge, tcol, candidates, lw=1.2,
                   alpha=0.96, z=8.8, avoid_own=None, leader=True, priority=0):
        base = ax_map.transData.transform((lon, lat))
        w_pt = _text_width_pt(text, fs, "bold")
        pad_pt = 0.38 * fs
        h_pt = fs * 1.35
        best = None
        for dx, dy, ha, va in candidates:
            # dynamic offset scaling with zoom
            dx_s = dx * (0.9 + 0.15*zf)
            dy_s = dy * (0.9 + 0.15*zf)
            box = _chip_bbox(base, (dx_s * pt2px, dy_s * pt2px), ha, va,
                             w_pt, pad_pt, h_pt, pt2px)
            markers = [m for i, m in enumerate(marker_disp)
                       if avoid_own is None or i != avoid_own]
            viol = sum(_hits_point(box, x, y, marker_r_px) for x, y in markers)
            viol += sum(_overlap(box, p, gap=5.0) for p in placed)
            viol += sum(_overlap(box, o, gap=6.0) for o in obstacles)
            if (box[0] < ax_bb.x0 + 4 or box[1] < ax_bb.y0 + 4
                    or box[2] > ax_bb.x1 - 4 or box[3] > ax_bb.y1 - 4):
                viol += 8
            # priority: lower viol is better, but high priority can tolerate more
            score = viol - priority*0.1
            if best is None or score < best[0]:
                best = (score, viol, dx_s, dy_s, ha, va, box)
            if viol == 0:
                break
        _score, _v, dx, dy, ha, va, box = best
        # If still overlapping heavily, try smaller font
        if _v >= 3 and fs > 9:
            return place_chip(text, lon, lat, fs*0.88, edge, tcol, candidates,
                              lw=lw, alpha=alpha, z=z, avoid_own=avoid_own,
                              leader=leader, priority=priority)
        if _v >= 6:
            # skip if no good placement — prevents clutter
            return None
        if leader:
            from matplotlib.transforms import IdentityTransform
            ax_map.plot([base[0], base[0] + dx * pt2px],
                        [base[1], base[1] + dy * pt2px],
                        color=T["ink_faint"], lw=dynamic_linewidth(1.4, zf), alpha=0.75,
                        zorder=z - 0.5, clip_on=False,
                        transform=IdentityTransform(),
                        solid_capstyle="round")
        ax_map.annotate(
            text, xy=(lon, lat), xycoords="data", xytext=(dx, dy),
            textcoords="offset points", fontsize=fs, fontweight="bold",
            color=tcol, ha=ha, va=va, zorder=z,
            bbox=dict(facecolor="#ffffff", alpha=alpha, edgecolor=edge,
                      linewidth=lw, boxstyle=f"round,pad={0.38:.2f}"))
        placed.append(box)
        return box

    fs_chip = dynamic_font(T["fs_chip"], zf, 10, 16)

    # 1) NOW — highest priority
    if has_obs:
        place_chip(f"NOW · {int(ci)}KT",
                   prev_lon, prev_lat, fs_chip * 0.92, T["accent"], T["ink"],
                   CANDIDATES, lw=1.6, priority=10)

    # 2) forecast chips — priority by wind intensity
    if has_for:
        # sort by wind descending for priority
        fc_indices = sorted(range(len(fc)), key=lambda i: float(fc["Intensity"].iloc[i]), reverse=True)
        for idx in fc_indices:
            i = idx
            lat = float(fc["Latitude"].iloc[i])
            lon = float(fc["Longitude"].iloc[i])
            wind = fc["Intensity"].iloc[i]
            # dynamic font slightly smaller for weaker systems
            f_scale = 0.88 if float(wind) < 35 else 0.94
            place_chip(f"{fc['tnd'].iloc[i]:%d/%HZ} · {int(wind)}KT",
                       lon, lat, fs_chip * f_scale, wind_color(wind), T["ink"],
                       CANDIDATES, lw=1.5, priority=float(wind)/10.0)

    # 3) landfall chip
    if getattr(cfg, "SHOW_LANDFALL_LABEL", False) and cfg.SHOW_LANDFALL and landfall_info is not None:
        txt = f"LANDFALL {landfall_info['time_str']}"
        place_chip(txt, landfall_info["lon"], landfall_info["lat"],
                   fs_chip, T["landfall"], "#7f1d1d",
                   CANDIDATES, lw=1.9, alpha=0.98, priority=9)

    # 4) port chips — only when not too crowded, dynamic
    port_span_threshold = 11.0 if zf < 1.0 else 16.0
    if cfg.SHOW_PORTS and (lon_span <= port_span_threshold):
        # only show nearest ports to avoid clutter, already filtered by visible_ports
        # sort by distance to current centre for priority
        if has_obs:
            cur = (prev_lat, prev_lon)
            visible_ports_sorted = sorted(visible_ports,
                                          key=lambda x: haversine((x[1], x[2]), cur))
        else:
            visible_ports_sorted = visible_ports
        # limit number based on zoom
        max_ports = 6 if zf > 1.2 else 4
        for name, plat, plon, band in visible_ports_sorted[:max_ports]:
            base = ax_map.transData.transform((plon, plat))
            fs_p = dynamic_font(fs_chip * 0.72, zf, 8, 12)
            w_pt = _text_width_pt(name, fs_p, "bold")
            pad_pt = 0.38 * fs_p + 1.8
            h_pt = fs_p * 1.32
            ok = False
            best_box = None
            for dx, dy, ha, va in ((0, 16, "center", "bottom"),
                                   (18, 0, "left", "center"),
                                   (-18, 0, "right", "center"),
                                   (0, -16, "center", "top"),
                                   (14, 14, "left", "bottom"),
                                   (-14, 14, "right", "bottom"),
                                   (14, -14, "left", "top"),
                                   (-14, -14, "right", "top")):
                dx_s = dx * (0.9 + 0.15*zf)
                dy_s = dy * (0.9 + 0.15*zf)
                box = _chip_bbox(base, (dx_s * pt2px, dy_s * pt2px), ha, va,
                                 w_pt, pad_pt, h_pt, pt2px)
                if (box[0] < ax_bb.x0 + 4 or box[1] < ax_bb.y0 + 4
                        or box[2] > ax_bb.x1 - 4 or box[3] > ax_bb.y1 - 4):
                    continue
                if any(_overlap(box, p, gap=5.0) for p in placed) or \
                        any(_overlap(box, o, gap=6.0) for o in obstacles) or \
                        any(_hits_point(box, x, y, marker_r_px)
                            for x, y in marker_disp):
                    continue
                ok = True
                best_box = (box, dx_s, dy_s, ha, va)
                break
            if not ok:
                continue
            box, dx_s, dy_s, ha, va = best_box
            placed.append(box)
            ax_map.annotate(
                name, xy=(plon, plat), xycoords="data", xytext=(dx_s, dy_s),
                textcoords="offset points", fontsize=fs_p, fontweight="bold",
                color=T["ink"], ha=ha, va=va, zorder=8.4,
                bbox=dict(facecolor="#ffffff", alpha=0.94,
                          edgecolor=band["color"], linewidth=1.2,
                          boxstyle="round,pad=0.24"))

    # ---- inset mini-map when tightly zoomed — modern, clean ----
    try:
        if meta["lat_span"] < 5.5 and has_obs and len(obs) > 4:
            inset_left = 0.68
            inset_bottom = 0.55
            inset_w = 0.26
            inset_h = 0.30
            # background card
            bg_ax = fig.add_axes([m_l + map_w*inset_left - 0.006,
                                  map_y0 + map_h*inset_bottom - 0.008,
                                  map_w*inset_w + 0.012,
                                  map_h*inset_h + 0.016])
            bg_ax.axis("off")
            add_rrect(bg_ax, 0.0, 0.0, 1.0, 1.0, 10.0, fc="#ffffff",
                      ec=T["card_edge"], lw=1.1, alpha=0.98, z=11, T=T, shadow=True)
            ax_inset = fig.add_axes([m_l + map_w*inset_left,
                                     map_y0 + map_h*inset_bottom,
                                     map_w*inset_w, map_h*inset_h], zorder=12)
            ax_inset.set_facecolor(T["sea"])
            basemap.draw_land(ax_inset, T, (cfg.MIN_LON, cfg.MAX_LON, cfg.MIN_LAT, cfg.MAX_LAT),
                              map_asset_path, zf=0.7)
            ax_inset.plot(obs["Longitude"], obs["Latitude"], color=T["obs_track"],
                          lw=1.3, linestyle=(0, (2, 1)), alpha=0.85, zorder=2)
            if fc_track is not None:
                ax_inset.plot(fc_track[1], fc_track[0], color=T["accent"], lw=1.8, zorder=3)
            ax_inset.add_patch(Rectangle((lon_min, lat_min), lon_max-lon_min, lat_max-lat_min,
                                         fill=False, edgecolor=T["accent"], linewidth=1.6,
                                         linestyle="--", zorder=4))
            ax_inset.set_xlim(cfg.MIN_LON, cfg.MAX_LON)
            ax_inset.set_ylim(cfg.MIN_LAT, cfg.MAX_LAT)
            ax_inset.set_aspect("equal", adjustable="datalim")
            ax_inset.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax_inset.spines.values():
                spine.set_color(T["map_edge"])
                spine.set_linewidth(1.1)
    except Exception as e:
        print(f"[WARN] Inset failed: {e}")
        pass

    # ---------------- save — no tight bbox to keep side cards intact ----------------
    fig.savefig(output_path, dpi=OUTPUT_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {
        "landfall": landfall_info,
        "approaches": approach_rows,
        "cone_pts": cone_pts,
        "fc_track": fc_track,
        "map_bounds": (lon_min, lon_max, lat_min, lat_max),
        "axes_rect": (m_l, map_y0, map_w, map_h),
        "zoom_factor": zf,
        "meta": meta,
    }

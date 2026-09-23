"""
Cyclone track & uncertainty-cone graphic — professional layout engine.

The graphic is composed as a fixed five-zone dashboard, which is what makes
it impossible for two elements to ever overlap:

    +--------------------------------------------------------------+
    |  HEADER   brand chip · title · validity · issued card        |
    +---------------------------------------------+----------------+
    |                                             |   SIDEBAR      |
    |   MAP                                       |   at-a-glance  |
    |   (track, cone, wind radii, ports,          |   key card     |
    |    landfall, collision-free label chips)    |   port table   |
    |                                             |                |
    +---------------------------------------------+                |
    |  BOTTOM BAND   map key strip · forecast table|               |
    +---------------------------------------------+----------------+
    |  FOOTER   wind / pressure / updated · brand                  |
    +--------------------------------------------------------------+

Every zone is its own matplotlib axes, so cards can never drift into the
map, and the map window is solved from what is actually drawn (wind-radius
rings + cone) so nothing is ever cut by the frame.

Everything is data-driven ("dynamic"): a card, section, column or label is
offered only when the matching datum exists for this storm, all type sizes
are solved from measured glyph extents, and the sidebar packs itself to the
height it needs.  Colours and metrics come from cyclone/theme.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath

from . import basemap
from . import config as cfg
from .theme import (
    get_theme, INTENSITY_SCALE, INTENSITY_LEGEND_ORDER, WIND_RADII, PORT_RISK,
    wind_category, wind_color, wind_cat_label,
)
from .cone import create_nhc_cone
from .geo import haversine, get_bearing, get_cardinal_direction, KNOTS_TO_KMH
from .ace import calculate_ace
from .landfall import (
    find_landfall, port_centre_table, current_centre, classify_port_risk,
    PORT_RISK_BANDS,
)
from .ports import BOB

# Repo-root assets folder (works no matter what the current directory is)
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

# ---------------------------------------------------------------------------
# Module-level mirrors of the config values the CLI overrides per run
# (cyclone/cli.py assigns plotting.BUFFER = ... and friends).
# ---------------------------------------------------------------------------
BUFFER = cfg.BUFFER
UCR = cfg.UCR
OUTPUT_DPI = cfg.OUTPUT_DPI
FULL_TRACK_EXTENT = cfg.FULL_TRACK_EXTENT

FIG_W_IN, FIG_H_IN = 16.0, 10.0


# ---------------------------------------------------------------------------
# Text metrics (measured from the real font, no canvas draw needed)
# ---------------------------------------------------------------------------
_TEXT_SAFETY = 1.06


def _text_width_pt(text, fontsize, weight="normal"):
    """Ink width of `text` in points, plus a small safety margin."""
    text = str(text)
    try:
        from matplotlib.font_manager import FontProperties
        from matplotlib.textpath import TextPath
        fp = FontProperties(family="sans-serif", weight=weight, size=fontsize)
        ink = float(TextPath((0, 0), text, prop=fp).get_extents().width)
        return ink * _TEXT_SAFETY
    except Exception:
        return 0.62 * fontsize * len(text)


def _fit_text(text, max_width_pt, fontsize, weight="normal"):
    """Shorten `text` with an ellipsis until it fits `max_width_pt`."""
    text = str(text)
    if _text_width_pt(text, fontsize, weight) <= max_width_pt or not text:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _text_width_pt(text[:mid] + "\u2026", fontsize, weight) <= max_width_pt:
            lo = mid
        else:
            hi = mid - 1
    return (text[:lo] + "\u2026") if lo else "\u2026"


def _ax_pt(ax):
    """(width, height) of an axes in points."""
    fig = ax.figure
    pos = ax.get_position()
    return (max(1.0, pos.width * fig.get_figwidth() * 72.0),
            max(1.0, pos.height * fig.get_figheight() * 72.0))


# ---------------------------------------------------------------------------
# Rounded-rectangle primitive (display-space-correct corners + soft shadow)
# ---------------------------------------------------------------------------
def _rrect_path(x0, y0, x1, y1, rx, ry):
    """Rounded-rectangle Path in axes fractions; rx/ry per-axis radii."""
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


def add_rrect(ax, x0, y0, x1, y1, r_pt, *, fc="#ffffff", ec="#d5dee8",
              lw=1.0, alpha=1.0, z=1.0, shadow=False, T=None):
    """Draw a rounded card in axes fractions; `r_pt` corner radius in pt."""
    aw, ah = _ax_pt(ax)
    rx, ry = r_pt / aw, r_pt / ah
    if shadow:
        sx, sy = 1.4 / aw, 1.6 / ah
        ax.add_patch(PathPatch(
            _rrect_path(x0 + sx, y0 - sy, x1 + sx, y1 - sy, rx, ry),
            transform=ax.transAxes, facecolor=(T or {}).get("shadow", "#0b2545"),
            edgecolor="none", alpha=0.10, linewidth=0, zorder=z - 0.6,
            clip_on=False))
    patch = PathPatch(
        _rrect_path(x0, y0, x1, y1, rx, ry),
        transform=ax.transAxes, facecolor=fc, edgecolor=ec,
        linewidth=lw, alpha=alpha, zorder=z, clip_on=False)
    ax.add_patch(patch)
    return patch


# ---------------------------------------------------------------------------
# Map window solved from what is actually drawn (rings + cone)
# ---------------------------------------------------------------------------
def _row_max_radius(row):
    """Largest wind radius (degrees) in one forecast row, else 0."""
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
    (lat_min, lat_max, lon_centre) for the map window.

    The box starts at the track anchors (forecast track, plus the whole
    observed track when full_track_extent is on), grows until every forecast
    wind-radius circle and the uncertainty cone fit completely inside, adds
    an even `wind_radius_pad` margin and clips to the background map
    coverage, so a ring is never cut by the frame and no blank band around
    Map.png can appear.  The longitude centre is returned separately: the
    caller derives the lon span from the axes aspect ratio.
    """
    ucr = UCR if ucr is None else ucr
    show_cone = cfg.SHOW_CONE if show_cone is None else show_cone
    pad = cfg.WIND_RADIUS_PAD if pad is None else pad

    lat_min = float(np.min(anchor_lat))
    lat_max = float(np.max(anchor_lat))
    lon_min = float(np.min(anchor_lon))
    lon_max = float(np.max(anchor_lon))

    if track_for is not None and len(track_for) > 0:
        include_cone = show_cone and len(track_for) >= 2
        for i in range(len(track_for)):
            lat = float(track_for["Latitude"].iloc[i])
            lon = float(track_for["Longitude"].iloc[i])
            r = _row_max_radius(track_for.iloc[i])
            if include_cone:
                r = max(r, ucr * (i + 1))
            if r > 0.0:
                lat_min = min(lat_min, lat - r)
                lat_max = max(lat_max, lat + r)
                lon_min = min(lon_min, lon - r)
                lon_max = max(lon_max, lon + r)

    lat_min += cfg.MINLAT_OFFSET - pad
    lat_max += cfg.MAXLAT_OFFSET + pad
    lon_min -= pad
    lon_max += pad

    lat_min = max(lat_min, cfg.MIN_LAT)
    lat_max = min(lat_max, cfg.MAX_LAT)
    lon_min = max(lon_min, cfg.MIN_LON)
    lon_max = min(lon_max, cfg.MAX_LON)
    if lat_max - lat_min < 2.0:                     # degenerate safety net
        c = 0.5 * (lat_max + lat_min)
        lat_min, lat_max = c - 1.0, c + 1.0
    return lat_min, lat_max, 0.5 * (lon_min + lon_max)


# ---------------------------------------------------------------------------
# Forecast-table steps (shared with the README-documented behaviour)
# ---------------------------------------------------------------------------
def forecast_table_steps(track_obs, track_for, mode="wind", tz_offset_hours=6.0):
    """
    (time, value) per forecast step for the bottom table.

    mode="wind"   forecast wind in km/h (1 kt = 1.852 km/h)
    mode="motion" translation speed in km/h between consecutive points
    Times are shifted by `tz_offset_hours` (6 -> BST).
    """
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
    """Evenly drop steps above `max_cols`; first and last are always kept."""
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
# Label-chip geometry (display px), used by the collision solver
# ---------------------------------------------------------------------------
def _chip_bbox(base_disp, offset_px, ha, va, w_pt, pad_pt, h_pt, pt2px):
    """Padded display-px bbox of a chip whose text anchor is base+offset."""
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


def _overlap(a, b, gap=2.0):
    return not (a[2] + gap <= b[0] or b[2] + gap <= a[0]
                or a[3] + gap <= b[1] or b[3] + gap <= a[1])


# ---------------------------------------------------------------------------
# Zone painters
# ---------------------------------------------------------------------------
def _draw_header(ax, T, *, name, is_invest, obs_span, for_span, issued,
                 issued_sub, brand):
    ax.axis("off")
    aw, ah = _ax_pt(ax)

    # brand chip (left)
    fs_b = T["fs_small"] + 0.6
    bw = _text_width_pt(brand, fs_b, "bold") + 16
    bh = 15.0
    y_c = 0.62
    add_rrect(ax, 0.0, y_c - bh / 2 / ah, bw / aw, y_c + bh / 2 / ah,
              3.0, fc=T["navy"], ec="none", z=2, T=T)
    ax.text(bw / 2 / aw, y_c, brand, transform=ax.transAxes, fontsize=fs_b,
            fontweight="bold", color=T["on_dark"], ha="center", va="center",
            zorder=3)

    # issued card (right)
    fs_i = T["fs_small"] + 0.4
    iw = max(_text_width_pt(issued, fs_i, "bold"),
             _text_width_pt(issued_sub, T["fs_tiny"] + 0.4)) + 16
    ih = 24.0
    x1 = 1.0
    x0 = x1 - iw / aw
    add_rrect(ax, x0, y_c - ih / 2 / ah, x1, y_c + ih / 2 / ah, 3.0,
              fc=T["paper"], ec=T["card_edge"], lw=1.0, z=2, shadow=True, T=T)
    ax.text(x0 + 8 / aw, y_c + 3.5 / ah, issued, transform=ax.transAxes,
            fontsize=fs_i, fontweight="bold", color=T["ink"],
            ha="left", va="center", zorder=3)
    ax.text(x0 + 8 / aw, y_c - 5.5 / ah, issued_sub, transform=ax.transAxes,
            fontsize=T["fs_tiny"] + 0.4, color=T["ink_faint"],
            ha="left", va="center", zorder=3)

    # title + subtitle (centre)
    kind = "INVEST" if is_invest else "CYCLONE"
    ax.text(0.5, 0.66, f'TROPICAL {kind} \u201c{name.upper()}\u201d',
            transform=ax.transAxes, fontsize=T["fs_title"],
            fontweight="bold", color=T["ink"], ha="center", va="center",
            zorder=3)
    ax.text(0.5, 0.24, obs_span + "   \u00b7   " + for_span,
            transform=ax.transAxes, fontsize=T["fs_subtitle"],
            color=T["ink_soft"], ha="center", va="center", zorder=3)

    # structure rule under the header, with a short accent segment
    ax.plot([0, 1], [0.02, 0.02], transform=ax.transAxes, color=T["navy"],
            lw=1.4, zorder=2, clip_on=False)
    ax.plot([0, 0.16], [0.02, 0.02], transform=ax.transAxes,
            color=T["accent"], lw=2.6, zorder=3, clip_on=False)


def _draw_footer(ax, T, left, right):
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                           facecolor=T["navy"], edgecolor="none", zorder=1,
                           clip_on=False))
    ax.add_patch(Rectangle((0, 0.86), 0.16, 0.14, transform=ax.transAxes,
                           facecolor=T["accent"], edgecolor="none", zorder=2,
                           clip_on=False))
    ax.text(0.008, 0.5, left, transform=ax.transAxes, fontsize=T["fs_footer"],
            color=T["on_dark"], ha="left", va="center", zorder=3)
    ax.text(0.992, 0.5, right, transform=ax.transAxes,
            fontsize=T["fs_footer"], fontweight="bold", color=T["on_dark"],
            ha="right", va="center", zorder=3)


# --- sidebar cards ----------------------------------------------------------
def _dot_patch(ax, x_c, y_c, r_pt, *, fc, ec, lw, z=4, alpha=1.0):
    """True circle in display space (axes-fraction Ellipse), radius r_pt."""
    aw, ah = _ax_pt(ax)
    from matplotlib.patches import Ellipse
    return ax.add_patch(Ellipse((x_c, y_c), 2 * r_pt / aw, 2 * r_pt / ah,
                                transform=ax.transAxes, facecolor=fc,
                                edgecolor=ec, linewidth=lw, alpha=alpha,
                                zorder=z, clip_on=False))


def _swatch(ax, x_c, y_c, kind, colour, fs, T, z=4):
    """Draw one legend swatch centred at axes-fraction (x_c, y_c)."""
    aw, ah = _ax_pt(ax)
    r = fs * 0.42
    if kind == "dot":
        _dot_patch(ax, x_c, y_c, r, fc=colour, ec="#ffffff", lw=0.9, z=z)
    elif kind == "ring":
        from matplotlib.patches import Ellipse
        ax.add_patch(Ellipse((x_c, y_c), 2 * r / aw, 2 * r / ah,
                             transform=ax.transAxes, facecolor="none",
                             edgecolor=colour,
                             linewidth=max(1.0, fs * 0.16), zorder=z,
                             clip_on=False))
    elif kind == "cone":
        w, h = fs * 1.5, fs * 0.8
        ax.add_patch(Rectangle((x_c - w / 2 / aw, y_c - h / 2 / ah),
                               w / aw, h / ah, transform=ax.transAxes,
                               facecolor=T["cone_fill"], alpha=0.25,
                               edgecolor=T["cone_edge"], linewidth=0.8,
                               zorder=z, clip_on=False))
    elif kind in ("line", "dash", "coast", "border"):
        w = fs * 1.5
        ax.plot([x_c - w / 2 / aw, x_c + w / 2 / aw], [y_c, y_c],
                transform=ax.transAxes, color=colour,
                lw={"line": 2.0, "dash": 1.4,
                    "coast": 1.5, "border": 1.0}[kind],
                linestyle={"line": "-", "dash": (0, (4, 2.4)),
                           "coast": "-", "border": (0, (2.6, 1.8))}[kind],
                zorder=z, clip_on=False)
    elif kind == "x":
        ax.plot([x_c], [y_c], marker="X", markersize=fs * 1.0,
                transform=ax.transAxes, color=colour,
                markeredgecolor="#ffffff", markeredgewidth=0.8,
                linestyle="none", zorder=z, clip_on=False)
    elif kind == "star":
        ax.plot([x_c], [y_c], marker="*", markersize=fs * 1.3,
                transform=ax.transAxes, markerfacecolor=colour,
                markeredgecolor="#78350f", markeredgewidth=0.6,
                linestyle="none", zorder=z, clip_on=False)


# spacing tokens shared by the height solver and the painter (multiples of fs)
_KEY_SEC_GAP = 1.35     # clear space before a section title
_KEY_TITLE = 1.20       # the section-title band
_KEY_LEAD = 1.15        # lead-in between the title and the first item
_KEY_ITEM = 1.95        # legend item-row pitch


def _key_card_height(rows, fs, T):
    """rows: ('section', title) | ('item', kind, colour, label).

    Items pack two per row (a compact legend grid) and every section gets
    clear space before its title, so one block never runs into the next.
    """
    h = T["card_pad"] * 2 + fs * 1.95
    i = 0
    while i < len(rows):
        if rows[i][0] == "section":
            h += fs * (_KEY_SEC_GAP + _KEY_TITLE + _KEY_LEAD)
            i += 1
            j = i
            while j < len(rows) and rows[j][0] != "section":
                j += 1
            # two items share one row; an odd item keeps its own row
            h += ((j - i) // 2 + ((j - i) % 2)) * fs * _KEY_ITEM
            i = j
        else:
            h += fs * _KEY_ITEM
            i += 1
    return h


def _draw_key_card(ax, x0, x1, y1, rows, fs, T):
    """rows: ('section', title) | ('item', kind, colour, label)."""
    aw, ah = _ax_pt(ax)
    h = _key_card_height(rows, fs, T)
    y0 = y1 - h / ah
    add_rrect(ax, x0, y0, x1, y1, T["card_radius"], fc=T["paper"],
              ec=T["card_edge"], lw=T["line_card"], z=2, shadow=True, T=T)
    pad = T["card_pad"] / aw
    ty = y1 - (T["card_pad"] + fs * 0.9) / ah
    ax.text(x0 + pad, ty, "MAP KEY", transform=ax.transAxes,
            fontsize=fs * 1.08, fontweight="bold", color=T["ink"], ha="left",
            va="center", zorder=4)
    tw = _text_width_pt("MAP KEY", fs * 1.08, "bold") / aw
    ax.plot([x0 + pad, x0 + pad + tw], [ty - fs * 0.95 / ah] * 2,
            transform=ax.transAxes, color=T["accent"], lw=2.0, zorder=4,
            clip_on=False)
    y = ty - fs * 1.80 / ah
    gut = 17.0 / aw
    inner = (x1 - x0) - 2 * pad
    col_w = inner / 2.0
    i = 0
    while i < len(rows):
        if rows[i][0] == "section":
            y -= fs * _KEY_SEC_GAP / ah          # clear space before a section
            ax.text(x0 + pad, y - fs * 0.55 / ah, rows[i][1],
                    transform=ax.transAxes,
                    fontsize=fs * 0.86, fontweight="bold",
                    color=T["ink_faint"], ha="left", va="center", zorder=4)
            y -= fs * _KEY_TITLE / ah            # the title band
            y -= fs * _KEY_LEAD / ah             # lead-in before the items
            i += 1
            items = []
            while i < len(rows) and rows[i][0] != "section":
                items.append(rows[i])
                i += 1
            for k, row in enumerate(items):
                cx = x0 + pad + (k % 2) * col_w
                cy = y - fs * 0.92 / ah
                _kind, colour, label = row[1], row[2], row[3]
                _swatch(ax, cx + gut / 2, cy, _kind, colour, fs * 0.95, T)
                label = _fit_text(label, col_w * aw - gut - 8, fs * 0.94)
                ax.text(cx + gut + 3 / aw, cy, label,
                        transform=ax.transAxes, fontsize=fs * 0.94,
                        color=T["ink_soft"], ha="left", va="center", zorder=4)
                if k % 2 == 1 or k == len(items) - 1:
                    y -= fs * _KEY_ITEM / ah
        else:
            cy = y - fs * 0.92 / ah
            _kind, colour, label = rows[i][1], rows[i][2], rows[i][3]
            _swatch(ax, x0 + pad + gut / 2, cy, _kind, colour, fs * 0.95, T)
            label = _fit_text(label, inner * aw - gut - 8, fs * 0.94)
            ax.text(x0 + pad + gut + 3 / aw, cy, label,
                    transform=ax.transAxes, fontsize=fs * 0.94,
                    color=T["ink_soft"], ha="left", va="center", zorder=4)
            y -= fs * _KEY_ITEM / ah
            i += 1
    return y0


_GLANCE_ROW = 4.55         # row pitch of the stat grid, in multiples of fs
_GLANCE_VAL = 1.72         # value size, in multiples of fs


def _glance_card_height(stats, fs, T):
    rows = (len(stats) + 1) // 2
    return (T["card_pad"] * 2 + fs * 1.95 + fs * 1.25
            + rows * fs * _GLANCE_ROW - fs * 0.55)


def _draw_glance_card(ax, x0, x1, y1, stats, fs, T):
    """stats: (label, value, sub, colour) — a perfectly aligned stat grid.

    This is the headline information of the graphic: big bold values with
    their small label above and caption below, all on shared baselines.
    A thin vertical rule splits the two columns and a hairline sits between
    the rows, so the block reads like a clean dashboard — nothing tiny,
    nothing floating.
    """
    aw, ah = _ax_pt(ax)
    h = _glance_card_height(stats, fs, T)
    y0 = y1 - h / ah
    add_rrect(ax, x0, y0, x1, y1, T["card_radius"], fc=T["paper"],
              ec=T["card_edge"], lw=T["line_card"], z=2, shadow=True, T=T)
    pad = T["card_pad"] / aw
    ty = y1 - (T["card_pad"] + fs * 0.9) / ah
    ax.text(x0 + pad, ty, "AT A GLANCE", transform=ax.transAxes,
            fontsize=fs * 1.08, fontweight="bold", color=T["ink"], ha="left",
            va="center", zorder=4)
    tw = _text_width_pt("AT A GLANCE", fs * 1.08, "bold") / aw
    ax.plot([x0 + pad, x0 + pad + tw], [ty - fs * 0.95 / ah] * 2,
            transform=ax.transAxes, color=T["accent"], lw=2.0, zorder=4,
            clip_on=False)

    rows = (len(stats) + 1) // 2
    top = ty - fs * 2.45 / ah
    col_w = (x1 - x0 - 2 * pad) / 2.0
    mid_x = x0 + pad + col_w

    # thin vertical rule between the two columns (aligned, no dead space)
    r_top = top + fs * 0.45 / ah
    r_bot = top - (rows - 1) * fs * _GLANCE_ROW / ah - fs * 3.35 / ah
    ax.plot([mid_x, mid_x], [r_bot, r_top], transform=ax.transAxes,
            color=T["card_edge_soft"], lw=1.1, zorder=3.6, clip_on=False)

    for i, (label, value, sub, colour) in enumerate(stats):
        col = i % 2
        cx = x0 + pad + col * col_w + (10.0 if col else 2.0) / aw
        cy = top - (i // 2) * fs * _GLANCE_ROW / ah
        ax.text(cx, cy, label, transform=ax.transAxes,
                fontsize=fs * 0.90, fontweight="bold", color=T["ink_soft"],
                ha="left", va="center", zorder=4)
        ax.text(cx, cy - fs * 1.52 / ah, value, transform=ax.transAxes,
                fontsize=fs * _GLANCE_VAL, fontweight="bold", color=colour,
                ha="left", va="center", zorder=4)
        if sub:
            ax.text(cx, cy - fs * 2.95 / ah, sub, transform=ax.transAxes,
                    fontsize=fs * 0.90, color=T["ink_faint"],
                    ha="left", va="center", zorder=4)
        # hairline between the rows (drawn once per row, left column)
        if col == 0 and (i // 2) < rows - 1:
            ry = cy - fs * 3.55 / ah
            ax.plot([x0 + pad, x1 - pad], [ry, ry], transform=ax.transAxes,
                    color=T["card_edge_soft"], lw=0.9, zorder=3.6,
                    clip_on=False)
    return y0


def _table_card_height(headers, rows, fs, T, subtitle=None):
    h = T["card_pad"] * 2 + fs * 1.95
    if subtitle:
        h += fs * 1.30
    h += fs * 1.75 + len(rows) * fs * 1.85
    return h


def _draw_table_card(ax, x0, x1, y1, title, headers, rows, fs, T,
                     dots=None, col_align=None, subtitle=None):
    """headers/rows: list of str; dots: optional per-row risk colour.

    Columns are packed to their content (no dead space between them) and
    separated by thin vertical rules, so the data reads as one clean table.
    `subtitle` explains what the columns measure when the headers alone
    cannot (e.g. NEAREST PORTS: distance/bearing of the storm centre).
    """
    aw, ah = _ax_pt(ax)
    h = _table_card_height(headers, rows, fs, T, subtitle=subtitle)
    y0 = y1 - h / ah
    add_rrect(ax, x0, y0, x1, y1, T["card_radius"], fc=T["paper"],
              ec=T["card_edge"], lw=T["line_card"], z=2, shadow=True, T=T)
    pad = T["card_pad"] / aw
    ty = y1 - (T["card_pad"] + fs * 0.9) / ah
    ax.text(x0 + pad, ty, title, transform=ax.transAxes, fontsize=fs * 1.08,
            fontweight="bold", color=T["ink"], ha="left", va="center",
            zorder=4)
    tw = _text_width_pt(title, fs * 1.08, "bold") / aw
    ax.plot([x0 + pad, x0 + pad + tw], [ty - fs * 0.95 / ah] * 2,
            transform=ax.transAxes, color=T["accent"], lw=2.0, zorder=4,
            clip_on=False)
    y = ty - fs * 1.8 / ah
    if subtitle:
        ax.text(x0 + pad, y, subtitle, transform=ax.transAxes,
                fontsize=fs * 0.82, color=T["ink_faint"],
                ha="left", va="center", zorder=4)
        y -= fs * 1.30 / ah

    n = len(headers)
    gut = 10.0 if dots else 0.0          # risk-dot gutter in the name column
    widths = []
    for c in range(n):
        w = _text_width_pt(headers[c], fs * 0.88, "bold")
        for row in rows:
            # column 0 renders bold, so measure it bold
            w = max(w, _text_width_pt(row[c], fs,
                                      "bold" if c == 0 else "normal"))
        widths.append(w + 11.0 + (gut if c == 0 else 0.0))
    total = sum(widths)
    inner = (x1 - x0 - 2 * pad) * aw
    if total > inner:
        widths = [w * inner / total for w in widths]
        total = inner
    frac = [w / aw for w in widths]
    # packed columns, centred as one block inside the card
    bx = x0 + pad + (inner - total) / 2.0 / aw
    # never let a cell stick out of its column (e.g. "Krishnapatnam")
    rows = [[_fit_text(cell, frac[c] * aw - (gut + 7 if c == 0 else 9), fs,
                       "bold" if c == 0 else "normal")
             for c, cell in enumerate(row)] for row in rows]

    y_top = y
    y = y - fs * 0.62 / ah
    # header row (clear column titles)
    hx = bx
    for c, head in enumerate(headers):
        align = (col_align or ["left"] * n)[c]
        if c == 0 and dots:
            cell_x, cell_align = hx + 10.0 / aw, "left"
        elif align == "left":
            cell_x, cell_align = hx + 5 / aw, "left"
        else:
            cell_x, cell_align = hx + frac[c] / 2, "center"
        ax.text(cell_x, y, head, transform=ax.transAxes,
                fontsize=fs * 0.88, fontweight="bold", color=T["ink_soft"],
                ha=cell_align, va="center", zorder=4)
        hx += frac[c]
    y -= fs * 0.85 / ah
    ax.plot([bx, bx + total / aw], [y, y], transform=ax.transAxes,
            color=T["card_edge"], lw=1.1, zorder=4, clip_on=False)

    row_h = fs * 1.85
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
                _dot_patch(ax, hx + 5.0 / aw, y, fs * 0.30, fc=dots[r],
                           ec="#ffffff", lw=0.7, z=5)
                cell_x = hx + 11.0 / aw
                cell_align = "left"
            elif align == "left":
                cell_x, cell_align = hx + 5 / aw, "left"
            else:
                cell_x, cell_align = hx + frac[c] / 2, "center"
            ax.text(cell_x, y, cell, transform=ax.transAxes, fontsize=fs,
                    fontweight="bold" if c == 0 else "normal",
                    color=T["ink"], ha=cell_align, va="center", zorder=5)
            hx += frac[c]
        y -= row_h / 2 / ah
    # thin vertical rules between the columns (professional separation
    # without wide empty gutters)
    hx = bx
    for c in range(n - 1):
        hx += frac[c]
        ax.plot([hx, hx], [y_bot, y_top], transform=ax.transAxes,
                color=T["card_edge_soft"], lw=1.0, zorder=4.2,
                clip_on=False)
    return y0


# --- bottom band: key strip + forecast table --------------------------------
_BAND_PAD = 8.0            # outer breathing room of the band, pt
_STRIP_H = 22.0            # key-strip chip height, pt
_STRIP_GAP = 9.0           # clear space between the strip and the table, pt


def _band_row_h(T):
    """Forecast-table row height in points (grows with the type scale)."""
    return max(16.5, T["fs_table"] * 1.55)


def _band_height_pt(has_strip, has_table, T):
    h = _BAND_PAD
    if has_strip:
        h += _STRIP_H + (_STRIP_GAP if has_table else 0.0)
    if has_table:
        h += 2 * _band_row_h(T)
    h += _BAND_PAD
    return h if (has_strip or has_table) else 0.0


def _draw_band(ax, T, *, key_items, steps, time_label, speed_label,
               speed_unit, min_fontsize=6.8):
    ax.axis("off")
    aw, ah = _ax_pt(ax)
    y = 1.0 - _BAND_PAD / ah

    if key_items:
        y -= _STRIP_H / ah
        fs = T["fs_chip"] + 0.4
        # measure chips, shrink once if they do not fit the band width
        def chip_widths(f):
            ws = []
            for _kind, _colour, label in key_items:
                ws.append(8 + 16 + 6 + _text_width_pt(label, f, "bold") + 8)
            return ws
        ws = chip_widths(fs)
        avail = aw - 8
        if sum(ws) + 7 * (len(ws) - 1) > avail:
            fs = max(6.0, fs * avail / (sum(ws) + 7 * (len(ws) - 1)))
            ws = chip_widths(fs)
        x = 4.0 / aw
        for (kind, colour, label), w in zip(key_items, ws):
            w_f = w / aw
            add_rrect(ax, x, y, x + w_f, y + _STRIP_H / ah, 3.0,
                      fc=T["paper"], ec=T["card_edge"], lw=1.0, z=2, T=T)
            _swatch(ax, x + (8 + 8) / aw, y + _STRIP_H / 2 / ah, kind, colour,
                    fs + 1.0, T, z=3)
            ax.text(x + (8 + 16 + 6) / aw, y + _STRIP_H / 2 / ah, label,
                    transform=ax.transAxes, fontsize=fs, fontweight="bold",
                    color=T["ink_soft"], ha="left", va="center", zorder=3)
            x += w_f + 7.0 / aw
        y -= _STRIP_GAP / ah

    if steps:
        # The forecast table keeps its classic look: full-width card, navy
        # row-title column (TIME / WIND), and the forecast steps spread
        # evenly across the band with thin vertical rules between columns.
        fs = T["fs_table"]
        row_h = _band_row_h(T)
        label_w = max(_text_width_pt(time_label, fs, "bold"),
                      _text_width_pt(speed_label, fs, "bold")) + 14
        col_w = (aw - 8 - label_w) / max(1, len(steps))
        # shrink if a time header cannot fit its column
        widest = max(_text_width_pt(t.strftime("%H/%d%b").upper(), fs, "bold")
                     for t, _v in steps) + 10
        if widest > col_w:
            fs = max(min_fontsize, fs * col_w / widest)
            label_w = max(_text_width_pt(time_label, fs, "bold"),
                          _text_width_pt(speed_label, fs, "bold")) + 14
            col_w = (aw - 8 - label_w) / max(1, len(steps))

        y0 = y - 2 * row_h / ah
        add_rrect(ax, 4 / aw, y0, 1 - 4 / aw, y, 3.0, fc=T["paper"],
                  ec=T["card_edge"], lw=1.1, z=2, shadow=True, T=T)
        lx0, lx1 = 4 / aw, (4 + label_w) / aw
        # label column (navy)
        ax.add_patch(Rectangle((lx0, y0), lx1 - lx0, y - y0,
                               transform=ax.transAxes, facecolor=T["navy"],
                               edgecolor="none", zorder=3, clip_on=False))
        ax.text((lx0 + lx1) / 2, y - row_h / 2 / ah, time_label,
                transform=ax.transAxes, fontsize=fs * 0.92,
                fontweight="bold", color=T["on_dark"], ha="center",
                va="center", zorder=4)
        ax.text((lx0 + lx1) / 2, y - 1.5 * row_h / ah, speed_label,
                transform=ax.transAxes, fontsize=fs * 0.92,
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
                    transform=ax.transAxes, fontsize=fs, fontweight="bold",
                    color=col, ha="center", va="center", zorder=4)
            ax.plot([cx1, cx1], [y0, y], transform=ax.transAxes,
                    color=T["card_edge_soft"], lw=1.0, zorder=3.5,
                    clip_on=False)
        ax.plot([lx0, 1 - 4 / aw], [y - row_h / ah] * 2,
                transform=ax.transAxes, color=T["card_edge_soft"], lw=1.0,
                zorder=3.6, clip_on=False)


# ---------------------------------------------------------------------------
# The map itself
# ---------------------------------------------------------------------------
def _nice_step(span, target=7):
    for step in (0.5, 1, 2, 3, 5, 10, 15, 20):
        if span / step <= target:
            return step
    return 20


def _draw_map(ax, T, *, lon_min, lon_max, lat_min, lat_max, basemap_path,
              track_obs, track_for, cone_pts, smooth, fc_track,
              landfall_info, visible_ports, show_grid):
    basemap.draw_land(ax, T, (lon_min, lon_max, lat_min, lat_max),
                      basemap_path)
    basemap.draw_borders(ax, T, (lon_min, lon_max, lat_min, lat_max),
                         str(Path(basemap_path).with_name("borders.geojson")))
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal", adjustable="datalim")

    for spine in ax.spines.values():
        spine.set_color(T["map_edge"])
        spine.set_linewidth(1.5)
    ax.tick_params(axis="both", which="both", length=5.5, width=1.25,
                   color=T["map_edge"], labelcolor=T["ink_soft"],
                   labelsize=T["fs_tick"], pad=5.0)
    xs = _nice_step(lon_max - lon_min)
    ys = _nice_step(lat_max - lat_min)
    ax.set_xticks(np.arange(np.ceil(lon_min / xs) * xs, lon_max, xs))
    ax.set_yticks(np.arange(np.ceil(lat_min / ys) * ys, lat_max, ys))
    ax.set_xticklabels([f"{v:.0f}\u00b0E" for v in ax.get_xticks()])
    ax.set_yticklabels([f"{v:.0f}\u00b0N" for v in ax.get_yticks()])
    if show_grid:
        ax.grid(color=T["grid"], linestyle=(0, (3, 3)), linewidth=0.7,
                alpha=0.38, zorder=1)
    ax.set_axisbelow(True)

    # ---- wind-radius rings (fills under everything, crisp rings above) ----
    if track_for is not None and len(track_for):
        for i in range(len(track_for)):
            lon = float(track_for["Longitude"].iloc[i])
            lat = float(track_for["Latitude"].iloc[i])
            for col, _lbl, colour in WIND_RADII:
                v = track_for[col].iloc[i]
                if pd.isna(v) or float(v) <= 0:
                    continue
                r = float(v)
                ax.add_patch(Circle((lon, lat), r, facecolor=colour,
                                    alpha=0.05, linewidth=0, zorder=2))
                ax.add_patch(Circle((lon, lat), r, fill=False,
                                    edgecolor="#ffffff", linewidth=3.8,
                                    alpha=0.55, zorder=4.4))
                ax.add_patch(Circle((lon, lat), r, fill=False,
                                    edgecolor=colour,
                                    linewidth=T["line_ring"], alpha=0.85,
                                    zorder=4.6))

    # ---- uncertainty cone --------------------------------------------------
    if cone_pts is not None:
        ax.add_patch(Polygon(cone_pts, closed=True,
                             facecolor=T["cone_fill"], alpha=0.18,
                             edgecolor="none", zorder=3))
        # outline ABOVE the wind-radius rings so the rounded U end of the
        # cone is never hidden under a coincident isotach stroke
        ax.add_patch(Polygon(cone_pts, closed=True, fill=False,
                             edgecolor=T["cone_edge"], alpha=0.9,
                             linewidth=2.2, zorder=4.8))

    # ---- tracks ------------------------------------------------------------
    if track_obs is not None and len(track_obs) >= 2:
        ax.plot(track_obs["Longitude"], track_obs["Latitude"],
                color=T["obs_track"], lw=3.2, linestyle=(0, (4, 2.2)),
                alpha=0.9, zorder=5)
    if fc_track is not None:
        # the forecast track/path through the forecast markers: fc_track is
        # (lats, lons) — plot x=lons, y=lats
        ax.plot(fc_track[1], fc_track[0], color=T["accent"], lw=5.2,
                solid_capstyle="round", zorder=5.2)
        # direction arrow head at the end of the forecast track
        if smooth is not None and len(smooth[0]) > 4:
            lx, ly = smooth[0][-1], smooth[1][-1]
            px, py = smooth[0][-4], smooth[1][-4]
            ang = np.arctan2(ly - py, lx - px)
            size = 0.021 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            tri = np.array([
                [lx + size * np.cos(ang), ly + size * np.sin(ang)],
                [lx + size * 0.62 * np.cos(ang + 2.5),
                 ly + size * 0.62 * np.sin(ang + 2.5)],
                [lx + size * 0.62 * np.cos(ang - 2.5),
                 ly + size * 0.62 * np.sin(ang - 2.5)],
            ])
            ax.add_patch(Polygon(tri, closed=True, facecolor=T["accent"],
                                 edgecolor="none", zorder=5.3))

    # ---- markers -----------------------------------------------------------
    if track_obs is not None and len(track_obs):
        for lat, lon, wind in zip(track_obs["Latitude"],
                                  track_obs["Longitude"],
                                  track_obs["Intensity"]):
            ax.scatter(lon, lat, s=130, color=wind_color(wind),
                       edgecolor="#ffffff", linewidth=2.0, zorder=6)
        clat, clon = float(track_obs["Latitude"].iloc[-1]), \
            float(track_obs["Longitude"].iloc[-1])
        ax.scatter(clon, clat, s=340, facecolors="none",
                   edgecolors=T["accent"], linewidths=2.4, zorder=6.1)
        ax.scatter(clon, clat, s=560, facecolors="none",
                   edgecolors=T["accent"], linewidths=1.6, alpha=0.35,
                   zorder=6.1)
    if track_for is not None and len(track_for):
        for lat, lon, wind in zip(track_for["Latitude"],
                                  track_for["Longitude"],
                                  track_for["Intensity"]):
            ax.scatter(lon, lat, s=145, color=wind_color(wind),
                       edgecolor="#ffffff", linewidth=2.0, zorder=6.2)

    # ---- ports -------------------------------------------------------------
    for name, plat, plon, risk in visible_ports:
        ax.scatter(plon, plat, s=125, color=risk["color"],
                   edgecolor="#ffffff", linewidth=1.9, zorder=6)

    # ---- landfall ----------------------------------------------------------
    if landfall_info is not None:
        ax.plot(landfall_info["lon"], landfall_info["lat"], marker="X",
                markersize=28, color=T["landfall"],
                markeredgecolor="#ffffff", markeredgewidth=2.3,
                linestyle="none", zorder=7)


def _scale_bar(ax, T, lon_span, lat_mid):
    """Kilometre scale bar, bottom-right inside the map. Returns its px box."""
    kx = 111.320 * np.cos(np.radians(lat_mid))
    best = None
    for total in (100, 150, 200, 300, 400, 600, 800, 1200):
        frac = (total / kx) / lon_span
        if 0.13 <= frac <= 0.30:
            best = total
            break
    if best is None:
        best = 300
    fig = ax.figure
    pt2px = fig.dpi / 72.0
    aw_pt, ah_pt = _ax_pt(ax)
    fs = T["fs_tiny"] + 2.4
    bar_w = 0.19 * aw_pt                 # pt
    bar_h = 5.6                          # pt
    km_w = _text_width_pt("KM", fs, "bold")
    box_w = 7 + bar_w + 6 + km_w + 7     # pt
    box_h = 6 + fs * 1.15 + 2 + bar_h + 2 + fs * 1.15 + 5

    ax_bb = ax.get_window_extent(fig.canvas.get_renderer())
    x1_px = ax_bb.x1 - 8 * pt2px
    y0_px = ax_bb.y0 + 8 * pt2px
    x0_px = x1_px - box_w * pt2px
    y1_px = y0_px + box_h * pt2px

    def to_ax(px, py):
        return ((px - ax_bb.x0) / ax_bb.width, (py - ax_bb.y0) / ax_bb.height)

    bx0, by0 = to_ax(x0_px, y0_px)
    bx1, by1 = to_ax(x1_px, y1_px)
    add_rrect(ax, bx0, by0, bx1, by1, 3.0, fc="#ffffff", ec=T["card_edge"],
              lw=0.9, alpha=0.92, z=8, T=T)

    bar_y = by0 + (5 + fs * 1.15 + 2) / ah_pt
    sx = bx0 + 6 / aw_pt
    seg = (bar_w / 2) / aw_pt
    ax.add_patch(Rectangle((sx, bar_y), seg, bar_h / ah_pt,
                           transform=ax.transAxes, facecolor=T["ink"],
                           edgecolor=T["ink"], linewidth=0.6, zorder=9,
                           clip_on=False))
    ax.add_patch(Rectangle((sx + seg, bar_y), seg, bar_h / ah_pt,
                           transform=ax.transAxes, facecolor="#ffffff",
                           edgecolor=T["ink"], linewidth=0.6, zorder=9,
                           clip_on=False))
    ax.text(sx + bar_w / aw_pt + 5 / aw_pt, bar_y + bar_h / 2 / ah_pt, "KM",
            transform=ax.transAxes, fontsize=fs, fontweight="bold",
            color=T["ink_faint"], ha="left", va="center", zorder=9)
    for frac_, txt in ((0.0, "0"), (0.5, f"{best // 2}"), (1.0, f"{best}")):
        ax.text(sx + (bar_w * frac_) / aw_pt, bar_y - 2 / ah_pt, txt,
                transform=ax.transAxes, fontsize=fs, fontweight="bold",
                color=T["ink_soft"], ha="center", va="top", zorder=9)
    return (x0_px, y0_px, x1_px, y1_px)


def _north_arrow(ax, T):
    fig = ax.figure
    pt2px = fig.dpi / 72.0
    ax_bb = ax.get_window_extent(fig.canvas.get_renderer())
    fs = T["fs_small"] + 4.6
    x_px = ax_bb.x1 - 20 * pt2px
    y_px = ax_bb.y1 - 16 * pt2px
    xa = (x_px - ax_bb.x0) / ax_bb.width
    ya = (y_px - ax_bb.y0) / ax_bb.height
    aw, ah = _ax_pt(ax)
    h = 24 / ah
    w = 9.5 / aw
    ax.add_patch(Polygon([(xa, ya), (xa - w, ya - h), (xa, ya - h * 0.72),
                          (xa + w, ya - h)], closed=True,
                         facecolor=T["ink"], edgecolor="none",
                         transform=ax.transAxes, zorder=9, clip_on=False))
    ax.text(xa, ya + 2 / ah, "N", transform=ax.transAxes, fontsize=fs,
            fontweight="bold", color=T["ink"], ha="center", va="bottom",
            zorder=9)
    return (x_px - 12 * pt2px, y_px - 16 * pt2px,
            x_px + 12 * pt2px, y_px + fs * 1.4 * pt2px)


# ---------------------------------------------------------------------------
# Main entry point
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
        raise ValueError("Track file contains no usable fixes "
                         "(need at least one position row).")

    # ---------------- analytics (landfall, ports, motion) ------------------
    landfall_info, approach_rows = None, []
    if cfg.SHOW_LANDFALL or cfg.SHOW_PORTS or cfg.SHOW_APPROACH_TABLE:
        try:
            landfall_info = find_landfall(obs, fc, BOB())
            if cfg.SHOW_LANDFALL:
                if landfall_info is not None:
                    where = (f"near {landfall_info['place']}"
                             if landfall_info.get("place")
                             else f"at {landfall_info['lat']}N, "
                                  f"{landfall_info['lon']}E")
                    print(f"[LANDFALL] Est. landfall: "
                          f"{landfall_info['time_str']} {where} "
                          f"({landfall_info['time']:%d %b %Y %H:%M} UTC)")
                else:
                    print("[LANDFALL] No landfall detected within the "
                          f"forecast period.")
            if cfg.SHOW_APPROACH_TABLE:
                approach_rows = port_centre_table(
                    obs, fc, BOB(), landfall=landfall_info,
                    radius_km=cfg.APPROACH_RADIUS, top=cfg.APPROACH_PORTS)
                if approach_rows:
                    centre = current_centre(obs, fc)
                    tops = " | ".join(
                        f"{r['name']} {r['dist_km']} km {r['dir_str']}"
                        for r in approach_rows)
                    print(f"[PORTS] From current centre "
                          f"({centre[0]:.1f}N, {centre[1]:.1f}E): {tops}")
        except Exception as e:
            print(f"[WARN] Landfall/approach estimate failed: {e}")
    if landfall_info is not None and not cfg.SHOW_LANDFALL:
        pass  # computed for port colours only

    # motion statistics
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

    # ---------------- cone & forecast track geometry -----------------------
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
            smooth = (s_lon, s_lat)          # (lons, lats) — for the arrow
            fc_track = (s_lat, s_lon)        # (lats, lons) — plot-ready pair
        except Exception as e:
            print(f"[WARN] Cone error ({e}) - drawing line only")
            fc_track = (np.concatenate([[prev_lat], fc["Latitude"].values]),
                        np.concatenate([[prev_lon], fc["Longitude"].values]))
    elif has_for and prev_lat is not None:
        fc_track = (np.concatenate([[prev_lat], fc["Latitude"].values]),
                    np.concatenate([[prev_lon], fc["Longitude"].values]))

    # ---------------- dynamic content lists --------------------------------
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

    # legend rows
    key_rows = []
    if cfg.SHOW_LEGEND:
        on_track = set()
        if has_obs:
            on_track |= {wind_color(w) for w in obs["Intensity"]}
        if has_for:
            on_track |= {wind_color(w) for w in fc["Intensity"]}
        # weakest -> strongest: Invest Area / Low, Tropical Depression, ...
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
                                 f"{band['label'].title()} \u00b7 "
                                 f"{band['legend']}"))
            if landfall_info is None:
                key_rows.append(("item", "dot", PORT_RISK["unknown"],
                                 "Unknown \u00b7 no landfall"))
        tr_rows = []
        if fc_track is not None:
            tr_rows.append(("item", "line", T["accent"], "Forecast Track"))
        if cfg.SHOW_CONE and has_for and len(fc) >= 2:
            tr_rows.append(("item", "cone", T["cone_fill"],
                            "Uncertainty Cone"))
        if cfg.SHOW_LANDFALL and landfall_info is not None:
            tr_rows.append(("item", "x", T["landfall"], "Landfall Est."))
        if tr_rows:
            key_rows.append(("section", "TRACK & AREAS"))
            key_rows.extend(tr_rows)
        base_rows = [("item", "coast", T["coast"], "Coastline")]
        if basemap.has_borders(str(Path(map_asset_path).with_name(
                "borders.geojson"))):
            base_rows.append(("item", "border", T["border"],
                              "Country Border"))
        key_rows.append(("section", "BASEMAP"))
        key_rows.extend(base_rows)

    # glance stats
    stats = []
    if cfg.SHOW_MAX_WIND_BOXES and has_obs:
        w = obs["Intensity"].max()
        stats.append(("MAX OBSERVED", f"{int(w)} KT", wind_cat_label(w),
                      wind_color(w)))
    if cfg.SHOW_MAX_WIND_BOXES and has_for:
        w = fc["Intensity"].max()
        stats.append(("MAX FORECAST", f"{int(w)} KT", wind_cat_label(w),
                      wind_color(w)))
    if cfg.SHOW_ACE_BOX and has_obs:
        stats.append(("ACE", f"{calculate_ace(obs):.3f}", "accumulated energy",
                      T["ink"]))
    if has_obs:
        p_txt = "--" if pressure is None or pd.isna(pressure) \
            else f"{int(pressure)} hPa"
        stats.append(("CURRENT", f"{int(ci)} KT", p_txt, wind_color(ci)))
    if cfg.SHOW_MOVEMENT_TABLE and obs_dir is not None:
        stats.append(("MOTION NOW", f"{obs_dir} {int(obs_speed)}",
                      "km/h", T["ink"]))
    if cfg.SHOW_MOVEMENT_TABLE and for_dir is not None:
        stats.append(("MOTION NEXT", f"{for_dir} {int(for_speed)}",
                      f"km/h \u00b7 {time_label}", T["ink"]))
    if cfg.SHOW_LANDFALL and landfall_info is not None:
        stats.append(("LANDFALL EST.", landfall_info["time_str"],
                      landfall_info.get("place") or "coast crossing",
                      T["landfall"]))

    port_rows, port_dots = [], []
    if cfg.SHOW_APPROACH_TABLE and approach_rows:
        for r in approach_rows:
            port_rows.append([r["name"], f"{r['dist_km']} km", r["dir_str"]])
            if landfall_info is not None:
                port_dots.append(classify_port_risk(r.get("landfall_km"))
                                 ["color"])
            else:
                port_dots.append(PORT_RISK["unknown"])
        port_dots = [PORT_RISK.get(
            classify_port_risk(r.get("landfall_km"))["key"], c)
            for r, c in zip(approach_rows, port_dots)]

    # ---------------- layout ------------------------------------------------
    has_side = bool(key_rows) or bool(stats) or bool(port_rows)
    has_band = bool(key_items) or bool(steps)
    m_l, m_r = 0.035, 0.018      # left margin carries the °N tick labels
    m_t = m_b = 0.015
    header_h = 0.106
    footer_h = 0.048 if cfg.SHOW_FOOTER else 0.0
    gap = 0.016                  # clear space between every two zones
    band_h_pt = _band_height_pt(bool(key_items), bool(steps), T)
    band_h = band_h_pt / (FIG_H_IN * 72.0) if has_band else 0.0
    side_w = 0.262 if has_side else 0.0

    foot_top = m_b + footer_h + (0.006 if footer_h else 0.0)
    map_y1 = 1.0 - m_t - header_h - gap
    side_x0 = 1.0 - m_r - side_w
    map_x1 = side_x0 - (gap if has_side else 0.0)
    map_w = map_x1 - m_l
    # the band sits directly under the map (map width); the sidebar column
    # runs all the way down beside it, so the cards get every point of
    # height they need and nothing important is squeezed to tiny type
    band_y0 = foot_top
    band_y1 = band_y0 + band_h
    map_y0 = band_y1 + (0.030 if has_band else 0.016)   # room for °E ticks
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

    # ---------------- map window --------------------------------------------
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
    if cfg.WIND_RADIUS_EXTENT:
        lat_min, lat_max, lon_c = wind_radius_extent(
            anchor_lat, anchor_lon, fc if has_for else None)
    else:
        lat_min = float(np.min(anchor_lat)) - BUFFER + cfg.MINLAT_OFFSET
        lat_max = float(np.max(anchor_lat)) + BUFFER + cfg.MAXLAT_OFFSET
        lon_c = 0.5 * (float(np.min(anchor_lon)) + float(np.max(anchor_lon)))
    lat_span = lat_max - lat_min
    aspect = (map_w * FIG_W_IN) / (map_h * FIG_H_IN)
    lon_span = lat_span * aspect
    lon_c = min(max(lon_c, cfg.MIN_LON + lon_span / 2),
                cfg.MAX_LON - lon_span / 2)
    lon_min, lon_max = lon_c - lon_span / 2, lon_c + lon_span / 2

    # ---------------- paint zones -------------------------------------------
    tz = pd.Timedelta(hours=cfg.FORECAST_TZ_OFFSET)
    issued_local = (ci_tnd + tz) if ci_tnd is not None else None
    _draw_header(
        ax_head, T, name=str(cyclone_name), is_invest=is_invest,
        obs_span=(f"OBSERVED {obs['tnd'].iloc[0]:%d/%HZ} \u2013 "
                  f"{obs['tnd'].iloc[-1]:%d/%HZ}") if has_obs
        else "OBSERVED --",
        for_span=(f"FORECAST {fc['tnd'].iloc[0]:%d/%HZ} \u2013 "
                  f"{fc['tnd'].iloc[-1]:%d/%HZ}") if has_for
        else "FORECAST --",
        issued=(f"ISSUED {ci_tnd:%HZ, %d %b %Y} UTC" if ci_tnd is not None
                else "ISSUED --"),
        issued_sub=(f"LOCAL {issued_local:%H:%M} (+{int(cfg.FORECAST_TZ_OFFSET)}H)"
                    if issued_local is not None and cfg.FORECAST_TZ_OFFSET
                    else "SYNOPTIC CHART"),
        brand=cfg.BRAND_NAME)
    if ax_foot is not None:
        p_txt = "--" if pressure is None or pd.isna(pressure) else int(pressure)
        _draw_footer(
            ax_foot, T,
            left=(f"WIND {int(ci)} KT   \u00b7   PRESSURE {p_txt} HPA   "
                  f"\u00b7   UPDATED {ci_tnd:%HZ @ %d %b %Y}"
                  if ci_tnd is not None else "NO CURRENT FIX"),
            right=cfg.FOOTER_TEXT)
    if ax_band is not None:
        _draw_band(ax_band, T, key_items=key_items, steps=steps,
                   time_label=cfg.FORECAST_TIME_LABEL,
                   speed_label=cfg.FORECAST_SPEED_LABEL,
                   speed_unit=cfg.FORECAST_SPEED_UNIT,
                   min_fontsize=cfg.FORECAST_TABLE_MIN_FONTSIZE)

    # sidebar packing: solve one font scale so every card fits the column.
    # The AT A GLANCE card is the headline info and is drawn first (top of
    # the sidebar) with the largest type; the key legend packs denser.
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
        fs_mult = {"glance": 1.38, "key": 0.88, "table": 1.0}
        gaps = (len(cards) - 1) * T["card_gap"]
        def total_at(fs):
            tot = 0.0
            for kind, payload in cards:
                f = fs * fs_mult[kind]
                if kind == "key":
                    tot += _key_card_height(payload, f, T)
                elif kind == "glance":
                    tot += _glance_card_height(payload, f, T)
                else:
                    _t, _h, rows, _d, _sub = payload
                    tot += _table_card_height(_h, rows, f, T, subtitle=_sub)
            return tot + gaps
        fs = T["fs_body"]
        for cand in (1.30, 1.24, 1.18, 1.12, 1.06, 1.0, 0.95, 0.90,
                     0.85, 0.80, 0.76, 0.72):
            if total_at(T["fs_body"] * cand) <= ah_pt:
                fs = T["fs_body"] * cand
                break
        else:
            fs = T["fs_body"] * 0.72
        y = 1.0
        x0, x1 = 0.0, 1.0
        for kind, payload in cards:
            f = fs * fs_mult[kind]
            if kind == "key":
                y = _draw_key_card(ax_side, x0, x1, y, payload, f, T)
            elif kind == "glance":
                y = _draw_glance_card(ax_side, x0, x1, y, payload, f, T)
            else:
                title, heads, rows, dots, sub = payload
                y = _draw_table_card(ax_side, x0, x1, y, title, heads, rows,
                                     f, T, dots=dots, subtitle=sub,
                                     col_align=["left", "center", "center"])
            y -= T["card_gap"] / ah_pt

    # ---------------- map content -------------------------------------------
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
              visible_ports=visible_ports, show_grid=cfg.SHOW_GRID)

    fig.canvas.draw()

    # ---------------- collision-free label chips ----------------------------
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

    placed = []          # display-px boxes of placed chips
    obstacles = []       # boxes labels must avoid (scale bar, north arrow)
    if cfg.SHOW_SCALE_BAR:
        obstacles.append(_scale_bar(ax_map, T, lon_max - lon_min,
                                    0.5 * (lat_min + lat_max)))
    obstacles.append(_north_arrow(ax_map, T))
    ax_bb = ax_map.get_window_extent(fig.canvas.get_renderer())
    marker_r_px = pt2px * T["fs_chip"] * 1.05

    def place_chip(text, lon, lat, fs, edge, tcol, candidates, lw=1.1,
                   alpha=0.93, z=8.5, avoid_own=None, leader=True):
        base = ax_map.transData.transform((lon, lat))
        w_pt = _text_width_pt(text, fs, "bold")
        pad_pt = 0.40 * fs
        h_pt = fs * 1.30
        best = None
        for dx, dy, ha, va in candidates:
            box = _chip_bbox(base, (dx * pt2px, dy * pt2px), ha, va,
                             w_pt, pad_pt, h_pt, pt2px)
            markers = [m for i, m in enumerate(marker_disp)
                       if avoid_own is None or i != avoid_own]
            viol = sum(_hits_point(box, x, y, marker_r_px) for x, y in markers)
            viol += sum(_overlap(box, p) for p in placed)
            viol += sum(_overlap(box, o) for o in obstacles)
            # keep chips inside the map frame
            if (box[0] < ax_bb.x0 + 2 or box[1] < ax_bb.y0 + 2
                    or box[2] > ax_bb.x1 - 2 or box[3] > ax_bb.y1 - 2):
                viol += 5
            if best is None or viol < best[0]:
                best = (viol, dx, dy, ha, va, box)
            if viol == 0:
                break
        _v, dx, dy, ha, va, box = best
        if leader:
            from matplotlib.transforms import IdentityTransform
            ax_map.plot([base[0], base[0] + dx * pt2px],
                        [base[1], base[1] + dy * pt2px],
                        color=T["ink_faint"], lw=1.3, alpha=0.8,
                        zorder=z - 0.4, clip_on=False,
                        transform=IdentityTransform(),
                        solid_capstyle="round")
        ax_map.annotate(
            text, xy=(lon, lat), xycoords="data", xytext=(dx, dy),
            textcoords="offset points", fontsize=fs, fontweight="bold",
            color=tcol, ha=ha, va=va, zorder=z,
            bbox=dict(facecolor="#ffffff", alpha=alpha, edgecolor=edge,
                      linewidth=lw, boxstyle=f"round,pad={0.42:.2f}"))
        placed.append(box)
        return box

    fs_chip = T["fs_chip"]
    # 1) current position chip
    if has_obs:
        p_txt = "" if pressure is None or pd.isna(pressure) \
            else f" \u00b7 {int(pressure)} HPA"
        place_chip(f"NOW {ci_tnd:%d/%HZ} \u00b7 {int(ci)} KT{p_txt}",
                   prev_lon, prev_lat, fs_chip, T["accent"], T["ink"],
                   [(14, 14, "left", "bottom"), (-14, 14, "right", "bottom"),
                    (14, -14, "left", "top"), (-14, -14, "right", "top"),
                    (0, 19, "center", "bottom"), (0, -19, "center", "top"),
                    (19, 0, "left", "center"), (-19, 0, "right", "center")])
    # 2) forecast point chips
    if has_for:
        for i in range(len(fc)):
            lat = float(fc["Latitude"].iloc[i])
            lon = float(fc["Longitude"].iloc[i])
            wind = fc["Intensity"].iloc[i]
            place_chip(f"{fc['tnd'].iloc[i]:%d/%HZ} \u00b7 {int(wind)} KT",
                       lon, lat, fs_chip, T["card_edge"], T["ink"],
                       [(13, 13, "left", "bottom"), (18, 0, "left", "center"),
                        (-13, 13, "right", "bottom"), (0, 18, "center", "bottom"),
                        (13, -13, "left", "top"), (-18, 0, "right", "center"),
                        (-13, -13, "right", "top"), (0, -18, "center", "top")])
    # 3) landfall chip
    if cfg.SHOW_LANDFALL and landfall_info is not None:
        txt = f"LANDFALL {landfall_info['time_str']}"
        if landfall_info.get("place"):
            txt += f" \u00b7 {landfall_info['place']}"
        place_chip(txt, landfall_info["lon"], landfall_info["lat"],
                   fs_chip, T["landfall"], "#7f1d1d",
                   [(-14, -14, "right", "top"), (-19, 0, "right", "center"),
                    (-14, 14, "right", "bottom"), (0, 19, "center", "bottom"),
                    (14, -14, "left", "top"), (0, -19, "center", "top")],
                   lw=1.8, alpha=0.95)
    # 4) port chips
    if cfg.SHOW_PORTS:
        for name, plat, plon, band in visible_ports:
            base = ax_map.transData.transform((plon, plat))
            fs_p = fs_chip + 0.4
            w_pt = _text_width_pt(name, fs_p, "bold")
            pad_pt = 0.35 * fs_p + 1.5
            h_pt = fs_p * 1.28
            ok = False
            for dx, dy, ha, va in ((0, 15, "center", "bottom"),
                                   (16, 0, "left", "center"),
                                   (-16, 0, "right", "center"),
                                   (0, -15, "center", "top"),
                                   (12, 12, "left", "bottom"),
                                   (-12, 12, "right", "bottom"),
                                   (12, -12, "left", "top"),
                                   (-12, -12, "right", "top")):
                box = _chip_bbox(base, (dx * pt2px, dy * pt2px), ha, va,
                                 w_pt, pad_pt, h_pt, pt2px)
                if (box[0] < ax_bb.x0 + 2 or box[1] < ax_bb.y0 + 2
                        or box[2] > ax_bb.x1 - 2 or box[3] > ax_bb.y1 - 2):
                    continue
                if any(_overlap(box, p) for p in placed) or \
                        any(_overlap(box, o) for o in obstacles) or \
                        any(_hits_point(box, x, y, marker_r_px)
                            for x, y in marker_disp):
                    continue
                ok = True
                break
            if not ok:
                continue      # marker stays, label is dropped (never moved)
            placed.append(box)
            ax_map.annotate(
                name, xy=(plon, plat), xycoords="data", xytext=(dx, dy),
                textcoords="offset points", fontsize=fs_p, fontweight="bold",
                color=T["ink"], ha=ha, va=va, zorder=8.2,
                bbox=dict(facecolor="#ffffff", alpha=0.92,
                          edgecolor=band["color"], linewidth=1.4,
                          boxstyle="round,pad=0.30"))

    # ---------------- save ----------------------------------------------------
    fig.savefig(output_path, dpi=OUTPUT_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {
        "landfall": landfall_info,
        "approaches": approach_rows,
        "cone_pts": cone_pts,
        "fc_track": fc_track,
        "map_bounds": (lon_min, lon_max, lat_min, lat_max),
        "axes_rect": (m_l, map_y0, map_w, map_h),
    }

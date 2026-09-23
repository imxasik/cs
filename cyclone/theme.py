"""
Design tokens for the cyclone track graphic ("the theme").

Every colour, line weight and type size used by ``cyclone/plotting.py``
lives here, in one place, so the whole graphic can be restyled by editing
this file only.  ``config.ini -> [style] theme`` selects a theme by name;
unknown names fall back to ``official``.

The tokens are plain data (hex colours + point sizes).  Nothing in here
imports matplotlib, so the module stays importable everywhere (Pydroid 3
included) and a new theme is just another dict of the same shape.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Intensity scale (1-minute sustained wind, knots).
# Thresholds are the project's long-standing convention; only the colours
# are part of the theme.  Ordered strongest -> weakest so a lookup stops at
# the first threshold the wind reaches.
# ---------------------------------------------------------------------------
INTENSITY_SCALE = (
    # (min wind kt, key,        legend label,       colour)
    (137, "cat5", "Category 5", "#7c3aed"),
    (114, "cat4", "Category 4", "#dc2626"),
    (97,  "cat3", "Category 3", "#ea580c"),
    (84,  "cat2", "Category 2", "#f59e0b"),
    (65,  "cat1", "Category 1", "#fde047"),
    (35,  "ts",   "Cyclonic Storm", "#22d3ee"),
    (24,  "td",   "Tropical Depression", "#3b82f6"),
    (0,   "low",  "Invest Area / Low", "#4ade80"),
)

# Display order for the map key: weakest -> strongest
# (Invest Area, Tropical Depression, ... Category 5).
INTENSITY_LEGEND_ORDER = tuple(reversed(INTENSITY_SCALE))

# Wind-radius rings (forecast isotachs), smallest radius first.
WIND_RADII = (
    # (column,   legend label, colour)
    ("WindR24", "24 KT Wind", "#0284c7"),
    ("WindR34", "34 KT Wind", "#ea580c"),
    ("WindR64", "64 KT Wind", "#be123c"),
)

# Port-risk bands, keyed exactly like cyclone.landfall.PORT_RISK_BANDS keys.
PORT_RISK = {
    "high":    "#dc2626",
    "medium":  "#f59e0b",
    "low":     "#16a34a",
    "unknown": "#94a3b8",
}

OFFICIAL = {
    "name": "official",

    # ---- page & panels ----------------------------------------------------
    "page_bg":     "#eef2f7",   # figure background
    "paper":       "#ffffff",   # card / table background
    "paper_tint":  "#f6f8fb",   # striped table rows
    "card_edge":   "#d5dee8",   # card + table borders
    "card_edge_soft": "#e6ecf3",
    "shadow":      "#0b2545",   # card drop-shadow tint (used at low alpha)

    # ---- ink (text) -------------------------------------------------------
    "ink":         "#0f2a4a",   # primary text (deep navy)
    "ink_soft":    "#44586d",   # secondary text
    "ink_faint":   "#7d8da0",   # captions, ticks
    "on_dark":     "#ffffff",   # text on navy surfaces

    # ---- brand / structure ------------------------------------------------
    "navy":        "#0b2545",   # header rule, footer band, table label col
    "accent":      "#c2255c",   # forecast track, header accent segment
    "accent_soft": "#e8c4d4",
    "grid":        "#8aa0b8",   # map graticule
    "map_edge":    "#b9c6d4",   # map frame

    # ---- basemap (vector land) ---------------------------------------------
    "sea":         "#d3e5f2",   # ocean
    "land":        "#eceadb",   # land fill (warm paper tone)
    "coast":       "#87a0b4",   # coastline stroke
    "coast_halo":  "#b7d3e6",   # shallow-water glow around the coast
    "border":      "#8f889c",   # country boundary (thin dashed)

    # ---- map graphics -----------------------------------------------------
    "obs_track":   "#33475c",   # observed track line
    "cone_fill":   "#64748b",
    "cone_edge":   "#475569",
    "landfall":    "#dc2626",
    "ai":          "#f59e0b",
    "bias":        "#7c3aed",

    # ---- type scale (points) ----------------------------------------------
    # Phone-friendly sizes: the graphic is usually viewed on a mobile screen,
    # so every label is sized to stay legible when the PNG is width-fit.
    "fs_title":    18.5,
    "fs_subtitle": 10.0,
    "fs_card_title": 10.0,
    "fs_body":     9.6,
    "fs_small":    8.4,
    "fs_tiny":     7.6,
    "fs_table":    10.4,
    "fs_chip":     9.8,
    "fs_footer":   9.8,
    "fs_tick":     9.2,

    # ---- metrics (points) --------------------------------------------------
    "card_pad":    9.0,         # inner padding of a card
    "card_gap":    10.0,        # gap between stacked cards (section spacing)
    "card_radius": 6.0,         # corner radius
    "row_h":       14.4,        # legend / table row height
    "line_card":   1.2,
    "line_ring":   1.7,
}


def get_theme(name: str | None = None) -> dict:
    """Return the theme dict for `name` (default/unknown -> official)."""
    return dict(OFFICIAL)


def wind_category(wind_kt) -> tuple:
    """(key, label, colour) of the intensity band a wind speed falls in."""
    try:
        w = float(wind_kt)
    except (TypeError, ValueError):
        w = 0.0
    for thr, key, label, colour in INTENSITY_SCALE:
        if w >= thr:
            return key, label, colour
    return INTENSITY_SCALE[-1][1:]


def wind_color(wind_kt) -> str:
    """Hex colour of the intensity band a wind speed falls in."""
    return wind_category(wind_kt)[2]


def wind_cat_label(wind_kt) -> str:
    """Short legend label ('CAT 3', 'TS', ...) of an intensity band."""
    key, _label, _c = wind_category(wind_kt)
    return {
        "cat5": "CAT 5", "cat4": "CAT 4", "cat3": "CAT 3", "cat2": "CAT 2",
        "cat1": "CAT 1", "ts": "TS", "td": "TD", "low": "LOW",
    }.get(key, "LOW")

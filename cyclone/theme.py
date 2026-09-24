"""
Design tokens for the cyclone track graphic — modern, gorgeous, mobile-first.

Every colour, line weight and type size lives here. config.ini [style] theme
selects a theme; unknown names fall back to modern.

This is the "super dynamic" edition:
- Mobile-first type scale (large, bold, high contrast)
- Soft, modern palette (slate + sky + rose)
- Larger corner radii, softer shadows, cleaner cards
- Dynamic zoom helpers for marker/text scaling
"""

from __future__ import annotations
import math

# ---------------------------------------------------------------------------
# Intensity scale (1-min sustained wind, knots)
# ---------------------------------------------------------------------------
INTENSITY_SCALE = (
    (137, "cat5", "Category 5", "#7c3aed"),   # violet
    (114, "cat4", "Category 4", "#dc2626"),   # red
    (97,  "cat3", "Category 3", "#ea580c"),   # orange-600
    (84,  "cat2", "Category 2", "#f59e0b"),   # amber-500
    (65,  "cat1", "Category 1", "#facc15"),   # yellow-400 (darker border)
    (35,  "ts",   "Cyclonic Storm", "#06b6d4"), # cyan-500
    (24,  "td",   "Tropical Depression", "#3b82f6"),   # blue-500
    (0,   "low",  "Invest Area", "#22c55e"),  # green-500
)

INTENSITY_LEGEND_ORDER = tuple(reversed(INTENSITY_SCALE))

WIND_RADII = (
    ("WindR34", "34 KT Wind", "#0ea5e9"),  # sky-500
    ("WindR64", "64 KT Wind", "#e11d48"),  # rose-600
)

# Risk colours are intentionally far apart in hue and luminance.  High risk
# is the requested vivid magenta; medium is amber/orange rather than red so
# the two bands remain instantly distinguishable on both paper and phones.
PORT_RISK = {
    "high":    "#d100ff",   # vivid magenta — immediate attention
    "medium":  "#f59e0b",   # amber — clearly distinct from magenta
    "low":     "#2563eb",   # blue — lower, but still visible
    "norisk":  "#16a34a",   # green — no immediate port risk
    "unknown": "#64748b",   # slate — only for unavailable estimates
}

# ---------------------------------------------------------------------------
# MODERN GORGEOUS THEME — mobile-first, ultra clean
# ---------------------------------------------------------------------------
MODERN = {
    "name": "modern",

    # ---- page & panels ----------------------------------------------------
    "page_bg":       "#f1f5f9",   # slate-100 — soft, eye-friendly
    "paper":         "#ffffff",
    "paper_tint":    "#f8fafc",   # slate-50 — striped rows
    "card_edge":     "#e2e8f0",   # slate-200 — crisp border
    "card_edge_soft":"#f1f5f9",   # slate-100 — inner dividers
    "shadow":        "#0f172a",   # slate-900 — shadow tint

    # ---- ink --------------------------------------------------------------
    "ink":           "#0f172a",   # slate-900 — primary
    "ink_soft":      "#475569",   # slate-600 — secondary
    "ink_faint":     "#94a3b8",   # slate-400 — captions
    "on_dark":       "#ffffff",

    # ---- brand / structure ------------------------------------------------
    "navy":          "#0f172a",   # header, footer, label column
    "accent":        "#e11d48",   # rose-600 — forecast track, accent
    "accent_soft":   "#ffe4e6",   # rose-100
    "grid":          "#94a3b8",   # slate-400 — visible, restrained graticule
    "map_edge":      "#64748b",   # slate-500 — map frame

    # ---- basemap ----------------------------------------------------------
    "sea":           "#e0f2fe",   # sky-100 — modern light ocean
    "land":          "#fefce8",   # yellow-50 — warm paper land
    "coast":         "#111827",   # near-black — crisp primary coastline
    "coast_halo":    "#bfdbfe",   # blue-200 — restrained water halo
    "border":        "#64748b",   # slate-500 — country border
    "state_border":  "#94a3b8",   # slate-400 — state border (lighter)
    "state_width":   0.6,
    "state_style":   ":",

    # ---- map graphics -----------------------------------------------------
    "obs_track":     "#334155",   # slate-700
    "cone_fill":     "#94a3b8",   # slate-400
    "cone_edge":     "#475569",   # slate-600
    "landfall":      "#dc2626",   # red-600
    "overland":      "#7c3aed",   # violet — status chip above forecast table
    "summary_line":  "#b8c4d3",   # clearly visible summary dividers

    # ---- type scale — MOBILE FIRST, large & bold -------------------------
    # These are base sizes; dynamic zoom will scale them further
    "fs_title":      28.0,   # was 23 — much larger for mobile
    "fs_subtitle":   14.0,   # was 12 — readable on phone
    "fs_card_title": 14.5,   # card titles
    "fs_body":       13.5,   # body text
    "fs_small":      11.5,
    "fs_tiny":       10.5,
    "fs_table":      13.5,   # forecast table
    "fs_chip":       14.0,   # map chips (NOW, forecast points)
    "fs_footer":     12.5,
    "fs_tick":       12.5,   # lat/lon ticks

    # ---- metrics — more breathing room -----------------------------------
    "card_pad":      14.0,   # inner padding
    "card_gap":      18.0,   # gap between cards
    "card_radius":   12.0,   # larger radius = more modern
    "row_h":         20.0,
    "line_card":     1.2,
    "line_ring":     2.8,    # thicker wind rings for clarity

    # ---- modern extras ----------------------------------------------------
    "chip_radius":   6.0,
    "shadow_alpha":  0.08,
    "shadow_blur":   12.0,
}

# Keep official for backward compat, but make it modern-ish too
OFFICIAL = dict(MODERN)
OFFICIAL["name"] = "official"
OFFICIAL.update({
    "page_bg": "#eef2f7",
    "sea": "#d3e5f2",
    "land": "#eceadb",
    "coast": "#111827",
    "coast_halo": "#bfdbfe",
    "border": "#64748b",
    "accent": "#c2255c",
})

THEMES = {
    "modern": MODERN,
    "official": OFFICIAL,
    "gorgeous": MODERN,
    "clean": MODERN,
}

def get_theme(name: str | None = None) -> dict:
    """Return theme dict for name, default modern."""
    if not name:
        return dict(MODERN)
    key = str(name).strip().lower()
    return dict(THEMES.get(key, MODERN))


def wind_category(wind_kt) -> tuple:
    try:
        w = float(wind_kt)
    except (TypeError, ValueError):
        w = 0.0
    for thr, key, label, colour in INTENSITY_SCALE:
        if w >= thr:
            return key, label, colour
    return INTENSITY_SCALE[-1][1:]

def wind_color(wind_kt) -> str:
    return wind_category(wind_kt)[2]

def wind_cat_label(wind_kt) -> str:
    key, _label, _c = wind_category(wind_kt)
    return {
        "cat5": "CAT 5", "cat4": "CAT 4", "cat3": "CAT 3", "cat2": "CAT 2",
        "cat1": "CAT 1", "ts": "TS", "td": "TD", "low": "LOW",
    }.get(key, "LOW")

# ---------------------------------------------------------------------------
# Dynamic zoom helpers — super dynamic scaling
# ---------------------------------------------------------------------------
def zoom_factor(lat_span: float, lon_span: float, ref_span: float = 10.0) -> float:
    """
    Compute dynamic zoom factor from map span.
    Small span (zoomed in) => factor >1 (larger markers/text)
    Large span (zoomed out) => factor <1 (smaller markers/text)
    Clamped to 0.65 – 2.2 for readability.
    """
    try:
        span = max(float(lat_span), float(lon_span))
    except Exception:
        span = ref_span
    if span <= 0.1:
        span = 0.1
    # inverse relation: factor = ref / span, but smoothed
    raw = ref_span / span
    # soft clamp with log-like curve for natural feel
    # map raw 0.3..3.0 to 0.65..2.2
    f = math.log(max(0.2, raw) + 1) * 1.2
    return max(0.65, min(2.2, f))

def dynamic_font(base_fs: float, zf: float, min_fs: float = 8.0, max_fs: float = 18.0) -> float:
    """Scale font size by zoom factor with limits."""
    return max(min_fs, min(max_fs, base_fs * (0.85 + 0.30 * zf)))

def dynamic_marker_size(base_size: float, zf: float) -> float:
    """Scale marker sizes with zoom factor."""
    return base_size * (0.7 + 0.5 * zf)

def dynamic_linewidth(base_lw: float, zf: float) -> float:
    """Scale line widths with zoom factor."""
    return base_lw * (0.8 + 0.35 * zf)

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from matplotlib.lines import Line2D

from .config import (
    BUFFER, UCR,
    MIN_LAT, MAX_LAT, MIN_LON, MAX_LON,
    MINLAT_OFFSET, MAXLAT_OFFSET,
    OUTPUT_DPI,
    SHOW_CONE, SHOW_LEGEND, SHOW_PORTS, SHOW_PORT_TABLE,
    SHOW_MOVEMENT_TABLE, SHOW_ACE_BOX, SHOW_MAX_WIND_BOXES,
    SHOW_FOOTER, SHOW_AI_POSITION, SHOW_BIAS_TRACK,
    SHOW_LANDFALL, SHOW_APPROACH_TABLE, APPROACH_RADIUS, APPROACH_PORTS,
    DATE_FORMAT, FOOTER_TEXT,
)
from .cone import create_nhc_cone
from .geo import haversine, get_bearing, get_cardinal_direction
from .ace import calculate_ace
from .landfall import (
    find_landfall,
    port_centre_table,
    current_centre,
    classify_port_risk,
)
from .ports import BOB
from features.ailoc import predict_landfall_latlon
from features.aibc import apply_simple_bias

# Repo-root assets folder (works no matter what the current directory is)
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


CAT = [
    (137, "mediumpurple", "CAT 5"),
    (114, "fuchsia",      "CAT 4"),
    (97,  "tomato",       "CAT 3"),
    (84,  "gold",         "CAT 2"),
    (65,  "lemonchiffon", "CAT 1"),
    (35,  "aqua",         "TS"),
    (24,  "steelblue",    "TD"),
    (0,   "lime",         "LOW"),
]


def wind_color(w):
    for thr, col, _ in CAT:
        if w >= thr:
            return col
    return "lime"


def wind_cat(w):
    for thr, _, name in CAT:
        if w >= thr:
            return name
    return "LOW"


# --------------------------------------------------------------------------
# Dynamic table sizing
# --------------------------------------------------------------------------
# Extra headroom on top of the measured glyph width: TextPath gives the ink
# extent, while the renderer also counts side bearings / advance width. The
# small safety factor keeps text comfortably inside its cell.
_TEXT_SAFETY = 1.05


def _text_width_pt(text, fontsize, weight="normal"):
    """
    Width of `text` in points, measured from the actual font glyphs (no
    canvas draw needed) plus a small safety margin. Falls back to a
    character-count estimate.
    """
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
    """
    Shorten `text` with an ellipsis until it fits `max_width_pt`, so a very
    long name can never stick out of its table cell.
    """
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


def _add_dynamic_table(ax, col_labels, rows, *, fontsize=11.0,
                       min_fontsize=6.5, x=0.005, y=0.045,
                       max_width_frac=0.34, cell_pad_frac=0.45,
                       row_height_frac=1.9, caption=None, zorder=7):
    """
    Draw a matplotlib table whose box is sized from the real text extents,
    so long cell contents (e.g. "Krishnapatnam", "Mawlamyine") can never
    spill outside the table box.

    The font is first shrunk (down to `min_fontsize`) to keep the table
    within `max_width_frac` of the axes; if it is *still* too wide, the
    longest cells are ellipsised. Either way the table stays inside its
    box and the box stays inside the map.

    Returns (table, bbox) where bbox is [x, y, w, h] in axes fractions.
    """
    fig = ax.figure
    dpi = fig.dpi
    ax_pos = ax.get_position()
    ax_w_px = max(1.0, ax_pos.width * fig.get_figwidth() * dpi)
    ax_h_px = max(1.0, ax_pos.height * fig.get_figheight() * dpi)

    pt_to_px = dpi / 72.0
    n_cols = len(col_labels)
    limit_px = max_width_frac * ax_w_px
    rows = [[str(c) for c in row] for row in rows]

    def pack(fs, pad_px):
        """Column widths in px needed by the current text at font size fs."""
        widths = []
        for c in range(n_cols):
            w = _text_width_pt(col_labels[c], fs, weight="bold")
            for row in rows:
                w = max(w, _text_width_pt(row[c], fs))
            widths.append(w * pt_to_px + 2.0 * pad_px)
        return widths

    # 1) shrink the font to fit the width budget
    pad_px = cell_pad_frac * fontsize * pt_to_px
    cols_px = pack(fontsize, pad_px)
    total_px = sum(cols_px)
    if total_px > limit_px and total_px > 0:
        fontsize = max(min_fontsize, fontsize * (limit_px / total_px))
        pad_px = cell_pad_frac * fontsize * pt_to_px
        cols_px = pack(fontsize, pad_px)
        total_px = sum(cols_px)

    # 2) still too wide -> ellipsise the cells to their column budget
    if total_px > limit_px:
        share = [c * (limit_px / total_px) for c in cols_px]
        budget_pt = [(w - 2.0 * pad_px) / pt_to_px for w in share]
        col_labels = [_fit_text(lbl, budget_pt[c], fontsize, "bold")
                      for c, lbl in enumerate(col_labels)]
        rows = [[_fit_text(cell, budget_pt[c], fontsize)
                 for c, cell in enumerate(row)] for row in rows]
        cols_px = pack(fontsize, pad_px)
        total_px = min(sum(cols_px), limit_px)

    row_h_px = fontsize * row_height_frac * pt_to_px
    bbox = [x, y, total_px / ax_w_px,
            (row_h_px * (len(rows) + 1)) / ax_h_px]

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc='center',
        colColours=['#f0f0f0'] * n_cols,
        zorder=zorder,
        bbox=bbox,
        colWidths=[c / total_px for c in cols_px],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (r, _c), cell in table.get_celld().items():
        cell.set_linewidth(0.7)
        cell.PAD = 0.06
        if r == 0:
            cell.set_text_props(fontweight='bold')

    if caption:
        ax.text(
            x, y + bbox[3] + 0.006, caption,
            transform=ax.transAxes,
            fontsize=max(min_fontsize, fontsize - 1.0),
            fontweight='bold', ha='left', va='bottom',
            bbox=dict(facecolor='white', alpha=0.65, edgecolor='none',
                      boxstyle='round,pad=0.25'),
            zorder=zorder,
        )

    return table, bbox


def plot_cyclone(cyclone_name, track_data_obs, track_data_for, is_invest,
                 map_image_path, output_path):

    # -------------- BACKGROUND MAP --------------------
    if os.path.exists(map_image_path):
        background_image = plt.imread(map_image_path)
    else:
        background_image = None

    lat_min = track_data_for["Latitude"].min() - BUFFER + MINLAT_OFFSET
    lat_max = track_data_for["Latitude"].max() + BUFFER + MAXLAT_OFFSET
    lon_min = track_data_for["Longitude"].min() - BUFFER - 0.5
    lon_max = track_data_for["Longitude"].max() + BUFFER + 1

    fig, ax = plt.subplots(figsize=(11, 10), dpi=OUTPUT_DPI)
    ax.set_xlim([lon_min, lon_max])
    ax.set_ylim([lat_min, lat_max])

    if background_image is not None:
        ax.imshow(background_image, extent=[MIN_LON, MAX_LON, MIN_LAT, MAX_LAT])
        ax.set_aspect("equal", adjustable="datalim")

    # ---------------- OBSERVED TRACK -----------------
    track_prev_lat = track_data_obs["Latitude"].iloc[0]
    track_prev_lon = track_data_obs["Longitude"].iloc[0]

    prev_conditions = [
        ("Invest Area / Low", 'lime', 'full'),
        ("Tropical Depression", 'steelblue', 'full'),
        ("Cyclonic Storm", 'aqua', 'full'),
        ("Category 1", 'lemonchiffon', 'full'),
        ("Category 2", 'gold', 'full'),
        ("Category 3", 'tomato', 'full'),
        ("Category 4", 'fuchsia', 'full'),
        ("Category 5", 'mediumpurple', 'full'),
        (" ", '', 'none'),
        ("64 KT Wind Radius", 'white', 'none'),
        (" ", ' ', 'none'),
        ("34 KT Wind Radius", 'white', 'none'),
        (" ", ' ', 'none'),
        ("24 KT Wind Radius", 'white', 'none'),
        (" ", ' ', 'none')
    ]

    legend_elements_prev = [
        Line2D(
            [0], [0],
            marker='o',
            color='w' if condition == " " else
                   'magenta' if condition == "64 KT Wind Radius" else
                   'red' if condition == "34 KT Wind Radius" else
                   'blue' if condition == "24 KT Wind Radius" else
                   'black',
            markerfacecolor=color,
            markersize=25 if condition == "24 KT Wind Radius" else
                        20 if condition == "34 KT Wind Radius" else
                        15 if condition == "64 KT Wind Radius" else
                        8,
            fillstyle=fillstyle,
            label=condition,
            lw=0
        )
        for condition, color, fillstyle in prev_conditions
    ]

    for lat, lon, wind in zip(
        track_data_obs["Latitude"],
        track_data_obs["Longitude"],
        track_data_obs["Intensity"]
    ):
        ax.plot([track_prev_lon, lon], [track_prev_lat, lat],
                "k--", lw=1, zorder=5)
        ax.plot(lon, lat, "o", ms=7, mec="k",
                color=wind_color(wind), zorder=6)
        track_prev_lat, track_prev_lon = lat, lon

    # Last observed info
    prev_lat = track_data_obs["Latitude"].iloc[-1]
    prev_lon = track_data_obs["Longitude"].iloc[-1]
    ci = track_data_obs["Intensity"].iloc[-1]
    ci_tnd = track_data_obs["tnd"].iloc[-1]

    # --------- CONE & FORECAST TRACK -----------
    n_forecast = len(track_data_for)

    # store whichever forecast track we actually use
    fc_track_lon = None  # <<< NEW
    fc_track_lat = None  # <<< NEW

    if SHOW_CONE and n_forecast >= 2:
        # Use cone only when at least 2 points
        extended_lons = np.concatenate([[prev_lon], track_data_for["Longitude"].values])
        extended_lats = np.concatenate([[prev_lat], track_data_for["Latitude"].values])

        try:
            cone_points, smooth_lon, smooth_lat = create_nhc_cone(
                extended_lons, extended_lats,
                initial_uncertainty=0.00,
                growth_rate=UCR,
            )

            cone = Polygon(
                cone_points, closed=False,
                facecolor='lightgray', edgecolor='gray',
                alpha=0.6, linewidth=1.0, zorder=3
            )
            ax.add_patch(cone)

            ax.plot(
                smooth_lon, smooth_lat,
                '-', color='magenta', linewidth=2.0,
                alpha=1.0, label='Forecast Track', zorder=4
            )

            fc_track_lon = np.asarray(smooth_lon)  # <<< NEW
            fc_track_lat = np.asarray(smooth_lat)  # <<< NEW

        except Exception as e:
            print(f"[WARN] Cone error ({e}) - drawing line only")
            # fall back to a simple line including last obs
            lons = np.concatenate([[prev_lon], track_data_for["Longitude"].values])
            lats = np.concatenate([[prev_lat], track_data_for["Latitude"].values])
            ax.plot(
                lons, lats,
                '-', color='magenta', linewidth=2.0,
                alpha=1.0, label='Forecast Track', zorder=4
            )

            fc_track_lon = np.asarray(lons)  # <<< NEW
            fc_track_lat = np.asarray(lats)  # <<< NEW

    elif n_forecast >= 1:
        # No cone or no points for cone
        # track from last obs through forecasts
        lons = np.concatenate([[prev_lon], track_data_for["Longitude"].values])
        lats = np.concatenate([[prev_lat], track_data_for["Latitude"].values])
        ax.plot(
            lons, lats,
            '-', color='magenta', linewidth=2.0,
            alpha=1.0, label='Forecast Track', zorder=4
        )

        fc_track_lon = np.asarray(lons)  # <<< NEW
        fc_track_lat = np.asarray(lats)  # <<< NEW

    else:
        print("[INFO] No forecast data available for track/cone.")

    # ---- Bias-Corrected TRACK (Dashed Magenta) ----
    if SHOW_BIAS_TRACK and fc_track_lon is not None and fc_track_lat is not None:
        # Estimate last direction of motion from last two obs points
        if len(track_data_obs) >= 2:
            coord1 = (
                track_data_obs["Latitude"].iloc[-2],
                track_data_obs["Longitude"].iloc[-2],
            )
            coord2 = (
                track_data_obs["Latitude"].iloc[-1],
                track_data_obs["Longitude"].iloc[-1],
            )
            last_dir_deg = get_bearing(coord1, coord2)
        else:
            last_dir_deg = None

        bc_lats, bc_lons = apply_simple_bias(
            fc_track_lat,
            fc_track_lon,
            last_lon=prev_lon,    # east/west
            last_wind=ci,         # weak/strong (64 kt)
            last_dir=last_dir_deg # west/south/east/north
        )

        ax.plot(
            bc_lons, bc_lats,
            '--', color='magenta', linewidth=1.5,
            alpha=0.7, label='Bias-Corrected Track', zorder=4
        )


    # ------- LANDFALL ESTIMATE & PORT TABLE --------
    landfall_info = None
    approach_rows = []

    # Port colours are also driven by the landfall estimate, so calculate it
    # whenever ports are visible (not only when the landfall X is enabled).
    if SHOW_LANDFALL or SHOW_PORTS or SHOW_APPROACH_TABLE or SHOW_PORT_TABLE:
        try:
            landfall_info = find_landfall(track_data_obs, track_data_for, BOB())

            if SHOW_LANDFALL:
                if landfall_info is not None:
                    where = (f"near {landfall_info['place']}"
                             if landfall_info['place']
                             else f"at {landfall_info['lat']}N, {landfall_info['lon']}E")
                    print(f"[LANDFALL] Est. landfall: {landfall_info['time_str']} "
                          f"{where} ({landfall_info['time']:%d %b %Y %H:%M} UTC)")
                else:
                    print("[LANDFALL] No landfall detected within the forecast period.")

            if SHOW_APPROACH_TABLE or SHOW_PORT_TABLE:
                # One merged table: the ports closest to the landfall
                # point (or to the track when there is no landfall), showing
                # the distance & direction of the *current* centre from each.
                approach_rows = port_centre_table(
                    track_data_obs, track_data_for, BOB(),
                    landfall=landfall_info,
                    radius_km=APPROACH_RADIUS,
                    top=APPROACH_PORTS,
                )
                if approach_rows:
                    centre = current_centre(track_data_obs, track_data_for)
                    where = (f"({centre[0]:.1f}N, {centre[1]:.1f}E)"
                             if centre else "(unknown)")
                    tops = " | ".join(
                        f"{r['name']} {r['dist_km']} km {r['dir_str']}"
                        for r in approach_rows
                    )
                    print(f"[PORTS] From current centre {where}: {tops}")
        except Exception as e:
            print(f"[WARN] Landfall/approach estimate failed: {e}")

    if SHOW_LANDFALL and landfall_info is not None:
        lf_lon, lf_lat = landfall_info["lon"], landfall_info["lat"]
        ax.plot(lf_lon, lf_lat, "X", ms=11, mec="k",
                mfc="red", zorder=9)
        # The label itself is drawn after the forecast point labels below,
        # so that its overlap check can see them (see "LANDFALL LABEL").

    # ------- FORECAST POINTS + LABELS --------
    label_positions = []  # in pixel space

    for index, (lat, lon, wind, wr24, wr34, wr64) in enumerate(
        zip(
            track_data_for["Latitude"],
            track_data_for["Longitude"],
            track_data_for["Intensity"],
            track_data_for["WindR24"],
            track_data_for["WindR34"],
            track_data_for["WindR64"]
        )
    ):
        # Marker
        ax.plot(lon, lat, "o", ms=7, mec='k',
                color=wind_color(wind), zorder=6)

        # Wind radii
        for wr, color, zord in [(wr24, 'blue', 1), (wr34, 'red', 2), (wr64, 'magenta', 3)]:
            if pd.notna(wr) and float(wr) > 0:
                r = float(wr)
                ax.add_patch(plt.Circle((lon, lat), r, color=color, alpha=0.05, zorder=zord))
                ax.add_patch(
                    plt.Circle(
                        (lon, lat), r, fill=False,
                        edgecolor=color, linewidth=1,
                        alpha=0.3 if color == 'blue' else 0.6 if color == 'red' else 1.0,
                        zorder=zord
                    )
                )

        # Label with constant screen offset + overlap check
        tnd_str = track_data_for['tnd'].iloc[index].strftime("%d/%H")
        label_text = f"{tnd_str}, {wind}KT"

        fig = ax.figure
        dpi = fig.dpi

        candidate_offsets = [
            (5, 5, 'left', 'bottom'),      # up-right
            (10, 0, 'left', 'center'),     # right
            (-5, 5, 'right', 'bottom'),    # up-left
            (0, 10, 'center', 'bottom'),   # above
            (5, -5, 'left', 'top'),        # down-right
        ]

        min_dist_px = 15
        base_disp = ax.transData.transform((lon, lat))

        placed = False
        for dx_pt, dy_pt, ha, va in candidate_offsets:
            dx_px = dx_pt * dpi / 72.0
            dy_px = dy_pt * dpi / 72.0
            cand_disp = base_disp + np.array([dx_px, dy_px])

            if all(
                np.hypot(cand_disp[0] - x, cand_disp[1] - y) > min_dist_px
                for (x, y) in label_positions
            ):
                label_positions.append((cand_disp[0], cand_disp[1]))

                ax.annotate(
                    label_text,
                    xy=(lon, lat),
                    xycoords='data',
                    xytext=(dx_pt, dy_pt),
                    textcoords='offset points',
                    fontsize=7,
                    fontweight='bold',
                    fontfamily='arial',
                    zorder=5,
                    bbox=dict(
                        facecolor='white',
                        alpha=0.6,
                        boxstyle='round,pad=0.50'
                    ),
                    ha=ha,
                    va=va
                )
                placed = True
                break

        if not placed:
            ax.annotate(
                label_text,
                xy=(lon, lat),
                xycoords='data',
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=6,
                fontweight='bold',
                fontfamily='arial',
                zorder=5,
                bbox=dict(
                    facecolor='white',
                    alpha=0.6,
                    boxstyle='round,pad=0.50'
                ),
                ha='left',
                va='bottom'
            )

    # -------------------- LANDFALL LABEL --------------------
    # Drawn after the forecast labels so the overlap check can see them.
    # Mirrors the forecast labels: they sit up-right of their marker, the
    # landfall label sits down-left of the red X.
    if SHOW_LANDFALL and landfall_info is not None:
        lf_label = f"LF-{landfall_info['time_str']}"

        fig = ax.figure
        dpi = fig.dpi

        lf_candidates = [
            (-5, -5, 'right', 'top'),     # down-left (default)
            (-10, 0, 'right', 'center'),  # left
            (-5, 5, 'right', 'bottom'),   # up-left
            (0, 10, 'center', 'bottom'),  # above
            (5, -5, 'left', 'top'),       # down-right
        ]

        lf_min_dist_px = 15
        lf_base_disp = ax.transData.transform((lf_lon, lf_lat))
        lf_dx_pt, lf_dy_pt, lf_ha, lf_va = lf_candidates[0]

        for dx_pt, dy_pt, ha, va in lf_candidates:
            dx_px = dx_pt * dpi / 72.0
            dy_px = dy_pt * dpi / 72.0
            cand_disp = lf_base_disp + np.array([dx_px, dy_px])

            if all(
                np.hypot(cand_disp[0] - x, cand_disp[1] - y) > lf_min_dist_px
                for (x, y) in label_positions
            ):
                lf_dx_pt, lf_dy_pt, lf_ha, lf_va = dx_pt, dy_pt, ha, va
                label_positions.append((cand_disp[0], cand_disp[1]))
                break

        ax.annotate(
            lf_label,
            xy=(lf_lon, lf_lat),
            xycoords='data',
            xytext=(lf_dx_pt, lf_dy_pt),
            textcoords='offset points',
            fontsize=8,
            fontweight='bold',
            ha=lf_ha,
            va=lf_va,
            bbox=dict(
                facecolor='white',
                alpha=0.8,
                edgecolor='darkred',
                boxstyle='round,pad=0.3'
            ),
            zorder=10
        )

  # -------------------- LEGEND --------------------
    if SHOW_LEGEND:
        # Keep the port-risk key separate from the long cyclone-status
        # legend.  This follows the reference layout: a compact, horizontal
        # key at the top of the map with Low / Mod / High in that order.
        if SHOW_PORTS:
            port_risk_handles = [
                Line2D(
                    [0], [0], marker='o', linestyle='None',
                    markerfacecolor='#22b957', markeredgecolor='black',
                    markeredgewidth=0.8, markersize=9,
                    label='Low Risk',
                ),
                Line2D(
                    [0], [0], marker='o', linestyle='None',
                    markerfacecolor='#ff9500', markeredgecolor='black',
                    markeredgewidth=0.8, markersize=9,
                    label='Mod Risk',
                ),
                Line2D(
                    [0], [0], marker='o', linestyle='None',
                    markerfacecolor='#e60000', markeredgecolor='black',
                    markeredgewidth=0.8, markersize=9,
                    label='High Risk',
                ),
            ]
            port_risk_legend = ax.legend(
                handles=port_risk_handles,
                loc='upper center',
                bbox_to_anchor=(0.52, 0.995),
                ncol=3,
                fontsize=10,
                frameon=True,
                fancybox=True,
                framealpha=0.96,
                edgecolor='#cbd5e1',
                borderpad=0.50,
                handletextpad=0.50,
                columnspacing=1.25,
            )
            for text in port_risk_legend.get_texts():
                text.set_fontweight('bold')
                text.set_color('#1f2937')
            port_risk_legend.set_zorder(10000)
            port_risk_legend.get_frame().set_zorder(10000)
            # A second ax.legend call below would otherwise replace it.
            ax.add_artist(port_risk_legend)

    # Bias-corrected track (dashed magenta)
        if SHOW_BIAS_TRACK:
          legend_elements_prev.append(
            Line2D(
                [0], [0],
                linestyle='--',
                color='magenta',
                lw=1.5,
                label='AI-BC Track'
            )
        )
            
        # Forecast track (solid magenta)
        legend_elements_prev.append(
            Line2D(
                [0], [0],
                linestyle='-',
                color='magenta',
                lw=2.0,
                label='Forecast Track'
            )
        )
    
        # Cone (if shown)
        if SHOW_CONE and n_forecast >= 1:
            legend_elements_prev.append(
                Patch(
                    facecolor='lightgray',
                    edgecolor='gray',
                    alpha=0.3,
                    label='Uncertainty Cone'
                )
            )

   # Insert AI Position right after Category 5
        if SHOW_AI_POSITION:
            legend_elements_prev.insert(
                8,
                Line2D(
                    [0], [0],
                    marker='*',
                    color='k',
                    markerfacecolor='yellow',
                    markersize=12,
                    lw=0,
                    label='AI Position'
                )
            )

        if SHOW_LANDFALL and landfall_info is not None:
            legend_elements_prev.append(
                Line2D(
                    [0], [0],
                    marker='X',
                    color='k',
                    markerfacecolor='red',
                    markersize=9,
                    lw=0,
                    label='Landfall Est.'
                )
            )

        legend = ax.legend(
            handles=legend_elements_prev,
            loc='upper right',
            title='INTENSITY SCALE'
        )
        legend.get_title().set_fontweight('bold')

  # Make sure legend is above all plotted data
        legend.set_zorder(9999)
        legend.get_frame().set_zorder(9999)
        

    # -------------------- TIME STRINGS --------------------
    observed_start_time = track_data_obs['tnd'].iloc[0].strftime(DATE_FORMAT)
    observed_end_time = track_data_obs['tnd'].iloc[-1].strftime(DATE_FORMAT)
    forecast_start_time = track_data_for['tnd'].iloc[0].strftime(DATE_FORMAT)
    forecast_end_time = track_data_for['tnd'].iloc[-1].strftime(DATE_FORMAT)

    # -------------------- PORT TABLE (merged) --------------------
    # Ports are chosen as the APPROACH_PORTS closest to the landfall point
    # (closest to the track if there is no landfall), but the table answers
    # the "right now" question: how far is the current centre from each
    # port and in which direction.
    if (SHOW_APPROACH_TABLE or SHOW_PORT_TABLE) and approach_rows:
        table_rows = [
            [r["name"], f"{r['dist_km']} km", r["dir_str"]]
            for r in approach_rows
        ]
        _centre = current_centre(track_data_obs, track_data_for)
        _where = (f" ({_centre[0]:.1f}N, {_centre[1]:.1f}E)"
                  if _centre is not None else "")
        if landfall_info is not None:
            _caption = (f"{len(table_rows)} PORTS NEAREST LANDFALL"
                        f" \u00b7 DIST FROM CURRENT CENTRE{_where}")
        else:
            _caption = (f"{len(table_rows)} PORTS NEAREST THE TRACK"
                        f" \u00b7 DIST FROM CURRENT CENTRE{_where}")
        _add_dynamic_table(
            ax,
            ["PORT", "DIS", "DIR"],
            table_rows,
            fontsize=10.0,
            x=0.005, y=0.045,
            max_width_frac=0.30,
            row_height_frac=1.85,
            caption=_caption,
        )
    elif SHOW_APPROACH_TABLE or SHOW_PORT_TABLE:
        ax.text(
            0.01, 0.05,
            "No ports within approach range",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom", ha="left",
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
            zorder=7
        )

    # -------------------- PORT MARKERS & LABELS --------------------
    # Every visible port gets a colour based on its distance to the estimated
    # landfall, rather than the current centre or the closest forecast point.
    # This keeps the map markers and the PORT RISK legend consistent.
    visible_cities = []
    if SHOW_PORTS:
        for location, (plat, plon) in BOB().items():
            if lat_min <= plat <= lat_max and lon_min <= plon <= lon_max:
                if landfall_info is not None:
                    landfall_distance = haversine(
                        (plat, plon),
                        (landfall_info["lat"], landfall_info["lon"]),
                    )
                else:
                    landfall_distance = None
                visible_cities.append((
                    location,
                    plat,
                    plon,
                    classify_port_risk(landfall_distance),
                ))

    if SHOW_PORTS:
        # City labels use real screen-space rectangles, not just marker-centre
        # distances.  That matters for neighbouring ports such as Balasore,
        # Digha and Contai, whose names have very different widths.
        city_label_bboxes = []
        fig = ax.figure
        dpi = fig.dpi
        label_fontsize = 10.0
        pt_to_px = dpi / 72.0
        label_pad_pt = 0.26 * label_fontsize + 2.0
        label_height_pt = label_fontsize * 1.25 + 2.0 * label_pad_pt

        # Keep every label in its normal position above its marker. If that
        # fixed position collides with another label, suppress the label but
        # keep its risk-coloured marker visible; no label is moved around the
        # map.
        label_candidates = [
            (0, 10, 'center', 'bottom'),
        ]

        def label_bbox(base_disp, dx_pt, dy_pt, ha, va, width_pt):
            """Approximate an annotation's padded bbox in display pixels."""
            anchor = base_disp + np.array([
                dx_pt * pt_to_px,
                dy_pt * pt_to_px,
            ])
            width_px = (width_pt + 2.0 * label_pad_pt) * pt_to_px
            height_px = label_height_pt * pt_to_px

            if ha == 'left':
                x0, x1 = anchor[0], anchor[0] + width_px
            elif ha == 'right':
                x0, x1 = anchor[0] - width_px, anchor[0]
            else:
                x0, x1 = anchor[0] - width_px / 2.0, anchor[0] + width_px / 2.0

            if va == 'bottom':
                y0, y1 = anchor[1], anchor[1] + height_px
            elif va == 'top':
                y0, y1 = anchor[1] - height_px, anchor[1]
            else:
                y0, y1 = anchor[1] - height_px / 2.0, anchor[1] + height_px / 2.0
            return (x0, y0, x1, y1)

        def boxes_overlap(box_a, box_b, gap_px=3.0):
            return not (
                box_a[2] + gap_px <= box_b[0]
                or box_b[2] + gap_px <= box_a[0]
                or box_a[3] + gap_px <= box_b[1]
                or box_b[3] + gap_px <= box_a[1]
            )

        # For the remaining special coastal label, retain the marker but hide
        # only the text when landfall is directly over the port (<100 km) or
        # that port is the closest one to the estimated landfall.
        hidden_near_landfall_ports = set()
        if landfall_info is not None:
            landfall_coord = (landfall_info["lat"], landfall_info["lon"])
            all_landfall_distances = {
                name: haversine((plat, plon), landfall_coord)
                for name, (plat, plon) in BOB().items()
            }
            closest_landfall_distance = min(all_landfall_distances.values())
            special_ports = {"Chandbali"}
            hidden_near_landfall_ports = {
                name for name in special_ports
                if (
                    all_landfall_distances[name] < 100.0
                    or np.isclose(
                        all_landfall_distances[name],
                        closest_landfall_distance,
                        atol=1e-6,
                    )
                )
            }

        axes_bbox = ax.bbox
        for city, plat, plon, risk in visible_cities:
            # Larger, outlined dots keep the risk colour visible over both
            # the pale land and blue ocean parts of Map.png.
            ax.scatter(
                plon,
                plat,
                marker='o',
                s=42,
                color=risk["color"],
                edgecolor='black',
                linewidth=0.9,
                zorder=6,
            )

            if city in hidden_near_landfall_ports:
                continue

            base_disp = ax.transData.transform((plon, plat))
            width_pt = _text_width_pt(city, label_fontsize, weight='bold')
            chosen = None

            for dx_pt, dy_pt, ha, va in label_candidates:
                bbox = label_bbox(base_disp, dx_pt, dy_pt, ha, va, width_pt)
                inside_axes = (
                    bbox[0] >= axes_bbox.x0
                    and bbox[1] >= axes_bbox.y0
                    and bbox[2] <= axes_bbox.x1
                    and bbox[3] <= axes_bbox.y1
                )
                if inside_axes and not any(
                    boxes_overlap(bbox, previous)
                    for previous in city_label_bboxes
                ):
                    chosen = (dx_pt, dy_pt, ha, va, bbox)
                    break

            if chosen is None:
                continue

            dx_pt, dy_pt, ha, va, bbox = chosen
            city_label_bboxes.append(bbox)
            ax.annotate(
                city,
                xy=(plon, plat),
                xycoords='data',
                xytext=(dx_pt, dy_pt),
                textcoords='offset points',
                fontsize=label_fontsize,
                fontweight='bold',
                color='#111827',
                ha=ha,
                va=va,
                bbox=dict(
                    facecolor='white',
                    alpha=0.94,
                    edgecolor=risk["color"],
                    linewidth=1.8,
                    boxstyle='round,pad=0.26',
                ),
                # Track, wind radii and forecast labels stay in front of the
                # port label, so the forecast remains readable underneath.
                zorder=2,
            )

    # -------------------- MOVEMENT INFO --------------------
    pressure = track_data_obs["Pressure"].iloc[-1]

    if len(track_data_obs) >= 2:
        coord1 = (
            track_data_obs["Latitude"].iloc[-2],
            track_data_obs["Longitude"].iloc[-2]
        )
        coord2 = (
            track_data_obs["Latitude"].iloc[-1],
            track_data_obs["Longitude"].iloc[-1]
        )
        obs_time_diff = (
            track_data_obs['tnd'].iloc[-1] -
            track_data_obs['tnd'].iloc[-2]
        ).total_seconds() / 3600
        obs_distance = haversine(coord1, coord2)
        obs_speed = obs_distance / obs_time_diff
        obs_bearing = get_bearing(coord1, coord2)
        obs_move_dir = get_cardinal_direction(obs_bearing)
    else:
        obs_speed = 0
        obs_move_dir = 'Stationary'

    last_obs_coord = (
        track_data_obs["Latitude"].iloc[-1],
        track_data_obs["Longitude"].iloc[-1]
    )
    last_obs_time = track_data_obs['tnd'].iloc[-1]

    if len(track_data_for) >= 1:
        target_24hr = last_obs_time + pd.Timedelta(hours=24)
        has_24hr_data = False
        closest_24hr_idx = None

        for idx, forecast_time in enumerate(track_data_for['tnd']):
            time_diff = abs((forecast_time - target_24hr).total_seconds() / 3600)
            if time_diff <= 3:
                has_24hr_data = True
                closest_24hr_idx = idx
                break

        if has_24hr_data and closest_24hr_idx is not None:
            forecast_coord = (
                track_data_for["Latitude"].iloc[closest_24hr_idx],
                track_data_for["Longitude"].iloc[closest_24hr_idx]
            )
            forecast_time = track_data_for['tnd'].iloc[closest_24hr_idx]
            time_label = "24hr"
        else:
            last_idx = len(track_data_for) - 1
            forecast_coord = (
                track_data_for["Latitude"].iloc[last_idx],
                track_data_for["Longitude"].iloc[last_idx]
            )
            forecast_time = track_data_for['tnd'].iloc[last_idx]

            time_diff_hours = (forecast_time - last_obs_time).total_seconds() / 3600
            if time_diff_hours >= 24:
                time_label = "24hr"
            elif time_diff_hours >= 12:
                time_label = f"{int(time_diff_hours)}h"
            else:
                time_label = f"{int(time_diff_hours * 10) / 10}h"

        time_diff = forecast_time - last_obs_time
        for_time_diff_hours = time_diff.total_seconds() / 3600
        for_distance = haversine(last_obs_coord, forecast_coord)
        for_speed = for_distance / for_time_diff_hours
        for_bearing = get_bearing(last_obs_coord, forecast_coord)
        for_move_dir = get_cardinal_direction(for_bearing)
    else:
        for_speed = 0
        for_move_dir = 'Stationary'
        time_label = 'N/A'

    if SHOW_MOVEMENT_TABLE:
        header2 = ["Movement (km/h)"]
        table_data2 = [
            [f"NOW: {obs_move_dir}, {int(obs_speed)}km"],
            [f"NEXT({time_label}): {for_move_dir}, {int(for_speed)}km"]
        ]

        table2 = ax.table(
            cellText=table_data2,
            loc='right',
            colLabels=header2,
            cellLoc='center',
            colColours=['#f0f0f0'],
            zorder=7,
            bbox=[0.805, 0.045, 0.19, 0.12]
        )
        table2.auto_set_font_size(False)
        table2.set_fontsize(9)
        table2.auto_set_column_width([0])
        table2[0, 0].set_text_props(fontweight='bold')
        table2.scale(1, 1.8)

    # -------------------- ACE BOX --------------------
    if SHOW_ACE_BOX:
        ace_value = calculate_ace(track_data_obs)
        ace = ax.text(
            0.01, 0.99,
            f"ACE: {ace_value:.3f}",
            fontsize=14, ha="left", va="top",
            color='white', transform=ax.transAxes, zorder=7
        )
        ace.set_bbox(dict(facecolor='black', alpha=0.4, edgecolor='none'))

    # -- ML LANDFALL PREDICTION (VERY SIMPLE k-NN) --
    ml_landfall_lat = None
    ml_landfall_lon = None

    try:
        ml_landfall_lat, ml_landfall_lon = predict_landfall_latlon(
            track_data_obs,
            training_csv_path=str(ASSETS_DIR / "climo.csv"),
            k=10
        )
    except Exception as e:
        print(f"[WARN] Landfall ML prediction failed: {e}")

    
 # --- DRAW ML LANDFALL POINT ON MAP ---
    if SHOW_AI_POSITION and ml_landfall_lat is not None and ml_landfall_lon is not None:
        # only plot if inside map bounds
        if lat_min <= ml_landfall_lat <= lat_max and lon_min <= ml_landfall_lon <= lon_max:
            ax.scatter(
                ml_landfall_lon, ml_landfall_lat,
                marker='*',
                s=150,
                edgecolor='k',
                facecolor='yellow',
                zorder=8
            )
            ax.annotate(
                "AI POINT",
                xy=(ml_landfall_lon, ml_landfall_lat),
                xycoords='data',
                xytext=(0, -12),
                textcoords='offset points',
                fontsize=8,
                fontweight='bold',
                ha='center',
                va='top',
                bbox=dict(
                    facecolor='white',
                    alpha=0.7,
                    boxstyle='round,pad=0.2'
                ),
                zorder=9
            )            
            
    # ----------- TITLES & MAX WIND BOXES ------
    title = (
        f"OBSERVED: {observed_start_time} To {observed_end_time}\n"
        f"FORECAST: {forecast_start_time} To {forecast_end_time}"
    )
    ax.set_title(title, fontsize=12, fontweight='bold')

    title_text = "Invest" if is_invest else "Cyclone"
    plt.suptitle(
        f'Tropical {title_text} "{cyclone_name.upper()}" Track',
        fontsize=20, ha='center', color='red',
        fontweight='bold', y=0.960
    )

    max_observed_wind = track_data_obs['Intensity'].max()
    max_forecast_wind = track_data_for['Intensity'].max()

    if SHOW_MAX_WIND_BOXES:
        for x_pos, wind, title_box in [
            (0.19, max_observed_wind, "MAX OBSERVED"),
            (0.84, max_forecast_wind, "MAX FORECAST")
        ]:
            plt.gcf().text(
                x_pos, 0.925,
                f"{title_box}\n{wind_cat(wind)} ({int(wind)}KT)",
                fontsize=11, color='black', va='top', ha='center',
                bbox=dict(
                    facecolor='skyblue', alpha=0.3,
                    edgecolor='darkred', boxstyle='round,pad=0.5'
                ),
                transform=plt.gcf().transFigure, zorder=10
            )
                       
    # -------------------- FOOTER --------------------
    if SHOW_FOOTER:
        for x, ha, t in [
            (0.01, "left",
             f"WIND: {ci}KT | PRESSURE: {pressure}MB | UPDATED: {ci_tnd:%HZ @ %d %b %Y}"),
            (0.99, "right", FOOTER_TEXT),
        ]:
            ax.text(
                x, 0.01, t,
                ha=ha, va="bottom",
                fontsize=14, color="white",
                bbox=dict(fc="black", alpha=.4, ec="none"),
                transform=ax.transAxes, zorder=7
            )

    # ------------- FINAL STYLING & SAVE ----------
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel("(1-MINUTE SUSTAINED WIND SCALE)")

    plt.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close(fig)

    # Hand the computed insights back to the caller (CLI -> feature plugins)
    return {"landfall": landfall_info, "approaches": approach_rows}
    
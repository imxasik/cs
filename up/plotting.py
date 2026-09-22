import os
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
)
from .cone import create_nhc_cone
from .geo import haversine, get_bearing, get_cardinal_direction
from .ace import calculate_ace
from assets.TcCites import BOB
from features.ailoc import predict_landfall_latlon
from features.aibc import apply_simple_bias  # <<< NEW


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

    WIND_RADIUS_SWATCH = {
        "24 KT Wind": ('blue', 25),
        "34 KT Wind": ('red', 20),
        "64 KT Wind": ('magenta', 15),
    }

    prev_conditions = [
        ("Invest Area (Low)", 'lime', 'full'),
        ("Tropical Depression", 'steelblue', 'full'),
        ("Cyclonic Storm", 'aqua', 'full'),
        ("Category 1", 'lemonchiffon', 'full'),
        ("Category 2", 'gold', 'full'),
        ("Category 3", 'tomato', 'full'),
        ("Category 4", 'fuchsia', 'full'),
        ("Category 5", 'mediumpurple', 'full'),
        (" ", '', 'none'),
        ("24 KT Wind", 'white', 'none'),
        (" ", ' ', 'none'),
        ("34 KT Wind", 'white', 'none'),
        (" ", ' ', 'none'),
        ("64 KT Wind", 'white', 'none'),
        (" ", ' ', 'none')
    ]

    legend_elements_prev = [
        Line2D(
            [0], [0],
            marker='o',
            color=('w' if condition == " "
                   else WIND_RADIUS_SWATCH.get(condition, ('black', 8))[0]),
            markerfacecolor=color,
            markersize=WIND_RADIUS_SWATCH.get(condition, (None, 8))[1],
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

  # -------------------- LEGEND --------------------
    if SHOW_LEGEND:
    
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

        legend = ax.legend(
            handles=legend_elements_prev,
            loc='upper right',
            title='MAP LEGEND'
        )
        legend.get_title().set_fontweight('bold')

  # Make sure legend is above all plotted data
        legend.set_zorder(9999)
        legend.get_frame().set_zorder(9999)
        

    # -------------------- TIME STRINGS --------------------
    observed_start_time = track_data_obs['tnd'].iloc[0].strftime("%HZ, %d %b %Y")
    observed_end_time = track_data_obs['tnd'].iloc[-1].strftime("%HZ, %d %b %Y")
    forecast_start_time = track_data_for['tnd'].iloc[0].strftime("%HZ, %d %b %Y")
    forecast_end_time = track_data_for['tnd'].iloc[-1].strftime("%HZ, %d %b %Y")

    # -------------------- PORTS & DISTANCES --------------------
    City = BOB()
    locations = City
    obs = (prev_lat, prev_lon)

    header = ["Port Distance(km)"]
    table_data = []
    visible_cities = []

    if SHOW_PORTS or SHOW_PORT_TABLE:
        for location, coord in locations.items():
            plat, plon = coord
            if lat_min <= plat <= lat_max and lon_min <= plon <= lon_max:
                dis = haversine(coord, obs)
                bear = get_bearing(coord, obs)
                dirc = get_cardinal_direction(bear)
                table_data.append([f"{location}: {int(dis)}, {dirc}"])
                visible_cities.append((location, plat, plon))

    if SHOW_PORT_TABLE and table_data:
        table = ax.table(
            cellText=table_data,
            loc='left',
            colLabels=header,
            cellLoc='center',
            colColours=['#f0f0f0'],
            zorder=7,
            bbox=[0.005, 0.045, 0.22, 0.12]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width([0])
        table[0, 0].set_text_props(fontweight='bold')
        table.scale(1, 1.5)
    elif SHOW_PORT_TABLE:
        ax.text(
            0.01, 0.05,
            "No ports in view",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom", ha="left",
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
            zorder=7
        )

    if SHOW_PORTS:
        # City labels: centered above marker + overlap control
        city_label_positions = []
        fig = ax.figure
        dpi = fig.dpi

        dy_pt_base = 10
        city_min_dist_px = 18

        for city, plat, plon in visible_cities:
            ax.scatter(plon, plat, marker='o', s=14, color='red', zorder=2)

            base_disp = ax.transData.transform((plon, plat))

            dy_pt = dy_pt_base
            dy_px = dy_pt * dpi / 72.0
            label_disp = base_disp + np.array([0, dy_px])

            if any(
                np.hypot(label_disp[0] - x, label_disp[1] - y) < city_min_dist_px
                for (x, y) in city_label_positions
            ):
                dy_pt = dy_pt_base + 6
                dy_px = dy_pt * dpi / 72.0
                label_disp = base_disp + np.array([0, dy_px])

            city_label_positions.append((label_disp[0], label_disp[1]))

            ax.annotate(
                city,
                xy=(plon, plat),
                xycoords='data',
                xytext=(0, dy_pt),
                textcoords='offset points',
                fontsize=10,
                color='red',
                ha='center',
                va='bottom',
                bbox=dict(
                    facecolor='white',
                    alpha=0.6,
                    boxstyle='round,pad=0.20',
                    edgecolor='none'
                ),
                zorder=3
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
            training_csv_path="data/climo.csv",  # adjust path if needed
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
            (0.99, "right", "© XP WEATHER"),
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
    
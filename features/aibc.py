# features/track_bias.py

import numpy as np
import pandas as pd

from pathlib import Path

# Repo-root assets folder (works no matter what the current directory is)
_ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"

# Simple in-memory cache so we only read CSV once
_BIAS_TABLE_CACHE = None
_BIAS_PATH_CACHE = None


def _get_bias_table(training_csv_path=None):
    """
    Load segmented bias table from climo_bias.csv.

    Expected columns:
        region   : 'west' / 'east' / 'all'
        strength : 'weak' / 'strong' / 'all'
        motion   : 'west' / 'south' / 'east' / 'north' / 'all'
        fhr      : forecast hour
        dlat     : mean latitude bias to ADD (deg)
        dlon     : mean longitude bias to ADD (deg)

    If the file or columns are missing/invalid, returns None
    and we fall back to a static heuristic bias.
    """
    global _BIAS_TABLE_CACHE, _BIAS_PATH_CACHE

    if _BIAS_TABLE_CACHE is not None and _BIAS_PATH_CACHE == training_csv_path:
        return _BIAS_TABLE_CACHE

    try:
        df = pd.read_csv(training_csv_path)

        required = {"region", "strength", "motion", "fhr", "dlat", "dlon"}
        if not required.issubset(df.columns):
            print("[WARN] climo_bias.csv missing required columns; "
                  "using fallback static bias.")
            _BIAS_TABLE_CACHE = None
            _BIAS_PATH_CACHE = training_csv_path
            return None

        df = df[["region", "strength", "motion", "fhr", "dlat", "dlon"]].dropna()
        df = df.sort_values(["region", "strength", "motion", "fhr"])

        if df.empty:
            print("[WARN] climo bias table is empty; using fallback static bias.")
            _BIAS_TABLE_CACHE = None
        else:
            _BIAS_TABLE_CACHE = df

        _BIAS_PATH_CACHE = training_csv_path
        return _BIAS_TABLE_CACHE

    except Exception as e:
        print(f"[WARN] Could not load climo bias table ({e}); "
              "using fallback static bias.")
        _BIAS_TABLE_CACHE = None
        _BIAS_PATH_CACHE = training_csv_path
        return None


def _classify_motion(dir_deg):
    """
    Same 4-class motion as training:
      west  : 225–315
      south : 135–225
      east  :  45–135
      north : everything else
    """
    try:
        if dir_deg is None or pd.isna(dir_deg):
            return "north"
        d = float(dir_deg) % 360.0
    except Exception:
        return "north"

    if 225.0 <= d < 315.0:
        return "west"
    if 135.0 <= d < 225.0:
        return "south"
    if 45.0 <= d < 135.0:
        return "east"
    return "north"


def _choose_bias_subset(bias_df, last_lon, last_wind, last_dir):
    """
    Choose the best bias subset based on last longitude, wind, and direction.

    - Region:
        west  if last_lon < 87E
        east  otherwise
    - Strength:
        strong if last_wind >= 64 kt, weak otherwise
    - Motion:
        west / south / east / north via _classify_motion(last_dir)

    Fallback order:
        1) exact (region, strength, motion)
        2) (region, strength, any motion)
        3) (region, any strength, motion)
        4) (any region, strength, motion)
        5) (region only)
        6) (strength only)
        7) (motion only)
        8) ('all', 'all', 'all') if present
        9) full table
    """
    if bias_df is None or bias_df.empty:
        return bias_df

    region = None
    strength = None
    motion = None

    if last_lon is not None:
        try:
            region = "west" if float(last_lon) < 87.0 else "east"
        except Exception:
            region = None

    if last_wind is not None:
        try:
            strength = "strong" if float(last_wind) >= 64.0 else "weak"
        except Exception:
            strength = None

    if last_dir is not None:
        motion = _classify_motion(last_dir)

    subset = bias_df

    # 1) exact (region, strength, motion)
    if region and strength and motion:
        cand = bias_df[
            (bias_df["region"] == region) &
            (bias_df["strength"] == strength) &
            (bias_df["motion"] == motion)
        ]
        if not cand.empty:
            return cand

    # 2) (region, strength, any motion)
    if region and strength:
        cand = bias_df[
            (bias_df["region"] == region) &
            (bias_df["strength"] == strength)
        ]
        if not cand.empty:
            subset = cand

    # 3) (region, any strength, motion)
    if region and motion:
        cand = bias_df[
            (bias_df["region"] == region) &
            (bias_df["motion"] == motion)
        ]
        if not cand.empty:
            subset = cand

    # 4) (any region, strength, motion)
    if strength and motion:
        cand = bias_df[
            (bias_df["strength"] == strength) &
            (bias_df["motion"] == motion)
        ]
        if not cand.empty:
            subset = cand

    # 5) region only
    if region:
        cand = bias_df[bias_df["region"] == region]
        if not cand.empty:
            subset = cand

    # 6) strength only
    if strength:
        cand = bias_df[bias_df["strength"] == strength]
        if not cand.empty:
            subset = cand

    # 7) motion only
    if motion:
        cand = bias_df[bias_df["motion"] == motion]
        if not cand.empty:
            subset = cand

    # 8) global 'all/all/all'
    cand = bias_df[
        (bias_df["region"] == "all") &
        (bias_df["strength"] == "all") &
        (bias_df["motion"] == "all")
    ]
    if not cand.empty:
        return cand

    # 9) fallback: whatever we have (possibly full table)
    return subset


def apply_simple_bias(
    latitudes,
    longitudes,
    last_lon=None,
    last_wind=None,
    last_dir=None,
    training_csv_path=None
):
    """
    Apply a very light 'bias correction' for the forecast track.

    Priority:
      1) Try segmented climatological bias from climo_bias.csv.
         - Choose group by last_lon (east/west), last_wind (weak/strong),
           and last_dir (west/south/east/north motion).
         - Interpolate dlat/dlon vs forecast hour.
      2) If climo_bias.csv not usable, fall back to simple southwest shift
         that grows with lead time and saturates by 72 h.

    Parameters
    ----------
    latitudes, longitudes : array-like
        1D arrays of forecast track lat/lon (deg).
    last_lon : float or None
        Last observed longitude (deg E).
    last_wind : float or None
        Last observed intensity (kt).
    last_dir : float or None
        Last observed direction of motion (deg).

    Returns
    -------
    bc_lats, bc_lons : np.ndarray
        Bias-corrected latitude and longitude arrays.
    """
    lats = np.asarray(latitudes, dtype=float)
    lons = np.asarray(longitudes, dtype=float)

    n = lats.size
    if n == 0:
        return lats, lons

    # Forecast hours assuming ~6h spacing
    fhr = np.arange(n, dtype=float) * 6.0  # [0, 6, 12, 18, ...]

    bias_table = _get_bias_table(training_csv_path)

    if bias_table is not None:
        sub = _choose_bias_subset(bias_table, last_lon, last_wind, last_dir)
        sub = sub.sort_values("fhr")

        hrs = sub["fhr"].values.astype(float)
        dlat_tab = sub["dlat"].values.astype(float)
        dlon_tab = sub["dlon"].values.astype(float)

        if hrs.size > 0:
            dlat = np.interp(fhr, hrs, dlat_tab)
            dlon = np.interp(fhr, hrs, dlon_tab)

            bc_lats = lats + dlat
            bc_lons = lons + dlon
            return bc_lats, bc_lons

    # ---------- Fallback: old simple southwest shift ----------
    scale = np.clip(fhr / 72.0, 0.0, 1.0)

    dlat_max = -0.3   # max southward bias (deg)
    dlon_max = -0.5   # max westward bias  (deg)

    dlat = dlat_max * scale
    dlon = dlon_max * scale

    bc_lats = lats + dlat
    bc_lons = lons + dlon

    return bc_lats, bc_lons

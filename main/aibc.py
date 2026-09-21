# features/track_bias.py

import numpy as np
import pandas as pd

# Simple in-memory cache so we only read CSV once
_BIAS_TABLE_CACHE = None
_BIAS_PATH_CACHE = None


def _get_bias_table(training_csv_path: str = "data/climo_bias.csv"):
    """
    Load segmented bias table from climo_bias.csv.

    Expected columns:
        region   : 'west' / 'east' / 'all'
        strength : 'weak' / 'strong' / 'all'
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

        required = {"region", "strength", "fhr", "dlat", "dlon"}
        if not required.issubset(df.columns):
            print("[WARN] climo_bias.csv missing required columns; "
                  "using fallback static bias.")
            _BIAS_TABLE_CACHE = None
            _BIAS_PATH_CACHE = training_csv_path
            return None

        df = df[["region", "strength", "fhr", "dlat", "dlon"]].dropna()
        df = df.sort_values(["region", "strength", "fhr"])

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


def _choose_bias_subset(bias_df: pd.DataFrame,
                        last_lon: float | None,
                        last_wind: float | None) -> pd.DataFrame:
    """
    Choose the best bias subset based on last longitude and last wind.

    - Region:
        west  if landfall / last lon < ~87E
        east  otherwise
    - Strength:
        strong if >= 64 kt, weak otherwise

    Fallback order:
        1) exact (region, strength)
        2) (region, 'all') or ('all', strength) if present
        3) ('all', 'all') if present
        4) whole table
    """
    if bias_df is None or bias_df.empty:
        return bias_df

    region = None
    strength = None

    # Region guess from last longitude
    if last_lon is not None:
        region = "west" if last_lon < 87.0 else "east"

    # Strength guess from last wind
    if last_wind is not None:
        strength = "strong" if last_wind >= 64.0 else "weak"

    # Start with full table
    subset = bias_df

    # Try exact (region, strength)
    if region is not None and strength is not None:
        cand = bias_df[(bias_df["region"] == region) &
                       (bias_df["strength"] == strength)]
        if not cand.empty:
            return cand

    # Try region-specific with any strength
    if region is not None:
        cand = bias_df[bias_df["region"] == region]
        if not cand.empty:
            subset = cand

    # Try strength-specific with any region
    if strength is not None:
        cand = subset[subset["strength"] == strength]
        if not cand.empty:
            subset = cand

    # Try global "all/all"
    cand = bias_df[(bias_df["region"] == "all") &
                   (bias_df["strength"] == "all")]
    if not cand.empty:
        return cand

    # Fallback: whatever subset we have (possibly full table)
    return subset


def apply_simple_bias(
    latitudes,
    longitudes,
    last_lon: float | None = None,
    last_wind: float | None = None,
    training_csv_path: str = "data/climo_bias.csv"
):
    """
    Apply a very light 'bias correction' for the forecast track.

    Priority:
      1) Try to read segmented climatological bias from climo_bias.csv:
         - Choose segment based on last_lon (east/west) and last_wind (weak/strong).
         - Interpolate dlat/dlon vs forecast hour.
      2) If climo_bias.csv not usable, fall back to a simple southwest shift
         that grows with lead time and saturates by 72 h.

    Parameters
    ----------
    latitudes, longitudes : array-like
        1D arrays of forecast track lat/lon (deg).
    last_lon : float or None
        Last observed longitude (deg E), used to pick east/west bias.
    last_wind : float or None
        Last observed intensity (kt), used to pick weak/strong bias.

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

    # Assume points roughly every 6 h – just for scaling forecast hours.
    fhr = np.arange(n, dtype=float) * 6.0  # [0, 6, 12, 18, ...]

    # Try segmented climatological bias
    bias_table = _get_bias_table(training_csv_path)

    if bias_table is not None:
        sub = _choose_bias_subset(bias_table, last_lon, last_wind)

        # Ensure sorted by forecast hour
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
    
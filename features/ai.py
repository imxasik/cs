# features/track_bias.py

import numpy as np

def apply_simple_bias(latitudes, longitudes):
    """
    Ultra-light 'bias correction' for the forecast track.

    This is intentionally simple and cheap enough for phones.
    It applies a small, increasing southwestward shift with
    forecast lead time (you can tune the constants).

    Parameters
    ----------
    latitudes, longitudes : array-like
        1D arrays of forecast track lat/lon (deg).

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

    # Assume points roughly every 6 h – only used for scaling.
    fhr = np.arange(n, dtype=float) * 6.0  # forecast hours

    # Bias grows with lead time and saturates by 72 h
    scale = np.clip(fhr / 72.0, 0.0, 1.0)

    # Tunable constants (degrees). Negative = south/west.
    dlat_max = -0.3   # max southward bias (deg)
    dlon_max = -0.5   # max westward bias  (deg)

    dlat = dlat_max * scale
    dlon = dlon_max * scale

    bc_lats = lats + dlat
    bc_lons = lons + dlon

    return bc_lats, bc_lons
    
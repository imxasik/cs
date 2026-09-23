import numpy as np
from scipy.interpolate import make_interp_spline

def create_nhc_cone(lons, lats, initial_uncertainty, growth_rate):
    """Return cone polygon points and smoothed track (lon, lat).

    NHC-style envelope: the two sides run the FULL length of the smoothed
    track (radius grows linearly along it) and the polygon is closed by a
    semicircular U-cap around the final point, bulging along the direction
    of motion.  The side lines meet the cap tangentially, so the end of the
    cone is always the classic rounded "U" shape.
    """
    steps = len(lons)
    if steps < 2:
        raise ValueError("Need at least two track points for a cone")
    t = np.arange(steps)
    t_new = np.linspace(0, steps - 1, 120)

    k = min(3, steps - 1)
    spline_lon = make_interp_spline(t, lons, k=k)
    spline_lat = make_interp_spline(t, lats, k=k)
    smooth_lon = spline_lon(t_new)
    smooth_lat = spline_lat(t_new)

    uncertainties = initial_uncertainty + growth_rate * t_new

    dt = t_new[1] - t_new[0]
    dx = np.gradient(smooth_lon, dt)
    dy = np.gradient(smooth_lat, dt)
    norms = np.sqrt(dx**2 + dy**2)
    norms[norms == 0] = 1
    perp_x = -dy / norms
    perp_y = dx / norms

    # Both sides span the whole track so they reach the final point, where
    # the U-cap picks them up tangentially.
    upper_body = np.column_stack((
        smooth_lon + perp_x * uncertainties,
        smooth_lat + perp_y * uncertainties,
    ))
    lower_body = np.column_stack((
        smooth_lon - perp_x * uncertainties,
        smooth_lat - perp_y * uncertainties,
    ))

    end_lon = smooth_lon[-1]
    end_lat = smooth_lat[-1]
    end_radius = uncertainties[-1]

    # Semicircle at the end point: from the upper side, through the forward
    # direction of motion, round to the lower side (the "U" end).
    forecast_bearing = np.arctan2(dy[-1], dx[-1])
    theta = np.linspace(forecast_bearing + np.pi / 2,
                        forecast_bearing - np.pi / 2, 41)[1:-1]
    arc_lon = end_lon + end_radius * np.cos(theta)
    arc_lat = end_lat + end_radius * np.sin(theta)

    nhc_cone_points = np.vstack((
        upper_body,
        np.column_stack((arc_lon, arc_lat)),
        lower_body[::-1],
    ))

    return nhc_cone_points, smooth_lon, smooth_lat

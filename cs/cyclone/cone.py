import numpy as np
from scipy.interpolate import make_interp_spline

def create_nhc_cone(lons, lats, initial_uncertainty, growth_rate):
    """Return cone polygon points and smoothed track (lon, lat)."""
    steps = len(lons)
    t = np.arange(steps)
    t_new = np.linspace(0, steps - 1, 100)

    spline_lon = make_interp_spline(t, lons, k=3)
    spline_lat = make_interp_spline(t, lats, k=3)
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

    body_end_idx = int(0.85 * len(t_new))

    upper_body = np.vstack((
        smooth_lon[:body_end_idx] + perp_x[:body_end_idx] * uncertainties[:body_end_idx],
        smooth_lat[:body_end_idx] + perp_y[:body_end_idx] * uncertainties[:body_end_idx],
    )).T

    lower_body = np.vstack((
        smooth_lon[:body_end_idx] - perp_x[:body_end_idx] * uncertainties[:body_end_idx],
        smooth_lat[:body_end_idx] - perp_y[:body_end_idx] * uncertainties[:body_end_idx],
    )).T

    end_lon = smooth_lon[-1]
    end_lat = smooth_lat[-1]
    end_radius = uncertainties[-1]

    end_idx = len(t_new) - 1
    forecast_dx = dx[end_idx]
    forecast_dy = dy[end_idx]
    forecast_bearing = np.arctan2(forecast_dy, forecast_dx)

    upper_dir = forecast_bearing + np.pi / 2
    lower_dir = forecast_bearing - np.pi / 2

    theta = np.linspace(upper_dir, lower_dir, 20)
    arc_lon = end_lon + end_radius * np.cos(theta)
    arc_lat = end_lat + end_radius * np.sin(theta)

    nhc_cone_points = np.vstack((
        upper_body,
        np.column_stack((arc_lon, arc_lat)),
        lower_body[::-1],
    ))

    return nhc_cone_points, smooth_lon, smooth_lat

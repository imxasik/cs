"""
Landfall estimation & closest-approach-per-port calculations.

Pure helpers — no plotting here. Used by cyclone/plotting.py (map output),
cyclone/cli.py (console + feature context).

Where the track comes ashore is measured on the SAME vector coastline that
the map draws (``assets/geo/land.geojson``), so the landfall marker always
sits exactly on the drawn coast/land border shape.  The coarse hand-drawn
``COASTLINES`` polylines below are only a fallback for when that asset is
missing.
"""

from math import cos, sin, radians, degrees, atan2, hypot, sqrt, isfinite
from functools import lru_cache
from pathlib import Path
import json

import pandas as pd

from .geo import haversine, get_bearing, get_cardinal_direction

# --------------------------------------------------------------------------
# Approximate coastline polylines: each is a list of (lon, lat) vertices.
# --------------------------------------------------------------------------
COASTLINES = [
    # India east coast: Chennai -> Sundarbans
    [
        (80.28, 13.10), (80.15, 13.55), (80.05, 14.20), (80.10, 14.90),
        (80.25, 15.50), (80.70, 15.90), (81.15, 16.18), (81.60, 16.35),
        (82.00, 16.60), (82.25, 16.98), (82.60, 17.20), (83.30, 17.70),
        (83.65, 18.30), (84.40, 18.85), (85.10, 19.35), (85.85, 19.95),
        (86.45, 20.35), (86.92, 21.10), (86.70, 21.60), (87.10, 21.85),
        (87.50, 21.70), (88.10, 21.65), (88.45, 21.60), (88.90, 21.90),
        (89.20, 22.15),
    ],
    # Bangladesh coast: Sundarbans -> Teknaf
    [
        (89.20, 22.15), (89.60, 21.85), (89.90, 22.05), (90.25, 21.90),
        (90.65, 21.85), (91.00, 22.05), (91.35, 21.95), (91.80, 22.25),
        (91.95, 21.90), (91.98, 21.45), (92.30, 20.90),
    ],
    # Myanmar (Arakan) coast -> Irrawaddy delta -> Malay peninsula (west)
    [
        (92.30, 20.90), (92.65, 20.20), (92.95, 20.13), (93.40, 19.55),
        (94.10, 18.90), (94.55, 18.20), (94.85, 17.40), (95.30, 16.20),
        (95.90, 15.80), (96.40, 16.10), (96.80, 16.50), (97.30, 16.20),
        (97.70, 15.40), (98.10, 14.30), (98.35, 12.60), (98.45, 11.20),
        (98.55, 10.00),
    ],
    # Malay peninsula west coast -> Singapore
    [
        (98.55, 10.00), (99.10, 8.70), (99.90, 7.20), (100.40, 5.90),
        (100.60, 4.60), (101.00, 3.30), (101.90, 2.20), (103.00, 1.55),
        (103.80, 1.30),
    ],
    # Gulf of Thailand -> Cambodia -> south Vietnam (Mekong)
    [
        (100.40, 7.15), (100.90, 8.60), (100.60, 9.90), (101.00, 11.20),
        (101.70, 12.65), (102.50, 12.20), (103.00, 11.40), (103.50, 10.60),
        (104.50, 10.30), (105.50, 9.20), (106.50, 8.60), (106.80, 9.60),
        (106.80, 10.60), (107.00, 11.30),
    ],
    # Sri Lanka (loop)
    [
        (79.85, 9.80), (79.95, 8.90), (80.05, 8.10), (80.40, 7.20),
        (80.95, 6.35), (81.55, 6.05), (81.85, 6.55), (81.70, 7.30),
        (81.20, 8.20), (80.60, 9.30), (80.15, 9.80), (79.85, 9.80),
    ],
    # Andaman Islands (loop)
    [
        (92.75, 13.40), (92.90, 12.90), (92.95, 12.20), (93.15, 11.60),
        (93.55, 11.20), (93.90, 10.55), (93.65, 10.25), (93.55, 10.75),
        (93.20, 11.35), (92.95, 12.05), (92.80, 12.90), (92.70, 13.40),
        (92.75, 13.40),
    ],
    # Nicobar Islands (loop)
    [
        (93.60, 9.20), (93.95, 8.90), (93.70, 8.30), (93.90, 7.80),
        (93.55, 7.30), (92.95, 7.75), (92.75, 8.30), (93.20, 8.85),
        (93.60, 9.20),
    ],
    # Sumatra east coast
    [
        (95.35, 5.55), (95.95, 5.05), (96.60, 4.25), (97.40, 3.40),
        (98.10, 2.55), (98.80, 1.70), (99.60, 0.90), (100.40, 0.10),
        (101.10, -0.90), (101.95, -1.95), (102.80, -2.95), (103.70, -3.95),
        (104.60, -4.95), (105.60, -5.90), (105.90, -5.85),
    ],
    # India west coast (Arabian Sea side, southern half within map bounds)
    [
        (72.75, 20.90), (72.85, 20.00), (72.95, 19.10), (73.10, 18.30),
        (73.30, 17.60), (73.60, 16.60), (74.05, 15.40), (74.55, 14.20),
        (75.10, 12.90), (75.70, 11.85), (76.25, 10.30), (76.30, 9.20),
        (76.95, 8.55), (77.55, 8.10),
    ],
]


# --------------------------------------------------------------------------
# Coast polylines used for the landfall crossing
# --------------------------------------------------------------------------
_GEO_DIR = Path(__file__).resolve().parent.parent / "assets" / "geo"


@lru_cache(maxsize=4)
def load_coast_rings(geojson_path=None):
    """Coast polylines as lists of (lon, lat) — the map's own border shape.

    Reads the same ``assets/geo/land.geojson`` polygons that
    ``cyclone/basemap.py`` draws, so the estimated landfall lies exactly on
    the coastline shown in the graphic.  Falls back to the coarse
    ``COASTLINES`` sketch when the asset is missing.
    """
    path = Path(geojson_path) if geojson_path else (_GEO_DIR / "land.geojson")
    rings = []
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for feat in data.get("features", ()):
            geom = feat.get("geometry") or {}
            if geom.get("type") != "Polygon":
                continue
            coords = geom.get("coordinates") or []
            if not coords:
                continue
            ring = [(float(x), float(y)) for x, y in coords[0]]
            if len(ring) >= 4:
                rings.append(ring)
    except (OSError, ValueError):
        rings = []
    return tuple(rings) if rings else tuple(tuple(r) for r in COASTLINES)


def _rings_near_track(polylines, pts, margin=2.0):
    """Polylines whose bbox meets the track corridor (cheap pre-filter)."""
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    x0, x1 = min(xs) - margin, max(xs) + margin
    y0, y1 = min(ys) - margin, max(ys) + margin
    keep = []
    for ring in polylines:
        rx = [p[0] for p in ring]
        ry = [p[1] for p in ring]
        if max(rx) < x0 or min(rx) > x1 or max(ry) < y0 or min(ry) > y1:
            continue
        keep.append(ring)
    return keep


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def _seg_cross_fraction(p1, p2, p3, p4):
    """
    Fraction t along segment p1->p2 where it crosses segment p3->p4,
    or None if they don't intersect. Points are (lon, lat).
    """
    d1x, d1y = p2[0] - p1[0], p2[1] - p1[1]
    d2x, d2y = p4[0] - p3[0], p4[1] - p3[1]
    denom = d1x * d2y - d1y * d2x
    if denom == 0:
        return None
    dx, dy = p3[0] - p1[0], p3[1] - p1[1]
    t = (dx * d2y - dy * d2x) / denom
    u = (dx * d1y - dy * d1x) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return t
    return None


def _dist_point_seg_km(p, a, b):
    """
    Distance in km from point p to segment a-b, plus the fraction t along
    the segment at the closest point. Points are (lon, lat); a local
    equirectangular km projection is used (fine for short segments).
    """
    lat0 = radians((p[1] + a[1] + b[1]) / 3.0)
    kx = 111.320 * cos(lat0)
    ky = 110.574

    px, py = p[0] * kx, p[1] * ky
    ax, ay = a[0] * kx, a[1] * ky
    bx, by = b[0] * kx, b[1] * ky

    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    if seg2 == 0:
        return hypot(px - ax, py - ay), 0.0

    t = ((px - ax) * dx + (py - ay) * dy) / seg2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return hypot(px - cx, py - cy), t


def format_track_time(ts):
    """Timestamp -> compact 'DD/HHZ' label used across the plot."""
    return pd.Timestamp(ts).strftime("%d/%HZ")


# --------------------------------------------------------------------------
# Track assembly
# --------------------------------------------------------------------------
def _track_points(track_obs, track_for):
    """
    Merged, time-sorted list of (tnd, lon, lat) from observed + forecast.
    """
    pts = []
    if track_obs is not None and len(track_obs) > 0:
        for t, lat, lon in zip(track_obs["tnd"], track_obs["Latitude"],
                               track_obs["Longitude"]):
            pts.append((pd.Timestamp(t), float(lon), float(lat)))
    if track_for is not None and len(track_for) > 0:
        seen = {p[0] for p in pts}
        for t, lat, lon in zip(track_for["tnd"], track_for["Latitude"],
                               track_for["Longitude"]):
            t = pd.Timestamp(t)
            if t not in seen:
                pts.append((t, float(lon), float(lat)))
    pts.sort(key=lambda p: p[0])
    return pts


# --------------------------------------------------------------------------
# Landfall estimation
# --------------------------------------------------------------------------
def find_landfall(track_obs, track_for, ports=None, place_radius_km=250.0,
                  coast=None):
    """
    Estimate the first forecast landfall: where the last-obs -> forecast
    polyline first crosses the coast of the map's own land polygons
    (assets/geo/land.geojson — the exact border shape drawn on the map).

    `coast` may pass an explicit polyline sequence; by default the vector
    basemap rings are used (coarse COASTLINES fallback if missing).

    Returns a dict:
        time      : pandas.Timestamp of the crossing
        lon, lat  : crossing position (on the drawn coast)
        place     : nearest port name (or None if none within place_radius_km)
        place_km  : distance to that port (km)
        time_str  : 'DD/HHZ'
    or None when no crossing is found (track stays over water) or there
    is not enough forecast data (< 2 points from last obs onward).
    """
    pts = _track_points(track_obs, track_for)
    if len(pts) < 2:
        return None

    # Keep only the most recent position + everything after it
    last_obs_t = pts[-1][0]
    if track_for is None or len(track_for) == 0:
        return None  # nothing forecast yet -> nothing to estimate

    start_idx = 0
    if track_obs is not None and len(track_obs) > 0:
        last_obs_t = pd.Timestamp(track_obs["tnd"].iloc[-1])
        start_idx = max((i for i, p in enumerate(pts) if p[0] <= last_obs_t),
                        default=0)
    pts = pts[start_idx:]
    if len(pts) < 2:
        return None

    if coast is None:
        coast = load_coast_rings()
    coast = _rings_near_track(coast, pts)

    crossings = []
    for i in range(len(pts) - 1):
        t0, lon0, lat0 = pts[i]
        t1, lon1, lat1 = pts[i + 1]
        p1, p2 = (lon0, lat0), (lon1, lat1)

        for coast_ring in coast:
            for j in range(len(coast_ring) - 1):
                t_frac = _seg_cross_fraction(p1, p2, coast_ring[j],
                                             coast_ring[j + 1])
                if t_frac is None:
                    continue
                cx = p1[0] + t_frac * (p2[0] - p1[0])
                cy = p1[1] + t_frac * (p2[1] - p1[1])
                ct = t0 + (t1 - t0) * t_frac
                crossings.append((ct, cx, cy))

        if crossings:
            # Earliest crossing on this segment is enough (time-ordered scan)
            break

    if not crossings:
        return None

    ct, cx, cy = min(crossings, key=lambda c: c[0])

    place, place_km = None, None
    if ports:
        best = min(
            ((name, haversine((lat, lon), (cy, cx))) for name, (lat, lon) in ports.items()),
            key=lambda kv: kv[1],
        )
        if best[1] <= place_radius_km:
            place, place_km = best[0], round(best[1])

    return {
        "time": ct,
        "lon": round(cx, 3),
        "lat": round(cy, 3),
        "place": place,
        "place_km": place_km,
        "time_str": format_track_time(ct),
    }


# --------------------------------------------------------------------------
# Closest approach per port
# --------------------------------------------------------------------------
def current_centre(track_obs, track_for=None):
    """
    The *current* storm centre as (lat, lon): the last observed fix when
    there is one, otherwise the last point of the track. None if no data.
    """
    if track_obs is not None and len(track_obs) > 0:
        return (float(track_obs["Latitude"].iloc[-1]),
                float(track_obs["Longitude"].iloc[-1]))
    pts = _track_points(track_obs, track_for)
    if not pts:
        return None
    return (pts[-1][2], pts[-1][1])


# Port-risk thresholds are deliberately distance based so the marker colours
# answer one clear question: how close is this port to the estimated landfall?
# The bands match the map legend drawn by cyclone.plotting.
PORT_RISK_BANDS = (
    {
        "key": "high",
        "label": "HIGH RISK",
        "legend": "0–100 km",
        "color": "#d100ff",       # vivid magenta (0-100 km)
        "min_km": 0.0,
        "max_km": 100.0,
    },
    {
        "key": "medium",
        "label": "MEDIUM RISK",
        "legend": "100–300 km",
        "color": "#f59e0b",       # amber (100-300 km)
        "min_km": 100.0,
        "max_km": 300.0,
    },
    {
        "key": "low",
        "label": "LOW RISK",
        "legend": "300–500 km",
        "color": "#2563eb",       # blue (300-500 km)
        "min_km": 300.0,
        "max_km": 500.0,
    },
    {
        "key": "norisk",
        "label": "NO RISK",
        "legend": ">500 km",
        "color": "#16a34a",       # green (>500 km)
        "min_km": 500.0,
        "max_km": float("inf"),
    },
)


def classify_port_risk(distance_km):
    """Return the display band for a port's distance from landfall."""
    try:
        distance = float(distance_km)
    except (TypeError, ValueError):
        distance = float("nan")

    if not isfinite(distance):
        return {
            "key": "unknown",
            "label": "UNKNOWN",
            "legend": "No landfall",
            "color": "#64748b",
            "distance_km": None,
        }

    distance = max(0.0, distance)
    
    if distance <= 100.0:
        band = PORT_RISK_BANDS[0]  # High Risk (Magenta)
    elif distance <= 300.0:
        band = PORT_RISK_BANDS[1]  # Medium Risk (Red)
    elif distance <= 500.0:
        band = PORT_RISK_BANDS[2]  # Low Risk (Orange)
    else:
        band = PORT_RISK_BANDS[3]  # No Risk (Green)

    result = dict(band)
    result["distance_km"] = distance
    return result


def _per_port_approaches(track_obs, track_for, ports, radius_km=800.0):
    """
    Hidden workhorse: for every port, the minimum distance the track
    (observed + forecast) comes to it, and when/where that happens.

    Returns a list of dicts (one per port within radius_km):
        name, plat, plon,
        approach_km  : closest approach distance (km, rounded)
        time         : pandas.Timestamp of the closest approach
        time_str     : 'DD/HHZ'
        past         : closest approach already happened
        c_lat, c_lon : centre position at the closest approach
    """
    pts = _track_points(track_obs, track_for)
    if len(pts) < 2 or not ports:
        return []

    last_obs_t = None
    if track_obs is not None and len(track_obs) > 0:
        last_obs_t = pd.Timestamp(track_obs["tnd"].iloc[-1])

    results = []
    for name, (plat, plon) in ports.items():
        best = None  # (dist, time, lon, lat) of the closest track point
        for i in range(len(pts) - 1):
            t0, lon0, lat0 = pts[i]
            t1, lon1, lat1 = pts[i + 1]

            d, frac = _dist_point_seg_km((plon, plat), (lon0, lat0),
                                         (lon1, lat1))
            if best is None or d < best[0]:
                ct = t0 + (t1 - t0) * frac
                clon = lon0 + (lon1 - lon0) * frac
                clat = lat0 + (lat1 - lat0) * frac
                best = (d, ct, clon, clat)

        if best is None:
            continue
        d, ct, clon, clat = best
        if d > radius_km:
            continue
        results.append({
            "name": name,
            "plat": float(plat),
            "plon": float(plon),
            "approach_km": int(round(d)),
            "time": ct,
            "time_str": format_track_time(ct),
            "past": bool(last_obs_t is not None and ct <= last_obs_t),
            "c_lat": float(clat),
            "c_lon": float(clon),
        })
    return results


def closest_approaches(track_obs, track_for, ports,
                       radius_km=800.0, top=8):
    """
    Legacy helper: per-port minimum track distance only.

    Returns a list of dicts sorted by distance:
        name     : port name
        dist_km  : closest approach distance (km, rounded)
        time     : pandas.Timestamp of closest approach
        time_str : 'DD/HHZ'
        past     : True if the closest approach already happened
                   (at/before the last observation time)
        bearing  : degrees (0-360) from the port to the storm centre at
                   closest approach
        dir_str  : 16-point cardinal of `bearing` ('N', 'SSE', ...) — i.e.
                   which side of the port the centre passes on
    Only ports within radius_km are returned, at most `top` entries.
    """
    rows = _per_port_approaches(track_obs, track_for, ports, radius_km)
    results = []
    for r in rows:
        bearing = get_bearing((r["plat"], r["plon"]), (r["c_lat"], r["c_lon"]))
        results.append({
            "name": r["name"],
            "dist_km": r["approach_km"],
            "time": r["time"],
            "time_str": r["time_str"],
            "past": r["past"],
            "bearing": round(bearing, 1),
            "dir_str": get_cardinal_direction(bearing),
        })
    results.sort(key=lambda r: r["dist_km"])
    return results[:top]


def port_centre_table(track_obs, track_for, ports, landfall=None,
                      radius_km=800.0, top=4):
    """
    The single merged port table used on the map.

    Port selection: the `top` ports closest to the *landfall point* when a
    landfall was estimated (that is the place the storm is heading for),
    otherwise the `top` ports the track comes closest to.

    The numbers shown are for RIGHT NOW: how far the current storm centre
    is from each of those ports and in which direction it lies from that
    port. Rows are ordered by that distance, nearest first.

    Returns a list of dicts:
        name         : port name
        dist_km      : distance from the port to the *current* centre (km)
        bearing      : degrees (0-360) from the port to the current centre
        dir_str      : 16-point cardinal of `bearing` ('SSE', ...)
        landfall_km  : distance from the port to the landfall point (km,
                       None when there is no landfall estimate)
        approach_km  : closest approach distance of the track (km)
        time         : pandas.Timestamp of the closest approach
        time_str     : 'DD/HHZ'
        past         : True when the closest approach already happened
    """
    centre = current_centre(track_obs, track_for)
    if not centre or not ports:
        return []

    limit = max(1, int(top))

    if landfall is not None:
        # Heading for a landfall: take the ports sitting closest to the
        # point where the storm is expected to come ashore.
        lf_coord = (landfall["lat"], landfall["lon"])
        ranked = sorted(
            ((name, haversine(coord, lf_coord))
             for name, coord in ports.items()),
            key=lambda kv: kv[1],
        )
        within = [name for name, d in ranked if d <= radius_km]
        # Ports that far out are rare; top the table up so it is never
        # half empty.
        chosen_names = (within if len(within) >= limit else [n for n, _ in ranked])[:limit]
        chosen = {n: ports[n] for n in chosen_names}
    else:
        # No landfall: keep the old behaviour, the ports the track comes
        # closest to.
        cands = _per_port_approaches(track_obs, track_for, ports, radius_km)
        cands.sort(key=lambda r: r["approach_km"])
        chosen = {r["name"]: ports[r["name"]] for r in cands[:limit]}

    # Track-approach numbers for the selected ports only (cheap) so the
    # returned dicts keep the same shape either way.
    info = {r["name"]: r for r in
            _per_port_approaches(track_obs, track_for, chosen,
                                 radius_km=float("inf"))}

    rows = []
    for name, (plat, plon) in chosen.items():
        dist = haversine((plat, plon), centre)
        bearing = get_bearing((plat, plon), centre)
        appr = info.get(name)
        rows.append({
            "name": name,
            "dist_km": int(round(dist)),
            "bearing": round(bearing, 1),
            "dir_str": get_cardinal_direction(bearing),
            "landfall_km": (int(round(haversine((plat, plon), lf_coord)))
                            if landfall is not None else None),
            "approach_km": appr["approach_km"] if appr else None,
            "time": appr["time"] if appr else None,
            "time_str": appr["time_str"] if appr else None,
            "past": appr["past"] if appr else None,
        })

    rows.sort(key=lambda r: r["dist_km"])
    return rows

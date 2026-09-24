"""
Forgiving reader for combined observed/forecast cyclone track files.

The reference format (see data/README.md) is::

    Cyclone Name: DITWAH            <- or "Invest Name: 06B"
    Synoptic Time, Latitude, Longitude, Intensity, Pressure
    2025-11-25 00:00, 05.20, 78.90, 15, 1009
    === FORECAST ===
    Synoptic Time, Latitude, Longitude, Intensity, WindR24, WindR34, WindR64
    2025-12-03 06:00, 11.9, 79.2, 20, 00, 00, 00

Everything on top of that core is accepted too, so a hand-written or
exported file rarely needs editing before it can be plotted:

* the name header is optional — without one the file name is used, and a
  name like ``06B``/``92B`` is recognised as an invest automatically;
  ``Name:``, ``Storm:``, ``Tropical Cyclone:`` ... all work;
* a one-line column-TITLE row (``TIME (UTC) | LAT | LON | WIND (KT) | ...``)
  may sit first in every section — units in parentheses are ignored;
* column headers are optional and may appear in any order/case with
  synonyms (``date``/``time``, ``lat``, ``lon``/``lng``, ``wind``/``kt``,
  ``mslp``/``hpa``, ``r34`` ...); without a header the canonical column
  order above is assumed;
* any of ``|``  ``,``  ``;``  tab  or plain spaces may separate the columns
  (``|`` "vertical line" separators keep hand-edited rows tidy without
  wide padding);
* blank lines and ``#``/``;`` comment lines are skipped;
* missing radii/pressures may be written ``00``, ``0``, ``-``, ``--`` or
  left empty;
* the forecast divider may be ``=== FORECAST ===``, ``--- forecast ---``,
  ``FORECAST`` ... (case-insensitive).

The returned frames always carry the canonical columns the plotter uses:
``tnd, Latitude, Longitude, Intensity, Pressure, WindR24, WindR34, WindR64``.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

_OBS_COLS = ["tnd", "Latitude", "Longitude", "Intensity", "Pressure"]
_FOR_COLS = ["tnd", "Latitude", "Longitude", "Intensity", "WindR34", "WindR64"]

# header synonym -> canonical column
_SYNONYMS = {
    "tnd": ("t", "time", "synoptic time", "synoptictime", "datetime", "date",
            "valid time", "validtime", "vt", "obs time", "obstime"),
    "Latitude": ("lat", "latitude", "lat n", "latitude n"),
    "Longitude": ("lon", "long", "lng", "longitude", "lon e", "longitude e"),
    "Intensity": ("intensity", "wind", "wind kt", "kt", "kts", "max wind",
                  "maxwind", "vmax", "wind speed", "windspeed"),
    "Pressure": ("pressure", "pres", "mslp", "hpa", "p", "slp"),
    "WindR34": ("windr34", "r34", "rad34", "radius 34", "wind radius 34",
                "windradius34"),
    "WindR64": ("windr64", "r64", "rad64", "radius 64", "wind radius 64",
                "windradius64"),
}

# multi-word column names, glued together so whitespace-separated headers
# ("Synoptic Time Latitude ...") map to the same columns as CSV headers
_GLUE = (
    (r"synoptic\s+time", "synoptictime"),
    (r"valid\s+time", "validtime"),
    (r"obs\s+time", "obstime"),
    (r"max\s+wind", "maxwind"),
    (r"wind\s+speed", "windspeed"),
    (r"wind\s+radius\s*34", "windradius34"),
    (r"wind\s+radius\s*64", "windradius64"),
)


def _glue_header(line):
    out = line
    for pat, rep in _GLUE:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return out


_NAME_RE = re.compile(
    r"^\s*(?:tropical\s+)?(cyclone|invest|storm|system|depression|name)\s*"
    r"(?:name)?\s*[:=]\s*(?P<name>.+?)\s*$", re.IGNORECASE)
_DIVIDER_RE = re.compile(r"^[\s=\-*_#]*forecast[\s=\-*_#]*$", re.IGNORECASE)
_COMMENT_RE = re.compile(r"^\s*(#|;)")
_TIME_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}[ T]\d{1,2}:\d{2}")


def _missing(value):
    s = str(value).strip()
    return s in ("", "-", "--", "nan", "none", "xx", "00", "0") and s != "0"


def _clean_number(value):
    """'00'/'-'/'--'/'' -> NaN, otherwise float."""
    s = str(value).strip()
    if s in ("", "-", "--", "nan", "none", "xx"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _sniff_delimiter(line):
    for d in (",", ";", "\t", "|"):
        if d in line:
            return d
    return None   # whitespace separated


def _norm_header_token(field):
    """One header cell -> comparable token ('TIME (UTC)' -> 'time')."""
    s = str(field)
    s = re.sub(r"\([^)]*\)", " ", s)          # drop unit hints: (UTC) (KT)
    s = _glue_header(s.lower())               # "synoptic time" -> "synoptictime"
    s = re.sub(r"[^a-z0-9]+", " ", s)         # punctuation / '|' -> space
    return " ".join(s.split())


def _map_header(fields):
    """Map a header line to canonical column names; None if not a header."""
    lowered = [_norm_header_token(f) for f in fields]
    if not any(("time" in x or "date" in x or x in ("lat", "latitude"))
               for x in lowered):
        return None
    mapping = []
    for field in lowered:
        canon = None
        key = field.replace(" ", "")
        for target, syns in _SYNONYMS.items():
            for cand in (target.lower(),) + tuple(syns):
                if field == cand or (key and key == cand.replace(" ", "")):
                    canon = target
                    break
            if canon is not None:
                break
        mapping.append(canon)
    return mapping


def _parse_rows(lines, is_forecast):
    """Turn raw data lines into a DataFrame with canonical columns."""
    if not lines:
        return pd.DataFrame(columns=_FOR_COLS + ["Pressure"])

    mapping = None
    header_idx = 0
    for i, line in enumerate(lines[:3]):
        delim = _sniff_delimiter(line)
        fields = line.split(delim) if delim else line.split()
        cand = _map_header(fields)
        if cand is not None and any(c is not None for c in cand):
            mapping = cand
            header_idx = i + 1
            break

    records = []
    for line in lines[header_idx:]:
        delim = _sniff_delimiter(line)
        fields = [f.strip() for f in (line.split(delim) if delim
                                      else line.split())]
        if not fields:
            continue
        # whitespace-separated rows split "2026-09-19 00:00" in two -> rejoin
        if (len(fields) > 1
                and re.match(r"^\d{4}-\d{1,2}-\d{1,2}$", fields[0])
                and re.match(r"^\d{1,2}:\d{2}$", fields[1])):
            fields = [fields[0] + " " + fields[1]] + fields[2:]
        if mapping is not None:
            rec = {c: None for c in _FOR_COLS}
            for col, value in zip(mapping, fields):
                if col:
                    rec[col] = value
        else:
            canon = _FOR_COLS if is_forecast else _OBS_COLS
            rec = {c: (fields[i] if i < len(fields) else None)
                   for i, c in enumerate(canon)}
        records.append(rec)

    if not records:
        df = pd.DataFrame(columns=_FOR_COLS + ["Pressure"])
        return df

    df = pd.DataFrame(records)
    for col in _FOR_COLS + ["Pressure"]:
        if col not in df.columns:
            df[col] = None

    df["tnd"] = pd.to_datetime(df["tnd"], format="%Y-%m-%d %H:%M",
                               errors="coerce")
    df = df.dropna(subset=["tnd", "Latitude", "Longitude"])
    df["Latitude"] = df["Latitude"].map(_clean_number)
    df["Longitude"] = df["Longitude"].map(_clean_number)
    df["Intensity"] = df["Intensity"].map(_clean_number).fillna(0.0)
    df["Pressure"] = df["Pressure"].map(_clean_number)
    for col in ("WindR34", "WindR64"):
        df[col] = df[col].map(_clean_number).fillna(0.0)
    return df.reset_index(drop=True)


_V2_KEYWORDS = {"NAME", "KIND", "OBS", "FORECAST"}


def _try_v2(lines):
    """
    Parse the clean v2 layout; return None when the file is not v2.

        NAME 06B                 <- optional (file name otherwise)
        KIND INVEST              <- optional (invest | cyclone)
        OBS
        2026-09-19 00:00  13.00  95.00  10  1007
        FORECAST
        2026-09-22 12:00  17.50  85.50  30  1.3  -  -

    Fixed column order, whitespace separated, '-' for "none", # comments.
    """
    if not any(l.split()[0].upper() in _V2_KEYWORDS
               for l in lines if l.split()):
        return None
    name = kind = None
    section = "obs"
    obs_lines, for_lines = [], []
    for line in lines:
        parts = line.split()
        key = parts[0].upper().rstrip(":")
        if key == "NAME" and len(parts) > 1:
            name = line.split(None, 1)[1].strip()
        elif key == "KIND" and len(parts) > 1:
            kind = parts[1].lower()
        elif key == "OBS" and len(parts) == 1:
            section = "obs"
        elif key == "FORECAST" and len(parts) == 1:
            section = "for"
        else:
            (for_lines if section == "for" else obs_lines).append(line)
    return name, kind, obs_lines, for_lines


def process_combined_cyclone_data(file_path):
    """
    Read a combined observed/forecast file (v2 or legacy layout).

    Returns (cyclone_name, track_obs, track_for, is_invest) with the
    canonical columns documented at the top of this module.
    """
    path = Path(file_path)
    raw = [ln.strip() for ln in
           path.read_text(encoding="utf-8", errors="replace").splitlines()]
    raw = [ln for ln in raw if ln and not _COMMENT_RE.match(ln)]

    v2 = _try_v2(raw)
    if v2 is not None:
        name, kind, obs_lines, for_lines = v2
        if name is None:
            name = path.stem
        if kind is not None:
            is_invest = kind.startswith("invest")
        else:
            is_invest = bool(re.match(r"^\d{2}B$", name, re.IGNORECASE))
        track_obs = _parse_rows(obs_lines, is_forecast=False)
        track_for = _parse_rows(for_lines, is_forecast=True)
        return _finalise(name, track_obs, track_for, is_invest)

    name, is_invest = None, False
    obs_lines, for_lines = [], []
    reading_forecast = False
    first_meaningful = True

    for stripped in raw:
        if _DIVIDER_RE.match(stripped):
            reading_forecast = True
            continue
        if first_meaningful:
            first_meaningful = False
            m = _NAME_RE.match(stripped)
            if m and not _TIME_RE.match(stripped):
                name = m.group("name").strip().strip('"').strip("'")
                is_invest = "invest" in stripped.lower()
                continue
            # no name header: derive it from the file name
            name = path.stem
            is_invest = bool(re.match(r"^\d{2}B$", name, re.IGNORECASE))
        (for_lines if reading_forecast else obs_lines).append(stripped)

    if name is None:
        name = path.stem
        is_invest = bool(re.match(r"^\d{2}B$", name, re.IGNORECASE))

    track_obs = _parse_rows(obs_lines, is_forecast=False)
    track_for = _parse_rows(for_lines, is_forecast=True)
    return _finalise(name, track_obs, track_for, is_invest)


def _finalise(name, track_obs, track_for, is_invest):
    if track_obs.empty and not track_for.empty:
        # forecast-only file: promote the first forecast fix to "current"
        track_obs = track_for.head(1).copy()
        track_for = track_for.iloc[1:].reset_index(drop=True)
    return name, track_obs, track_for, is_invest

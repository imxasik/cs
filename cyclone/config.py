import configparser
from pathlib import Path

# Default numeric settings
FLOAT_DEFAULTS = {
    "buffer": 2.0,
    "ucr": 0.20,
    "min_lat": -4.0,
    "max_lat": 31.0,
    "min_lon": 73.2,
    "max_lon": 107.0,
    "minlat_offset": -0.0,
    "maxlat_offset": -0.5,
    "output_dpi": 300.0,
    "approach_radius": 800.0,  # ports farther than this (km) are skipped in the port table
    "approach_ports": 4.0,  # how many of the closest ports the table lists
    # Bottom-centre forecast table (Time / Speed rows)
    "forecast_tz_offset": 6.0,        # hours added to the UTC synoptic times (6 = BST)
    "forecast_table_max_cols": 8.0,   # thin the steps out if there are more than this
    "forecast_table_min_fontsize": 6.8,
    # Wind-radius driven map window: margin beyond the outermost ring
    "wind_radius_pad": 0.2,
}

# Default boolean toggles
BOOL_DEFAULTS = {
    "show_cone": True,
    "show_legend": True,
    "show_ports": True,
    "show_movement_table": True,
    "show_ace_box": True,
    "show_max_wind_boxes": True,
    "show_footer": True,
    "organize_by_year": False,  # save into output/plots/<year>/
    "show_landfall": True,      # estimate + mark where the track hits the coast
    "show_approach_table": True,  # port table: current distance & direction per port
    "show_forecast_table": True,  # bottom-centre Time / Speed forecast table
    "show_forecast_key": True,    # map key strip above the forecast table
    "full_track_extent": False,   # zoom out so the whole observed track shows
    "wind_radius_extent": True,   # size the map window from the wind-radius rings
    "show_grid": True,            # map graticule (lat/lon grid lines)
    "show_scale_bar": True,       # kilometre scale bar inside the map
    "show_landfall": True,      # estimate + mark where the track hits the coast
    "show_landfall_label": False, # Landfall text box on map (Default: False / 0)
    "show_states": True,  # State / Province border দেখানোর জন্য
    "show_state_labels": True,  # State / Province নাম দেখানোর জন্য (নতুন)
    
}

# Default strings under [style]
STYLE_DEFAULTS = {
    "theme": "official",                # design tokens: cyclone/theme.py
    "brand_name": "XP WEATHER",         # header brand chip (left corner)
    "date_format": "%HZ, %d %b %Y",      # title date format (strftime)
    "footer_text": "\u00a9 XP WEATHER",  # right-hand footer on the map
    "forecast_time_label": "TIME (BST)",   # first cell of the forecast table
    "forecast_speed_label": "WIND (KM)",  # first cell of the speed row
    "forecast_speed_unit": "KM/H",         # appended to every speed cell
    # What the "Speed" row shows:
    #   wind   - the forecast wind intensity in km/h (knots x 1.852), so the
    #            row matches the knots on the track (25KT -> 46KM/H)
    #   motion - the storm's translation speed in km/h
    "forecast_speed_mode": "wind",
    # Output PNG name (without .png).  Empty = "<Name>_Track".
    # "{name}" is replaced by the cyclone/invest name, so
    #   output_name = {name}_Track_v2
    # saves 06B as 06B_Track_v2.png.
    "output_name": "",
}


def _load_config():
    root_dir = Path(__file__).resolve().parent.parent
    cfg_path = root_dir / "config.ini"
    # interpolation=None so values may contain '%' (e.g. date_format = %HZ, %d %b %Y)
    parser = configparser.ConfigParser(interpolation=None)

    float_values = FLOAT_DEFAULTS.copy()
    bool_values = BOOL_DEFAULTS.copy()
    style_values = STYLE_DEFAULTS.copy()

    if cfg_path.exists():
        parser.read(cfg_path)
        if parser.has_section("plot"):
            section = parser["plot"]

            # Floats
            for key in FLOAT_DEFAULTS:
                if key in section:
                    try:
                        float_values[key] = float(section[key])
                    except ValueError:
                        # Keep default if invalid
                        pass

            # Bools (accept 1/0, true/false, yes/no)
            for key in BOOL_DEFAULTS:
                if key in section:
                    raw = section[key].strip().lower()
                    if raw in ("1", "true", "yes", "on"):
                        bool_values[key] = True
                    elif raw in ("0", "false", "no", "off"):
                        bool_values[key] = False
                    # else keep default

        # Strings under [style] (empty values keep defaults)
        if parser.has_section("style"):
            for key in STYLE_DEFAULTS:
                if key in parser["style"] and parser["style"][key].strip():
                    style_values[key] = parser["style"][key]

    return float_values, bool_values, style_values


_FLOATS, _BOOLS, _STYLE = _load_config()

BUFFER = _FLOATS["buffer"]
UCR = _FLOATS["ucr"]
MIN_LAT = _FLOATS["min_lat"]
MAX_LAT = _FLOATS["max_lat"]
MIN_LON = _FLOATS["min_lon"]
MAX_LON = _FLOATS["max_lon"]
MINLAT_OFFSET = _FLOATS["minlat_offset"]
MAXLAT_OFFSET = _FLOATS["maxlat_offset"]
OUTPUT_DPI = int(_FLOATS["output_dpi"])

SHOW_CONE = _BOOLS["show_cone"]
SHOW_LEGEND = _BOOLS["show_legend"]
SHOW_PORTS = _BOOLS["show_ports"]
SHOW_MOVEMENT_TABLE = _BOOLS["show_movement_table"]
SHOW_ACE_BOX = _BOOLS["show_ace_box"]
SHOW_MAX_WIND_BOXES = _BOOLS["show_max_wind_boxes"]
SHOW_FOOTER = _BOOLS["show_footer"]

ORGANIZE_BY_YEAR = _BOOLS["organize_by_year"]
SHOW_LANDFALL = _BOOLS["show_landfall"]
SHOW_APPROACH_TABLE = _BOOLS["show_approach_table"]
SHOW_FORECAST_TABLE = _BOOLS["show_forecast_table"]
SHOW_FORECAST_KEY = _BOOLS["show_forecast_key"]
FULL_TRACK_EXTENT = _BOOLS["full_track_extent"]

APPROACH_RADIUS = _FLOATS["approach_radius"]
APPROACH_PORTS = int(_FLOATS["approach_ports"])

FORECAST_TZ_OFFSET = _FLOATS["forecast_tz_offset"]
FORECAST_TABLE_MAX_COLS = int(_FLOATS["forecast_table_max_cols"])
FORECAST_TABLE_MIN_FONTSIZE = _FLOATS["forecast_table_min_fontsize"]

WIND_RADIUS_EXTENT = _BOOLS["wind_radius_extent"]
WIND_RADIUS_PAD = _FLOATS["wind_radius_pad"]
SHOW_GRID = _BOOLS["show_grid"]
SHOW_SCALE_BAR = _BOOLS["show_scale_bar"]

THEME = _STYLE["theme"]
BRAND_NAME = _STYLE["brand_name"]
DATE_FORMAT = _STYLE["date_format"]
FOOTER_TEXT = _STYLE["footer_text"]
FORECAST_TIME_LABEL = _STYLE["forecast_time_label"]
FORECAST_SPEED_LABEL = _STYLE["forecast_speed_label"]
FORECAST_SPEED_UNIT = _STYLE["forecast_speed_unit"]
FORECAST_SPEED_MODE = _STYLE["forecast_speed_mode"].strip().lower()
OUTPUT_NAME = _STYLE["output_name"]
SHOW_LANDFALL = _BOOLS["show_landfall"]
SHOW_LANDFALL_LABEL = _BOOLS["show_landfall_label"]  # <- এই লাইনটি যোগ করুন
SHOW_STATES = _BOOLS["show_states"]
SHOW_STATE_LABELS = _BOOLS["show_state_labels"]



def output_stem(cyclone_name, override=None):
    """
    File name (without extension) for the output PNG.

    Priority: `--name` on the command line, then [style] output_name in
    config.ini, then the built-in "<Name>_Track".  Any "{name}" in the
    template is replaced by the cyclone/invest name and a trailing ".png"
    (if the user typed one) is dropped.
    """
    template = (override or OUTPUT_NAME or "{name}_Track").strip()
    stem = template.replace("{name}", str(cyclone_name))
    if stem.lower().endswith(".png"):
        stem = stem[:-4]
    return stem

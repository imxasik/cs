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
}

# Default boolean toggles
BOOL_DEFAULTS = {
    "show_cone": True,
    "show_legend": True,
    "show_ports": True,
    "show_port_table": True,
    "show_movement_table": True,
    "show_ace_box": True,
    "show_max_wind_boxes": True,
    "show_footer": True,
    "show_ai_position": True,   # NEW
    "show_bias_track": True,    # NEW
}


def _load_config():
    root_dir = Path(__file__).resolve().parent.parent
    cfg_path = root_dir / "config.ini"
    parser = configparser.ConfigParser()

    float_values = FLOAT_DEFAULTS.copy()
    bool_values = BOOL_DEFAULTS.copy()

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

    return float_values, bool_values


_FLOATS, _BOOLS = _load_config()

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
SHOW_PORT_TABLE = _BOOLS["show_port_table"]
SHOW_MOVEMENT_TABLE = _BOOLS["show_movement_table"]
SHOW_ACE_BOX = _BOOLS["show_ace_box"]
SHOW_MAX_WIND_BOXES = _BOOLS["show_max_wind_boxes"]
SHOW_FOOTER = _BOOLS["show_footer"]

# NEW exports
SHOW_AI_POSITION = _BOOLS["show_ai_position"]
SHOW_BIAS_TRACK = _BOOLS["show_bias_track"]

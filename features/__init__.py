"""Pluggable feature modules.

To add a new feature, create a new .py file in this folder with a function:

    def run_feature(context: dict):
        ...

The context dict contains:
    - cyclone_name
    - track_obs
    - track_for
    - is_invest
    - output_dir
    - map_file
    - output_image
    - landfall    (dict or None: time/lon/lat/place/place_km/time_str)
    - approaches  (list of dicts: name/dist_km/bearing/dir_str for the
                   CURRENT centre, plus side/approach_km/time/time_str/past.
                   dir_str is the 16-point cardinal direction of the storm
                   centre as seen from that port)

Your feature file will be auto-discovered and run after the main plot
is generated, without editing any other Python file.
Files whose names start with "_" are skipped.
"""

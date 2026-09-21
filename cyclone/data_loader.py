import io
import pandas as pd

def process_combined_cyclone_data(file_path):
    """
    Read a combined observed/forecast file in the following format:

    First line:  "Tropical Cyclone: NAME" or "Invest: NAME"
    Then:
        Synoptic Time,Latitude,Longitude,Intensity,Pressure
        ...observed rows...
    Then a line: "=== FORECAST ==="
    Then:
        Synoptic Time,Latitude,Longitude,Intensity,WindR24,WindR34,WindR64
        ...forecast rows...
    """
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        is_invest = "Invest" in first_line
        cyclone_name = first_line.split(":", 1)[1].strip()

        observed_lines, forecast_lines = [], []
        reading_observed = True
        observed_header_skipped = False
        forecast_header_skipped = False

        for line in file:
            line = line.strip()
            if not line:
                continue
            if line == "=== FORECAST ===":
                reading_observed = False
                continue

            if reading_observed:
                if not observed_header_skipped and line.startswith("Synoptic"):
                    observed_header_skipped = True
                    continue
                if observed_header_skipped:
                    observed_lines.append(line)
            else:
                if not forecast_header_skipped and line.startswith("Synoptic"):
                    forecast_header_skipped = True
                    continue
                if forecast_header_skipped:
                    forecast_lines.append(line)

    if observed_lines:
        obs_data_str = "\n".join(observed_lines)
        track_obs = pd.read_csv(
            io.StringIO(obs_data_str),
            names=["tnd", "Latitude", "Longitude", "Intensity", "Pressure"],
            header=None
        )
        track_obs["WindR24"] = 0
        track_obs["WindR34"] = 0
        track_obs["WindR64"] = 0
    else:
        track_obs = pd.DataFrame()

    if forecast_lines:
        for_data_str = "\n".join(forecast_lines)
        track_for = pd.read_csv(
            io.StringIO(for_data_str),
            names=["tnd", "Latitude", "Longitude", "Intensity", "WindR24", "WindR34", "WindR64"],
            header=None
        )
        track_for["Pressure"] = float("nan")
    else:
        track_for = pd.DataFrame()

    if not track_obs.empty:
        track_obs["tnd"] = pd.to_datetime(track_obs["tnd"], format="%Y-%m-%d %H:%M")
    if not track_for.empty:
        track_for["tnd"] = pd.to_datetime(track_for["tnd"], format="%Y-%m-%d %H:%M")

    return cyclone_name, track_obs, track_for, is_invest

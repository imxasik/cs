import pandas as pd

def calculate_ace(track_data):
    """Accumulate Cyclone Energy (ACE) from observed track dataframe."""
    if len(track_data) < 2:
        return 0.0

    track_data_sorted = track_data.sort_values("tnd").copy()
    track_data_sorted["tnd"] = pd.to_datetime(track_data_sorted["tnd"])

    ace = 0.0
    i = 0
    while i < len(track_data_sorted):
        window_end_time = track_data_sorted["tnd"].iloc[i] + pd.Timedelta(hours=6)
        window_end_idx = i
        while (
            window_end_idx < len(track_data_sorted)
            and track_data_sorted["tnd"].iloc[window_end_idx] < window_end_time
        ):
            window_end_idx += 1

        max_wind = track_data_sorted["Intensity"].iloc[i:window_end_idx].max()
        if max_wind >= 34:
            ace += (max_wind ** 2)

        i = window_end_idx if window_end_idx > i + 1 else i + 1

    return round(ace / 10000, 3)

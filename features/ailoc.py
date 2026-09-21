import os
import numpy as np
import pandas as pd


def load_landfall_dataset(csv_path="../assets/climo.csv"):
    """
    Load pre-built training file.
    Features:  LastLat, LastLon, LastWind
    Targets:   LandfallLat, LandfallLon
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    
    X = df[["LastLat", "LastLon", "LastWind"]].values.astype(float)
    y_lat = df["LandfallLat"].values.astype(float)
    y_lon = df["LandfallLon"].values.astype(float)

    return X, y_lat, y_lon


def _knn_predict(x, X, y, k=10):
    """
    Very small k-NN regression written by hand.
    x : (n_features,)   current cyclone feature vector
    X : (n_samples, n_features) training features
    y : (n_samples,)   target (lat or lon)
    """
    x = np.asarray(x, dtype=float)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.shape[0] == 0:
        # no training data
        return float("nan")

    # Euclidean distance in feature space
    diffs = X - x
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))

    # choose up to k nearest neighbours
    k = min(k, len(dists))
    idx = np.argsort(dists)[:k]
    sel_d = dists[idx]
    sel_y = y[idx]

    # avoid division by zero
    sel_d[sel_d == 0] = 1e-6

    # inverse distance weighting
    w = 1.0 / sel_d
    w = w / np.sum(w)

    return float(np.sum(sel_y * w))


def extract_current_features(track_data_obs):
    """
    Build feature vector for the *current* cyclone
    from your existing track_data_obs DataFrame.

    Uses only the last observed point:
    [Latitude, Longitude, Intensity]
    """
    last = track_data_obs.iloc[-1]

    last_lat = float(last["Latitude"])
    last_lon = float(last["Longitude"])

    try:
        last_wind = float(last["Intensity"])
        if pd.isna(last_wind):
            last_wind = 0.0
    except Exception:
        last_wind = 0.0

    return np.array([last_lat, last_lon, last_wind], dtype=float)


def predict_landfall_latlon(track_data_obs,
                            training_csv_path=None,
                            k=10):
    """
    Public function to call from plotting.py.

    Returns:
        (pred_lat, pred_lon) as floats
    """
    X, y_lat, y_lon = load_landfall_dataset(training_csv_path)
    x_current = extract_current_features(track_data_obs)

    pred_lat = _knn_predict(x_current, X, y_lat, k=k)
    pred_lon = _knn_predict(x_current, X, y_lon, k=k)

    return pred_lat, pred_lon

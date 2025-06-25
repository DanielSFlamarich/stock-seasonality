# src/utils/load_data.py

import pandas as pd


def load_latest_parquet(ticker: str, folder="data/external"):
    """
    Loads the most recent file matching the ticker from the external folder.
    """
    import glob
    import os

    pattern = os.path.join(folder, f"{ticker}_*.parquet")
    files = sorted(glob.glob(pattern), reverse=True)
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return pd.read_parquet(files[0])

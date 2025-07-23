import numpy as np
import pandas as pd


def generate_perfect_seasonality(
    start="2020-01-01", end="2025-01-01", freq="D", ticker="PERFECT.SYN", seed=None
):
    """
    Create a synthetic time series with perfect seasonality,
    a linear trend, and small noise to validate seasonality metrics.

    Args:
        start (str): Start date.
        end (str): End date.
        freq (str): Frequency of data ('D' for daily).
        ticker (str): Ticker label for synthetic asset.
        seed (int or None): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Time series with columns ['date', 'ticker', 'interval', 'close'].
    """
    if seed is not None:
        np.random.seed(seed)

    dates = pd.date_range(start=start, end=end, freq=freq)
    n = len(dates)

    days_in_year = 365.25
    days_in_month = days_in_year / 12
    days_in_week = 7

    # seasonal components
    yearly = 10 * np.sin(2 * np.pi * np.arange(n) / days_in_year)
    monthly = 3 * np.sin(2 * np.pi * np.arange(n) / days_in_month)
    weekly = 1.5 * np.sin(2 * np.pi * np.arange(n) / days_in_week)

    # trend and noise
    trend = 0.05 * np.arange(n)
    noise = np.random.normal(0, 0.1, size=n)

    close = 100 + trend + yearly + monthly + weekly + noise

    df = pd.DataFrame(
        {"date": dates, "ticker": ticker, "interval": "1d", "close": close}
    )

    return df


def generate_perfect_seasonality_all_intervals(
    start="2020-01-01", end="2025-01-01", ticker="PERFECT.SYN", seed=None
):
    """
    Create synthetic time series data with perfect yearly, monthly, and weekly
    seasonality across daily, weekly, and monthly intervals.

    Args:
        start (str): Start date.
        end (str): End date.
        ticker (str): Ticker label for synthetic asset.
        seed (int or None): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Combined DataFrame across intervals ['1d', '1wk', '1mo'].
    """
    df_daily = generate_perfect_seasonality(
        start=start, end=end, ticker=ticker, seed=seed
    )

    def resample_df(df, rule, interval_label):
        df_r = df.copy()
        df_r.set_index("date", inplace=True)

        df_resampled = df_r.resample(rule).agg({"close": "last"}).dropna()

        df_resampled["ticker"] = df["ticker"].iloc[0]
        df_resampled["interval"] = interval_label
        df_resampled = df_resampled.reset_index()

        return df_resampled

    df_weekly = resample_df(df_daily, "W", "1wk")
    df_monthly = resample_df(df_daily, "ME", "1mo")

    return pd.concat([df_daily, df_weekly, df_monthly], ignore_index=True)

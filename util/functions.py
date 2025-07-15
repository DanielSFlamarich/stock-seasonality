from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def detect_outliers(df, n, features):
    """
    Detect observations with more than n outliers based on IQR logic.

    Args:
        df (pd.DataFrame): DataFrame to analyze.
        n (int): Number of outlier features to consider an observation an outlier.
        features (list): Columns to check for outliers.

    Returns:
        list: Indexes of rows with n or more outlier values.
    """
    outlier_indices = []

    for col in features:
        q_1 = np.percentile(df[col], 25)
        q_3 = np.percentile(df[col], 75)
        iqr = q_3 - q_1
        outlier_step = 1.5 * iqr

        outlier_list_col = df[
            (df[col] < q_1 - outlier_step) | (df[col] > q_3 + outlier_step)
        ].index

        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = [k for k, v in outlier_indices.items() if v >= n]

    return multiple_outliers


def metric_characterisation(data, var, confidence):
    """
    Describe the metric's central tendency and variation.

    Args:
        data (pd.DataFrame): Input DataFrame.
        var (str): Column name of metric.
        confidence (float): Confidence level for CI.

    Returns:
        None (prints stats and plots)
    """
    a = np.array(data[var], dtype=float)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)

    q_1 = np.percentile(data[var], 25)
    q_3 = np.percentile(data[var], 75)
    qcd = (q_3 - q_1) / (q_3 + q_1)

    ten_up = m * 1.1
    ten_down = m * 0.9

    print(f"                                          {var}")
    print(f"                           Baseline (average): {m}")
    print(f"                           Standard Deviation: {a.std()}")
    print(f"                           QCD: {qcd}")
    print(f"                           Coefficient of Variation: {a.std() / m}")
    print(f"                           {confidence * 100}% CI Lower: {m - h}")
    print(f"                           {confidence * 100}% CI Upper: {m + h}")
    print(f"                           10% increase: {ten_up}")
    print(f"                           10% decrease: {ten_down}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    data[var].plot(ax=ax1)
    sns.histplot(data[var], kde=True, ax=ax2)
    ax1.tick_params(axis="x", rotation=45)


def calculate_anomalies(df, alpha):
    """
    Create anomaly signals and aggregate them using heuristic weights.

    Args:
        df (pd.DataFrame): Source pipeline.
        alpha (float): EMA smoothing factor.

    Returns:
        pd.DataFrame: DataFrame with raw and smoothed anomaly signals.
    """
    anomaly_1 = ((df["created_total"] >= 1) & (df["pending_total"] == 0)).astype(int)
    anomaly_2 = (df["errors_total"] > df["completed_total"]).astype(int)
    anomaly_3 = (
        (df["completed_total"].diff() < 0)
        & (df["errors_total"].diff() > 0)
        & (df["transf_start_total"].diff() > 0)
    ).astype(int)
    anomaly_4 = (
        (df["completed_total"] == 0)
        & (df["errors_total"].diff() > 0)
        & (df["transf_start_total"].diff() > 0)
    ).astype(int)
    anomaly_5 = df["created_total"] > df["pending_total"].quantile(0.2)
    anomaly_6 = (
        (df["completed_total"] == 0)
        & (df["pending_total"] != 0)
        & (df["errors_total"] > df["pending_total"])
    ).astype(int)

    df_with_metric = df.copy()
    df_with_metric["health_anom_1"] = (
        (anomaly_1 * 2)
        + (anomaly_2 * 3)
        + anomaly_3
        + (anomaly_4 * 3)
        + anomaly_5
        + anomaly_6
    )

    scaler = MinMaxScaler()
    df_with_metric["health_anom_1_normed"] = scaler.fit_transform(
        df_with_metric["health_anom_1"].values.reshape(-1, 1)
    )

    df_with_metric["health_anom_exp_1"] = (
        df_with_metric["health_anom_1"].ewm(alpha=alpha).mean()
    )

    df_with_metric["health_anom_normed_exp_1"] = (
        df_with_metric["health_anom_1_normed"].ewm(alpha=alpha).mean()
    )

    return df_with_metric


def calculate_flow(df, alpha):
    """
    Calculate smoothed flow categories based on completed volume.

    Args:
        df (pd.DataFrame): Source pipeline.
        alpha (float): EMA smoothing factor.

    Returns:
        pd.DataFrame: DataFrame with flow metric columns.
    """
    a1 = ((df["completed_total"] >= 1) & (df["completed_total"] <= 6)).astype(int)
    a2 = ((df["completed_total"] >= 7) & (df["completed_total"] <= 12)).astype(int)

    df_with_metric = df.copy()
    df_with_metric["health_anom_2"] = a1 + a2

    scaler = MinMaxScaler()
    df_with_metric["health_anom_2_normed"] = scaler.fit_transform(
        df_with_metric["health_anom_2"].values.reshape(-1, 1)
    )

    df_with_metric["health_anom_exp_2"] = (
        df_with_metric["health_anom_2"].ewm(alpha=alpha).mean()
    )

    df_with_metric["health_anom_normed_exp_2"] = (
        df_with_metric["health_anom_2_normed"].ewm(alpha=alpha).mean()
    )

    return df_with_metric


def linear_combination(df, metric1, metric2, w1, w2):
    """
    Linearly combine two metrics and normalize the result.

    Args:
        df (pd.DataFrame): Input pipeline.
        metric1 (str): First metric column name.
        metric2 (str): Second metric column name.
        w1 (float): Weight for metric1.
        w2 (float): Weight for metric2.

    Returns:
        pd.DataFrame: Updated DataFrame with combined metrics.
    """
    df["combined_metric"] = w1 * df[metric1] + w2 * df[metric2]
    df["norm_combined_metric"] = (
        df["combined_metric"] - df["combined_metric"].min()
    ) / (df["combined_metric"].max() - df["combined_metric"].min())

    return df

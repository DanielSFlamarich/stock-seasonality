# tests/reporting/test_build_features.py

"""
Unit tests for src/reporting/build_features.py.

Test strategy:
- All tests use synthetic fixtures from conftest.py — no network calls.
- Tests are grouped by concern using classes for organisation.
- Where possible, assertions check mathematical properties (e.g. mean within
  bounds) rather than hardcoded floats, so they remain valid if the generator
  changes slightly.
"""

import numpy as np
import pandas as pd
import pytest

from src.reporting.build_features import (
    ALL_METRIC_COLS,
    build_features,
)

# schema tests; correct columns, types, and row counts


class TestSchema:

    def test_output_columns_present(self, df_rolling_perfect):
        """Every expected aggregation column must exist in the output."""
        result = build_features(df_rolling_perfect)

        assert "ticker" in result.columns
        assert "freq" in result.columns
        assert "window_count" in result.columns
        assert "last_window" in result.columns

        for col in ALL_METRIC_COLS:
            for suffix in ("_mean", "_std", "_latest", "_trend"):
                assert f"{col}{suffix}" in result.columns, f"Missing expected column: {col}{suffix}"

    def test_one_row_per_ticker_freq(self, df_rolling_combined):
        """Output must have exactly one row per (ticker, freq) combination."""
        result = build_features(df_rolling_combined)
        assert not result.duplicated(subset=["ticker", "freq"]).any()

    def test_row_count_matches_unique_groups(self, df_rolling_combined):
        """Row count must equal the number of unique (ticker, freq) pairs."""
        result = build_features(df_rolling_combined)
        expected = df_rolling_combined.groupby(["ticker", "freq"]).ngroups
        assert len(result) == expected

    def test_last_window_is_datetime(self, df_rolling_perfect):
        """last_window column must be datetime dtype."""
        result = build_features(df_rolling_perfect)
        assert pd.api.types.is_datetime64_any_dtype(result["last_window"])

    def test_window_count_is_positive(self, df_rolling_perfect):
        """window_count must be a positive integer for every row."""
        result = build_features(df_rolling_perfect)
        assert (result["window_count"] > 0).all()


# value correctness; aggregation logic


class TestValues:

    def test_mean_within_metric_range(self, df_rolling_perfect):
        """
        Normalised metrics and scores live in [0, 1].
        Their means must therefore also be in [0, 1].
        NaN is acceptable — STL legitimately fails on short windows (e.g. ME
        windows only have ~20 observations, below the 2*period threshold).
        We only assert bounds on rows where a value is actually present.
        """
        result = build_features(df_rolling_perfect)
        for col in ALL_METRIC_COLS:
            non_null = result[f"{col}_mean"].dropna()
            assert (non_null >= 0).all(), f"{col}_mean has negative values"
            assert (non_null <= 1).all(), f"{col}_mean exceeds 1"

    def test_std_is_non_negative(self, df_rolling_perfect):
        """
        Standard deviation can never be negative.
        Skip NaN rows — same STL constraint as test_mean_within_metric_range.
        """
        result = build_features(df_rolling_perfect)
        for col in ALL_METRIC_COLS:
            non_null = result[f"{col}_std"].dropna()
            assert (non_null >= 0).all(), f"{col}_std has negative values"

    def test_latest_matches_last_window(self, df_rolling_perfect):
        """
        The _latest value must equal the metric value in the most recent
        window for that (ticker, freq) group.
        When both values are NaN (e.g. STL on short windows), that is also
        a valid match — np.isclose(nan, nan) returns False so we check
        explicitly.
        """
        result = build_features(df_rolling_perfect)
        df = df_rolling_perfect.copy()
        df["window_start"] = pd.to_datetime(df["window_start"])

        for _, row in result.iterrows():
            group = df[(df["ticker"] == row["ticker"]) & (df["freq"] == row["freq"])].sort_values(
                "window_start"
            )

            for col in ALL_METRIC_COLS:
                expected = group[col].iloc[-1]
                actual = row[f"{col}_latest"]

                # both NaN is a valid match
                both_nan = pd.isna(actual) and pd.isna(expected)
                if not both_nan:
                    assert np.isclose(actual, expected), (
                        f"{col}_latest mismatch for {row['ticker']} {row['freq']}: "
                        f"got {actual}, expected {expected}"
                    )

    def test_trend_is_finite(self, df_rolling_perfect):
        """
        Where a trend value is present it must be a finite float.
        NaN trend is acceptable when the underlying metric (e.g. stl_strength)
        is all-NaN for that group — we don't assert on those rows.
        """
        result = build_features(df_rolling_perfect)
        for col in ALL_METRIC_COLS:
            non_null = result[f"{col}_trend"].dropna()
            assert np.isfinite(non_null).all(), f"{col}_trend contains non-finite values"

    def test_window_count_matches_actual_windows(self, df_rolling_perfect):
        """window_count must equal the actual row count for that group in the input."""
        result = build_features(df_rolling_perfect)
        for _, row in result.iterrows():
            actual = len(
                df_rolling_perfect[
                    (df_rolling_perfect["ticker"] == row["ticker"])
                    & (df_rolling_perfect["freq"] == row["freq"])
                ]
            )
            assert row["window_count"] == actual


# last_n_windows parameter


class TestLastNWindows:

    def test_window_count_capped_at_n(self, df_rolling_perfect):
        """When last_n_windows=6, window_count must be <= 6 for every row."""
        result = build_features(df_rolling_perfect, last_n_windows=6)
        assert (result["window_count"] <= 6).all()

    def test_last_window_unchanged_by_restriction(self, df_rolling_perfect):
        """
        The most recent window date must be identical whether we use all
        windows or restrict to last N — we trim from the old end, not the new.
        """
        full = build_features(df_rolling_perfect)
        restricted = build_features(df_rolling_perfect, last_n_windows=6)

        merged = full.merge(restricted, on=["ticker", "freq"], suffixes=("_full", "_n"))
        assert (merged["last_window_full"] == merged["last_window_n"]).all()

    def test_last_n_windows_affects_mean(self, df_rolling_perfect):
        """
        Restricting to last N windows should produce a different mean than
        the full series. If identical, the parameter is not filtering.
        """
        full = build_features(df_rolling_perfect)
        restricted = build_features(df_rolling_perfect, last_n_windows=3)

        mean_cols = [f"{col}_mean" for col in ALL_METRIC_COLS]
        any_diff = any(not full[c].equals(restricted[c]) for c in mean_cols)
        assert any_diff, (
            "Full and last_n_windows=3 produced identical means — "
            "last_n_windows parameter may not be filtering correctly."
        )

    def test_invalid_last_n_windows_raises(self, df_rolling_perfect):
        """Non-positive last_n_windows must raise ValueError."""
        with pytest.raises(ValueError):
            build_features(df_rolling_perfect, last_n_windows=0)

        with pytest.raises(ValueError):
            build_features(df_rolling_perfect, last_n_windows=-5)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_empty_dataframe_returns_empty(self):
        """An empty input must return an empty DataFrame, not raise."""
        empty = pd.DataFrame(
            columns=[
                "ticker",
                "interval",
                "freq",
                "window_start",
                "acf_lag_val",
                "p2m_val",
                "stl_strength",
                "seasonality_score_linear",
                "seasonality_score_geom",
                "seasonality_score_harmonic",
            ]
        )
        result = build_features(empty)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_missing_column_raises_valueerror(self, df_rolling_perfect):
        """Dropping a required column must raise ValueError naming the missing column."""
        broken = df_rolling_perfect.drop(columns=["acf_lag_val"])
        with pytest.raises(ValueError, match="acf_lag_val"):
            build_features(broken)

    def test_single_window_per_group(self):
        """
        A group with only one window must produce a valid row with std=0 and
        trend=0, since last and first are the same observation.
        """
        single = pd.DataFrame(
            [
                {
                    "ticker": "SINGLE.SYN",
                    "interval": "1d",
                    "freq": "ME",
                    "window_start": pd.Timestamp("2024-01-31"),
                    "acf_lag_val": 0.8,
                    "p2m_val": 0.7,
                    "stl_strength": 0.9,
                    "seasonality_score_linear": 0.8,
                    "seasonality_score_geom": 0.78,
                    "seasonality_score_harmonic": 0.76,
                }
            ]
        )

        result = build_features(single)
        assert len(result) == 1
        assert result["window_count"].iloc[0] == 1

        for col in ALL_METRIC_COLS:
            assert (
                result[f"{col}_std"].iloc[0] == 0.0
            ), f"{col}_std should be 0 for single-window group"
            assert (
                result[f"{col}_trend"].iloc[0] == 0.0
            ), f"{col}_trend should be 0 for single-window group"

# tests/reporting/conftest.py

"""
Shared pytest fixtures for src/reporting/ test suite.

Design principles:
- No network call --> all data is generated synthetically.
- seed=42 throughout --> results are deterministic across runs and machines.
- Two contrast tickers:
    PERFECT.SYN : strong, known seasonality (yearly + monthly + weekly sine waves).
                  Any flagging logic MUST flag this ticker.
    FLAT.SYN    : pure Gaussian noise, no seasonal structure.
                  Any flagging logic MUST NOT flag this ticker.
- df_rolling_* fixtures run the real SeasonalityETL so the DataFrame schema
  is identical to what the production pipeline produces.
"""

import numpy as np
import pandas as pd
import pytest
from dotenv import load_dotenv

from src.pipeline.seasonality_etl import SeasonalityETL
from src.visualization.synthetic_data_generator import generate_perfect_seasonality

load_dotenv()

# raw price fixtures (before ETL)


@pytest.fixture(scope="session")
def df_synth_perfect() -> pd.DataFrame:
    """
    Perfectly seasonal synthetic price series.
    5 years of daily data (2020-2025), seed=42.
    Columns: date, ticker, interval, close.
    """
    return generate_perfect_seasonality(
        start="2020-01-01",
        end="2025-01-01",
        ticker="PERFECT.SYN",
        seed=42,
    )


@pytest.fixture(scope="session")
def df_synth_flat() -> pd.DataFrame:
    """
    Flat (pure noise) synthetic price series with no seasonal structure.
    Same date range and shape as df_synth_perfect for a fair comparison.
    Columns: date, ticker, interval, close.
    """
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", end="2025-01-01", freq="D")
    close = 100 + np.random.normal(0, 1, size=len(dates))

    return pd.DataFrame(
        {
            "date": dates,
            "ticker": "FLAT.SYN",
            "interval": "1d",
            "close": close,
        }
    )


# rolling metrics fixtures (after ETL; this is what build_features consumes)


@pytest.fixture(scope="session")
def df_rolling_perfect(df_synth_perfect) -> pd.DataFrame:
    """
    Rolling seasonality metrics for PERFECT.SYN.

    Uses only ME and QE frequencies to keep fixture fast while still covering
    two different window sizes. W is omitted here because weekly windows on
    daily data produce many small windows where STL often returns NaN, which
    would muddy the aggregation assertions.
    """
    etl = SeasonalityETL()
    return etl.fit_rolling(df_synth_perfect, frequencies=["ME", "QE"])


@pytest.fixture(scope="session")
def df_rolling_flat(df_synth_flat) -> pd.DataFrame:
    """
    Rolling seasonality metrics for FLAT.SYN (noise only).
    Same frequencies as df_rolling_perfect for a direct comparison.
    """
    etl = SeasonalityETL()
    return etl.fit_rolling(df_synth_flat, frequencies=["ME", "QE"])


@pytest.fixture(scope="session")
def df_rolling_combined(df_rolling_perfect, df_rolling_flat) -> pd.DataFrame:
    """
    Both tickers concatenated into a single df_rolling.
    Used by tests that need a realistic multi-ticker input,
    e.g. percentile ranking in flag_tickers.py.
    """
    return pd.concat([df_rolling_perfect, df_rolling_flat], ignore_index=True)


@pytest.fixture(scope="session")
def df_synth_combined(df_synth_perfect, df_synth_flat) -> pd.DataFrame:
    """
    Raw price data for both synthetic tickers concatenated.
    Used by peak_analysis and any test that needs combined raw prices.
    Mirrors df_rolling_combined but at the raw (pre-ETL) level.
    """
    return pd.concat([df_synth_perfect, df_synth_flat], ignore_index=True)

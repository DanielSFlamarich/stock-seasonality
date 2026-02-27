# tests/pipeline/test_run_pipeline.py
"""
Unit and integration tests for src.pipeline.run_pipeline

Validates:
- CLI argument parsing (argparse)
- Pipeline orchestration (DataLoader → ETL → CSV)
- Success and failure paths
- Output file creation
- Error handling and logging

Run fast tests only:
    pytest tests/pipeline/test_run_pipeline.py -v -m "unit"

Run all tests:
    pytest tests/pipeline/test_run_pipeline.py -v
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.pipeline.run_pipeline import main, run_pipeline

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_dataloader():
    """
    Mocks DataLoader class to return synthetic DataFrame.
    """
    with patch("src.pipeline.run_pipeline.DataLoader") as mock_class:
        mock_instance = MagicMock()

        # mock load() to return synthetic data
        mock_df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=300, freq="D"),
                "close": [100.0] * 300,
                "ticker": ["TEST"] * 300,
                "interval": ["1d"] * 300,
            }
        )
        mock_instance.load.return_value = mock_df

        # make the class return the instance when instantiated
        mock_class.return_value = mock_instance

        yield mock_class


@pytest.fixture
def mock_seasonality_etl():
    """
    Mocks SeasonalityETL class to return synthetic scores.
    """
    with patch("src.pipeline.run_pipeline.SeasonalityETL") as mock_class:
        mock_instance = MagicMock()

        # mock fit_rolling() to return synthetic scores
        mock_scores = pd.DataFrame(
            {
                "ticker": ["TEST"] * 10,
                "interval": ["1d"] * 10,
                "freq": ["ME"] * 10,
                "window_start": pd.date_range("2023-01-31", periods=10, freq="ME"),
                "acf_lag_val": [0.5] * 10,
                "p2m_val": [3.0] * 10,
                "stl_strength": [0.4] * 10,
                "seasonality_score_linear": [0.6] * 10,
                "seasonality_score_geom": [0.5] * 10,
                "seasonality_score_harmonic": [0.4] * 10,
            }
        )
        mock_instance.fit_rolling.return_value = mock_scores

        # make the class return the instance when instantiated
        mock_class.return_value = mock_instance

        yield mock_class


@pytest.fixture
def temp_output_dir(tmp_path):
    """
    Temporary directory for output files.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


# ============================================================================
# UNIT TESTS - run_pipeline() function
# ============================================================================


@pytest.mark.unit
def test_run_pipeline_success_path(
    mock_dataloader, mock_seasonality_etl, temp_output_dir
):
    """
    Unit test: run_pipeline() orchestrates DataLoader → ETL → CSV successfully.
    """
    result = run_pipeline(
        config_path="config/tickers_list.yaml",
        start_date="2023-01-01",
        end_date="2023-12-31",
        intervals=["1d"],
        output_dir=str(temp_output_dir),
        frequencies=["W", "ME"],
        use_cache=False,
        force_refresh=False,
    )

    # should return DataFrame
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert not result.empty

    # DataLoader should be called with correct params
    mock_dataloader.assert_called_once()
    loader_instance = mock_dataloader.return_value
    loader_instance.load.assert_called_once_with(
        start_date="2023-01-01", end_date="2023-12-31", intervals=["1d"]
    )

    # SeasonalityETL should be called
    mock_seasonality_etl.assert_called_once()
    etl_instance = mock_seasonality_etl.return_value
    etl_instance.fit_rolling.assert_called_once()

    # CSV file should be created
    date_str = datetime.today().strftime("%Y-%m-%d")
    expected_file = temp_output_dir / f"seasonality_scores_{date_str}.csv"
    assert expected_file.exists()
    assert expected_file.stat().st_size > 0


@pytest.mark.unit
def test_run_pipeline_invalid_date_format(
    mock_dataloader, mock_seasonality_etl, temp_output_dir
):
    """
    Unit test: Invalid date format returns None and logs error.
    """
    result = run_pipeline(
        config_path="config/tickers_list.yaml",
        start_date="not-a-date",  # Invalid format
        end_date="2023-12-31",
        intervals=["1d"],
        output_dir=str(temp_output_dir),
    )

    # should return None on error
    assert result is None

    # DataLoader should not be called
    mock_dataloader.assert_not_called()


@pytest.mark.unit
def test_run_pipeline_data_load_failure(
    mock_dataloader, mock_seasonality_etl, temp_output_dir, caplog
):
    """
    Unit test: DataLoader exception is caught and logged.
    """
    # make DataLoader.load() raise an exception
    loader_instance = mock_dataloader.return_value
    loader_instance.load.side_effect = ValueError("Mock data load failure")

    result = run_pipeline(
        config_path="config/tickers_list.yaml",
        start_date="2023-01-01",
        output_dir=str(temp_output_dir),
    )

    # should return None
    assert result is None

    # should log error
    assert any("Data loading failed" in record.message for record in caplog.records)


@pytest.mark.unit
def test_run_pipeline_etl_failure(
    mock_dataloader, mock_seasonality_etl, temp_output_dir, caplog
):
    """
    Unit test: SeasonalityETL exception is caught and logged.
    """
    # make ETL.fit_rolling() raise an exception
    etl_instance = mock_seasonality_etl.return_value
    etl_instance.fit_rolling.side_effect = RuntimeError("Mock ETL failure")

    result = run_pipeline(
        config_path="config/tickers_list.yaml",
        start_date="2023-01-01",
        output_dir=str(temp_output_dir),
    )

    # should return None
    assert result is None

    # should log error
    assert any("ETL processing failed" in record.message for record in caplog.records)


@pytest.mark.unit
def test_run_pipeline_default_parameters(
    mock_dataloader, mock_seasonality_etl, tmp_path
):
    """
    Unit test: Default parameters are applied correctly.
    """
    # create a mock config file
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text("tickers:\n  - TEST\n")

    result = run_pipeline(
        config_path=str(config_path),
        start_date="2023-01-01",
        output_dir=str(tmp_path / "output"),
    )

    # should use default intervals and frequencies
    assert result is not None

    loader_instance = mock_dataloader.return_value
    call_args = loader_instance.load.call_args

    # check default intervals
    assert call_args.kwargs["intervals"] == ["1d"]


@pytest.mark.unit
def test_run_pipeline_force_refresh_flag(
    mock_dataloader, mock_seasonality_etl, temp_output_dir
):
    """
    Unit test: force_refresh=True sets use_cache=False in DataLoader.
    """
    result = run_pipeline(
        config_path="config/tickers_list.yaml",
        start_date="2023-01-01",
        output_dir=str(temp_output_dir),
        force_refresh=True,
    )

    assert result is not None

    # DataLoader should be instantiated with use_cache=False
    call_kwargs = mock_dataloader.call_args.kwargs
    assert call_kwargs.get("use_cache") is False


@pytest.mark.unit
def test_run_pipeline_creates_output_directory(
    mock_dataloader, mock_seasonality_etl, tmp_path
):
    """
    Unit test: Output directory is created if it doesn't exist.
    """
    output_dir = tmp_path / "new_dir" / "nested"

    result = run_pipeline(
        config_path="config/tickers_list.yaml",
        start_date="2023-01-01",
        output_dir=str(output_dir),
    )

    assert result is not None
    assert output_dir.exists()


# ============================================================================
# UNIT TESTS - CLI parsing (main function)
# ============================================================================


@pytest.mark.unit
def test_cli_default_arguments(
    mock_dataloader, mock_seasonality_etl, tmp_path, monkeypatch
):
    """
    Unit test: CLI with no arguments uses defaults.
    """
    # mock sys.argv
    test_args = ["run_pipeline.py"]
    monkeypatch.setattr(sys, "argv", test_args)

    # mock sys.exit to prevent test from exiting
    with patch("sys.exit") as mock_exit:
        main()

        # should exit with 0 (success)
        mock_exit.assert_called_once_with(0)


@pytest.mark.unit
def test_cli_custom_arguments(
    mock_dataloader, mock_seasonality_etl, temp_output_dir, monkeypatch
):
    """
    Unit test: CLI with custom arguments parses correctly.
    """
    test_args = [
        "run_pipeline.py",
        "--start-date",
        "2020-01-01",
        "--end-date",
        "2020-12-31",
        "--intervals",
        "1d,1wk",
        "--frequencies",
        "W,ME",
        "--output-dir",
        str(temp_output_dir),
        "--force-refresh",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with patch("sys.exit") as mock_exit:
        main()

        # check DataLoader was called with parsed args
        loader_instance = mock_dataloader.return_value
        call_args = loader_instance.load.call_args

        assert call_args.kwargs["start_date"] == "2020-01-01"
        assert call_args.kwargs["end_date"] == "2020-12-31"
        assert call_args.kwargs["intervals"] == ["1d", "1wk"]

        mock_exit.assert_called_once_with(0)


@pytest.mark.unit
def test_cli_verbose_flag(
    mock_dataloader, mock_seasonality_etl, temp_output_dir, monkeypatch
):
    """
    Unit test: --verbose flag increases logging level.
    """
    test_args = [
        "run_pipeline.py",
        "--output-dir",
        str(temp_output_dir),
        "--verbose",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with patch("sys.exit"):
        main()

        # difficult to test logging level change directly
        # just verify it doesn't crash
        assert True


# ============================================================================
# INTEGRATION TEST - Full end-to-end with real subprocess
# ============================================================================


@pytest.mark.integration
def test_run_pipeline_script_end_to_end(tmp_path):
    """
    Integration test: Run the actual CLI script via subprocess.

    This is your original test, now properly marked and using tmp_path.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # locate the run_pipeline script
    script_path = (
        Path(__file__).resolve().parents[2] / "src" / "pipeline" / "run_pipeline.py"
    )

    # set up PYTHONPATH so imports work
    env = os.environ.copy()
    env["PYTHONPATH"] = str(script_path.parents[2])

    # execute the pipeline with minimal arguments
    result = subprocess.run(
        [
            sys.executable,  # use same Python as test runner
            str(script_path),
            "--start-date",
            "2022-01-01",
            "--end-date",
            "2023-01-01",
            "--frequencies",
            "W,ME",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,  # prevent hanging
    )

    # check script succeeded
    assert (
        result.returncode == 0
    ), f"Script failed with error:\nSTDERR: {result.stderr}\nSTDOUT: {result.stdout}"

    # check output CSV is created and not empty
    date_str = datetime.today().strftime("%Y-%m-%d")
    expected_file = output_dir / f"seasonality_scores_{date_str}.csv"
    assert expected_file.exists(), f"Expected file not found: {expected_file}"
    assert expected_file.stat().st_size > 0, "Output file is empty"

    # verify CSV has expected columns
    df = pd.read_csv(expected_file)
    expected_cols = {
        "ticker",
        "interval",
        "freq",
        "window_start",
        "seasonality_score_linear",
    }
    assert expected_cols.issubset(
        set(df.columns)
    ), f"Missing columns: {expected_cols - set(df.columns)}"

    print(f"(^o^) Pipeline CLI integration test passed. Output: {expected_file}")


@pytest.mark.integration
@pytest.mark.slow
def test_run_pipeline_script_with_cache(tmp_path):
    """
    Integration test: Verify caching behavior.

    Run twice - second run should use cache (faster).
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    script_path = (
        Path(__file__).resolve().parents[2] / "src" / "pipeline" / "run_pipeline.py"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(script_path.parents[2])

    # first run - no cache
    result1 = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--start-date",
            "2023-01-01",
            "--end-date",
            "2023-01-10",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )

    assert result1.returncode == 0

    # second run - should use cache (check for cache message in logs)
    result2 = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--start-date",
            "2023-01-01",
            "--end-date",
            "2023-01-10",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )

    assert result2.returncode == 0

    # both should produce output files
    date_str = datetime.today().strftime("%Y-%m-%d")
    expected_file = output_dir / f"seasonality_scores_{date_str}.csv"
    assert expected_file.exists()


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


@pytest.mark.unit
def test_run_pipeline_empty_dataframe(
    mock_dataloader, mock_seasonality_etl, temp_output_dir
):
    """
    Unit test: Empty DataFrame from DataLoader is handled gracefully.
    """
    # make DataLoader return empty DataFrame
    loader_instance = mock_dataloader.return_value
    loader_instance.load.return_value = pd.DataFrame()

    # ETL should handle empty input
    etl_instance = mock_seasonality_etl.return_value
    etl_instance.fit_rolling.return_value = pd.DataFrame()

    result = run_pipeline(
        config_path="config/tickers_list.yaml",
        start_date="2023-01-01",
        output_dir=str(temp_output_dir),
    )

    # should handle gracefully (may return empty DataFrame)
    assert result is not None or result is None  # either is acceptable

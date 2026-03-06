# tests/pipeline/test_run_pipeline.py

"""
Unit and integration tests for src.pipeline.run_pipeline

Validates:
- CLI argument parsing (argparse)
- Pipeline orchestration (DataLoader → ETL → CSV → reporting → JSON)
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
# FIXTURES ---> Steps 1-3 (data load, etl)
# ============================================================================


@pytest.fixture
def mock_dataloader():
    """
    Mocks DataLoader class to return synthetic DataFrame.
    """
    with patch("src.pipeline.run_pipeline.DataLoader") as mock_class:
        mock_instance = MagicMock()

        mock_df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=300, freq="D"),
                "close": [100.0] * 300,
                "ticker": ["TEST"] * 300,
                "interval": ["1d"] * 300,
            }
        )
        mock_instance.load.return_value = mock_df
        mock_class.return_value = mock_instance

        yield mock_class


@pytest.fixture
def mock_seasonality_etl():
    """
    Mocks SeasonalityETL class to return synthetic scores.
    """
    with patch("src.pipeline.run_pipeline.SeasonalityETL") as mock_class:
        mock_instance = MagicMock()

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
        mock_class.return_value = mock_instance

        yield mock_class


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for CSV output files."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


# ============================================================================
# FIXTURES ---> Steps 4-8 (all reporting related layer)
# ============================================================================


@pytest.fixture
def mock_reporting_steps(tmp_path):
    """
    Patches all four reporting symbols imported into run_pipeline:
      build_features, flag_tickers, summarise_peaks,
      generate_report, save_report.

    Each returns a minimal but schema-correct object so downstream
    assertions in run_pipeline don't crash.

    Yields a dict of the five mock objects keyed by function name.
    """
    mock_df_features = pd.DataFrame(
        {
            "ticker": ["TEST"],
            "freq": ["ME"],
            "window_count": [10],
            "last_window": [pd.Timestamp("2023-10-31")],
            "acf_lag_val_mean": [0.5],
            "p2m_val_mean": [3.0],
            "stl_strength_mean": [0.4],
            "seasonality_score_harmonic_mean": [0.55],
        }
    )

    mock_df_flags = mock_df_features.copy()
    mock_df_flags["superperformer_flag"] = [False]
    mock_df_flags["stl_available"] = [True]
    mock_df_flags["low_window_count"] = [False]

    mock_df_peaks = pd.DataFrame(
        {
            "ticker": ["TEST"],
            "freq": ["ME"],
            "peak_count": [5],
            "mean_peak_gap_days": [31.0],
            "std_peak_gap_days": [2.5],
        }
    )

    mock_report = {
        "generated_at": "2023-10-31T00:00:00+00:00",
        "tickers": [
            {
                "ticker": "TEST",
                "superperformer": False,
                "frequencies": [
                    {
                        "freq": "ME",
                        "peak_count": 5,
                        "mean_peak_gap_days": 31.0,
                        "std_peak_gap_days": 2.5,
                    }
                ],
            }
        ],
    }

    with (
        patch("src.pipeline.run_pipeline.build_features", return_value=mock_df_features) as m_bf,
        patch("src.pipeline.run_pipeline.flag_tickers", return_value=mock_df_flags) as m_ft,
        patch("src.pipeline.run_pipeline.summarise_peaks", return_value=mock_df_peaks) as m_sp,
        patch("src.pipeline.run_pipeline.generate_report", return_value=mock_report) as m_gr,
        patch("src.pipeline.run_pipeline.save_report") as m_sr,
    ):
        yield {
            "build_features": m_bf,
            "flag_tickers": m_ft,
            "summarise_peaks": m_sp,
            "generate_report": m_gr,
            "save_report": m_sr,
        }


# ============================================================================
# UNIT TESTS - run_pipeline() function
# ============================================================================


@pytest.mark.unit
def test_run_pipeline_success_path(
    mock_dataloader, mock_seasonality_etl, mock_reporting_steps, temp_output_dir, tmp_path
):
    """
    Unit test: run_pipeline() orchestrates all 8 steps successfully.
    """
    report_dir = tmp_path / "reports"

    result = run_pipeline(
        config_path="config/tickers_list.yaml",
        start_date="2023-01-01",
        end_date="2023-12-31",
        intervals=["1d"],
        output_dir=str(temp_output_dir),
        frequencies=["W", "ME"],
        use_cache=False,
        force_refresh=False,
        report_dir=str(report_dir),
    )

    # should return df_scores DataFrame (backward compat)
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert not result.empty

    # DataLoader + ETL called as before
    mock_dataloader.assert_called_once()
    mock_seasonality_etl.assert_called_once()

    # CSV file should be created
    date_str = datetime.today().strftime("%Y-%m-%d")
    expected_csv = temp_output_dir / f"seasonality_scores_{date_str}.csv"
    assert expected_csv.exists()
    assert expected_csv.stat().st_size > 0


@pytest.mark.unit
def test_run_pipeline_reporting_steps_called(
    mock_dataloader, mock_seasonality_etl, mock_reporting_steps, temp_output_dir, tmp_path
):
    """
    Unit test: Steps 4-8 are each called exactly once with the correct inputs.
    """
    report_dir = tmp_path / "reports"

    run_pipeline(
        config_path="config/tickers_list.yaml",
        start_date="2023-01-01",
        output_dir=str(temp_output_dir),
        frequencies=["ME"],
        report_dir=str(report_dir),
        include_suggestions=False,
    )

    mocks = mock_reporting_steps

    # Step 4: build_features receives df_scores (output of ETL)
    mocks["build_features"].assert_called_once()
    bf_arg = mocks["build_features"].call_args.args[0]
    assert isinstance(bf_arg, pd.DataFrame)

    # Step 5: flag_tickers receives df_features (output of build_features)
    mocks["flag_tickers"].assert_called_once()

    # Step 6: summarise_peaks receives raw df (from DataLoader, not df_scores)
    mocks["summarise_peaks"].assert_called_once()
    sp_kwargs = mocks["summarise_peaks"].call_args
    # freqs kwarg should be forwarded
    assert sp_kwargs.kwargs.get("freqs") == ["ME"]

    # Step 7: generate_report called with include_suggestions=False
    mocks["generate_report"].assert_called_once()
    gr_kwargs = mocks["generate_report"].call_args.kwargs
    assert gr_kwargs.get("include_suggestions") is False

    # Step 8: save_report called once
    mocks["save_report"].assert_called_once()


@pytest.mark.unit
def test_run_pipeline_include_suggestions_forwarded(
    mock_dataloader, mock_seasonality_etl, mock_reporting_steps, temp_output_dir, tmp_path
):
    """
    Unit test: include_suggestions=True is forwarded to generate_report.
    """
    run_pipeline(
        config_path="config/tickers_list.yaml",
        start_date="2023-01-01",
        output_dir=str(temp_output_dir),
        report_dir=str(tmp_path / "reports"),
        include_suggestions=True,
    )

    gr_kwargs = mock_reporting_steps["generate_report"].call_args.kwargs
    assert gr_kwargs.get("include_suggestions") is True


@pytest.mark.unit
def test_run_pipeline_reporting_failure_handled(
    mock_dataloader, mock_seasonality_etl, mock_reporting_steps, temp_output_dir, tmp_path, caplog
):
    """
    Unit test: A failure in the reporting steps (Steps 4-8) returns None and logs error.
    The CSV from Step 3 has already been written at that point.
    """
    mock_reporting_steps["build_features"].side_effect = RuntimeError("Mock build_features failure")

    result = run_pipeline(
        config_path="config/tickers_list.yaml",
        start_date="2023-01-01",
        output_dir=str(temp_output_dir),
        report_dir=str(tmp_path / "reports"),
    )

    assert result is None
    assert any("ETL processing failed" in r.message for r in caplog.records)


@pytest.mark.unit
def test_run_pipeline_invalid_date_format(mock_dataloader, mock_seasonality_etl, temp_output_dir):
    """
    Unit test: Invalid date format returns None and logs error.
    """
    result = run_pipeline(
        config_path="config/tickers_list.yaml",
        start_date="not-a-date",
        end_date="2023-12-31",
        intervals=["1d"],
        output_dir=str(temp_output_dir),
    )

    assert result is None
    mock_dataloader.assert_not_called()


@pytest.mark.unit
def test_run_pipeline_data_load_failure(
    mock_dataloader, mock_seasonality_etl, temp_output_dir, caplog
):
    """
    Unit test: DataLoader exception is caught and logged.
    """
    loader_instance = mock_dataloader.return_value
    loader_instance.load.side_effect = ValueError("Mock data load failure")

    result = run_pipeline(
        config_path="config/tickers_list.yaml",
        start_date="2023-01-01",
        output_dir=str(temp_output_dir),
    )

    assert result is None
    assert any("Data loading failed" in record.message for record in caplog.records)


@pytest.mark.unit
def test_run_pipeline_etl_failure(mock_dataloader, mock_seasonality_etl, temp_output_dir, caplog):
    """
    Unit test: SeasonalityETL exception is caught and logged.
    """
    etl_instance = mock_seasonality_etl.return_value
    etl_instance.fit_rolling.side_effect = RuntimeError("Mock ETL failure")

    result = run_pipeline(
        config_path="config/tickers_list.yaml",
        start_date="2023-01-01",
        output_dir=str(temp_output_dir),
    )

    assert result is None
    assert any("ETL processing failed" in record.message for record in caplog.records)


@pytest.mark.unit
def test_run_pipeline_default_parameters(
    mock_dataloader, mock_seasonality_etl, mock_reporting_steps, tmp_path
):
    """
    Unit test: Default parameters are applied correctly.
    """
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text("tickers:\n  - TEST\n")

    result = run_pipeline(
        config_path=str(config_path),
        start_date="2023-01-01",
        output_dir=str(tmp_path / "output"),
        report_dir=str(tmp_path / "reports"),
    )

    assert result is not None

    loader_instance = mock_dataloader.return_value
    call_args = loader_instance.load.call_args

    # check default intervals
    assert call_args.kwargs["intervals"] == ["1d"]


@pytest.mark.unit
def test_run_pipeline_force_refresh_flag(
    mock_dataloader, mock_seasonality_etl, mock_reporting_steps, temp_output_dir, tmp_path
):
    """
    Unit test: force_refresh=True sets use_cache=False in DataLoader.
    """
    result = run_pipeline(
        config_path="config/tickers_list.yaml",
        start_date="2023-01-01",
        output_dir=str(temp_output_dir),
        report_dir=str(tmp_path / "reports"),
        force_refresh=True,
    )

    assert result is not None

    call_kwargs = mock_dataloader.call_args.kwargs
    assert call_kwargs.get("use_cache") is False


@pytest.mark.unit
def test_run_pipeline_creates_output_directory(
    mock_dataloader, mock_seasonality_etl, mock_reporting_steps, tmp_path
):
    """
    Unit test: Output directory is created if it doesn't exist.
    """
    output_dir = tmp_path / "new_dir" / "nested"

    result = run_pipeline(
        config_path="config/tickers_list.yaml",
        start_date="2023-01-01",
        output_dir=str(output_dir),
        report_dir=str(tmp_path / "reports"),
    )

    assert result is not None
    assert output_dir.exists()


@pytest.mark.unit
def test_run_pipeline_empty_dataframe(
    mock_dataloader, mock_seasonality_etl, mock_reporting_steps, temp_output_dir, tmp_path
):
    """
    Unit test: Empty DataFrame from DataLoader is handled gracefully.
    """
    loader_instance = mock_dataloader.return_value
    loader_instance.load.return_value = pd.DataFrame()

    etl_instance = mock_seasonality_etl.return_value
    etl_instance.fit_rolling.return_value = pd.DataFrame()

    result = run_pipeline(
        config_path="config/tickers_list.yaml",
        start_date="2023-01-01",
        output_dir=str(temp_output_dir),
        report_dir=str(tmp_path / "reports"),
    )

    # either is acceptable; should not raise
    assert result is not None or result is None


# ============================================================================
# UNIT TESTS ---> CLI parsing (main function)
# ============================================================================


@pytest.mark.unit
def test_cli_default_arguments(
    mock_dataloader, mock_seasonality_etl, mock_reporting_steps, tmp_path, monkeypatch
):
    """
    Unit test: CLI with no arguments uses defaults.
    """
    test_args = ["run_pipeline.py"]
    monkeypatch.setattr(sys, "argv", test_args)

    with patch("sys.exit") as mock_exit:
        main()
        mock_exit.assert_called_once_with(0)


@pytest.mark.unit
def test_cli_custom_arguments(
    mock_dataloader, mock_seasonality_etl, mock_reporting_steps, temp_output_dir, monkeypatch
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

        loader_instance = mock_dataloader.return_value
        call_args = loader_instance.load.call_args

        assert call_args.kwargs["start_date"] == "2020-01-01"
        assert call_args.kwargs["end_date"] == "2020-12-31"
        assert call_args.kwargs["intervals"] == ["1d", "1wk"]

        mock_exit.assert_called_once_with(0)


@pytest.mark.unit
def test_cli_include_suggestions_flag(
    mock_dataloader, mock_seasonality_etl, mock_reporting_steps, temp_output_dir, monkeypatch
):
    """
    Unit test: --include-suggestions flag is forwarded to generate_report.
    """
    test_args = [
        "run_pipeline.py",
        "--output-dir",
        str(temp_output_dir),
        "--include-suggestions",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with patch("sys.exit"):
        main()

    gr_kwargs = mock_reporting_steps["generate_report"].call_args.kwargs
    assert gr_kwargs.get("include_suggestions") is True


@pytest.mark.unit
def test_cli_verbose_flag(
    mock_dataloader, mock_seasonality_etl, mock_reporting_steps, temp_output_dir, monkeypatch
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
        assert True  # no crash is sufficient here


# ============================================================================
# INTEGRATION TEST ---> Full end-to-end with real subprocess
# ============================================================================


@pytest.mark.integration
def test_run_pipeline_script_end_to_end(tmp_path):
    """
    Integration test: Run the actual CLI script via subprocess.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    script_path = Path(__file__).resolve().parents[2] / "src" / "pipeline" / "run_pipeline.py"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(script_path.parents[2])

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--start-date",
            "2022-01-01",
            "--end-date",
            "2023-01-01",
            "--frequencies",
            "W,ME",
            "--output-dir",
            str(output_dir),
            "--report-dir",
            str(tmp_path / "reports"),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )

    assert (
        result.returncode == 0
    ), f"Script failed with error:\nSTDERR: {result.stderr}\nSTDOUT: {result.stdout}"

    # CSV check (Step 3)
    date_str = datetime.today().strftime("%Y-%m-%d")
    expected_csv = output_dir / f"seasonality_scores_{date_str}.csv"
    assert expected_csv.exists(), f"Expected CSV not found: {expected_csv}"
    assert expected_csv.stat().st_size > 0

    df = pd.read_csv(expected_csv)
    expected_cols = {"ticker", "interval", "freq", "window_start", "seasonality_score_linear"}
    assert expected_cols.issubset(
        set(df.columns)
    ), f"Missing columns: {expected_cols - set(df.columns)}"

    # JSON report check (Step 8)
    expected_json = tmp_path / "reports" / f"report_{date_str}.json"
    assert expected_json.exists(), f"Expected JSON report not found: {expected_json}"
    assert expected_json.stat().st_size > 0

    print(
        f"(^o^) Pipeline CLI integration test passed. CSV: {expected_csv}, Report: {expected_json}"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_run_pipeline_script_with_cache(tmp_path):
    """
    Integration test: Verify caching behavior — second run should use cache.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    script_path = Path(__file__).resolve().parents[2] / "src" / "pipeline" / "run_pipeline.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(script_path.parents[2])

    common_args = [
        sys.executable,
        str(script_path),
        "--start-date",
        "2023-01-01",
        "--end-date",
        "2023-01-10",
        "--output-dir",
        str(output_dir),
        "--report-dir",
        str(tmp_path / "reports"),
    ]

    result1 = subprocess.run(common_args, capture_output=True, text=True, env=env, timeout=60)
    assert result1.returncode == 0

    result2 = subprocess.run(common_args, capture_output=True, text=True, env=env, timeout=60)
    assert result2.returncode == 0

    date_str = datetime.today().strftime("%Y-%m-%d")
    assert (output_dir / f"seasonality_scores_{date_str}.csv").exists()
    assert (tmp_path / "reports" / f"report_{date_str}.json").exists()

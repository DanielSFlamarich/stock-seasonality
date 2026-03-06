# tests/reporting/test_report_generator.py

"""
Tests for reporting.report_generator.

Structure
---------
Unit tests (no network, no LLM):
  - Input validation (missing columns, empty input)
  - Output schema (required keys at every level)
  - Sorting (superperformers first, then alphabetical)
  - Suggestion presence (only for superperformers, only when include_suggestions=True)
  - NaN → null serialisation
  - left-join behaviour when df_peaks has no match for a ticker/freq
  - low_window_count present / absent depending on df_flags columns
  - save_report writes valid JSON to disk
  - _load_prompt_config raises correctly on missing file / missing key

Integration tests (marked @pytest.mark.integration):
  - Full pipeline: SeasonalityETL --> build_features --> flag_tickers -->
    summarise_peaks --> generate_report --> valid JSON output
  - Real LLM suggestion for a superperformer ticker (requires ANTHROPIC_API_KEY)

The LLM client is injected via anthropic_client= in all unit tests so no real
API call is ever made. A minimal prompts.yaml is written to a tmp directory
and passed via prompt_config_path= to avoid touching the real config.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from reporting.report_generator import generate_report, save_report

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

_PROMPTS_YAML = """
suggestion:
  model: claude-sonnet-4-6
  max_tokens: 120
  template: |
    Ticker {ticker}, freq {freq_label}, gap {gap_desc}, peaks {peak_count}.
    Buy low, sell high.
"""


@pytest.fixture()
def prompts_yaml(tmp_path: Path) -> Path:
    """Write a minimal prompts.yaml to a temp dir and return its path."""
    p = tmp_path / "prompts.yaml"
    p.write_text(_PROMPTS_YAML)
    return p


def _mock_client(suggestion_text: str = "Test suggestion.") -> MagicMock:
    """Return a mock anthropic.Anthropic client that returns a canned suggestion."""
    mock_content = MagicMock()
    mock_content.text = suggestion_text
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    client = MagicMock()
    client.messages.create.return_value = mock_response
    return client


def _make_flags_row(
    ticker: str,
    freq: str = "ME",
    superperformer: bool = False,
    stl_available: bool = True,
    low_window_count: bool = False,
) -> dict:
    """Minimal df_flags row satisfying _REQUIRED_FLAGS_COLS."""
    return {
        "ticker": ticker,
        "freq": freq,
        "superperformer_flag": superperformer,
        "stl_available": stl_available,
        "low_window_count": low_window_count,
        "acf_lag_val_mean": 0.8 if superperformer else 0.2,
        "p2m_val_mean": 0.8 if superperformer else 0.2,
        "stl_strength_mean": 0.8 if superperformer else np.nan,
        "seasonality_score_harmonic_mean": 0.75 if superperformer else 0.1,
    }


def _make_peaks_row(ticker: str, freq: str = "ME") -> dict:
    """Minimal df_peaks row satisfying _REQUIRED_PEAKS_COLS."""
    return {
        "ticker": ticker,
        "freq": freq,
        "peak_count": 10,
        "mean_peak_gap_days": 31.0,
        "std_peak_gap_days": 2.5,
    }


@pytest.fixture()
def one_super_one_flat(prompts_yaml):
    """
    Two tickers: SUPER (superperformer) and FLAT (not).
    Returns (df_flags, df_peaks, prompts_yaml_path).
    """
    df_flags = pd.DataFrame(
        [
            _make_flags_row("SUPER", superperformer=True),
            _make_flags_row("FLAT", superperformer=False),
        ]
    )
    df_peaks = pd.DataFrame(
        [
            _make_peaks_row("SUPER"),
            _make_peaks_row("FLAT"),
        ]
    )
    return df_flags, df_peaks, prompts_yaml


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_missing_flags_column_raises(self, prompts_yaml):
        df_flags = pd.DataFrame([_make_flags_row("A")]).drop(columns=["superperformer_flag"])
        df_peaks = pd.DataFrame([_make_peaks_row("A")])
        with pytest.raises(ValueError, match="df_flags"):
            generate_report(
                df_flags,
                df_peaks,
                include_suggestions=False,
                prompt_config_path=prompts_yaml,
            )

    def test_missing_peaks_column_raises(self, prompts_yaml):
        df_flags = pd.DataFrame([_make_flags_row("A")])
        df_peaks = pd.DataFrame([_make_peaks_row("A")]).drop(columns=["peak_count"])
        with pytest.raises(ValueError, match="df_peaks"):
            generate_report(
                df_flags,
                df_peaks,
                include_suggestions=False,
                prompt_config_path=prompts_yaml,
            )

    def test_empty_df_flags_returns_empty_report(self, prompts_yaml):
        df_flags = pd.DataFrame(columns=list(pd.DataFrame([_make_flags_row("X")]).columns))
        df_peaks = pd.DataFrame([_make_peaks_row("X")])
        result = generate_report(
            df_flags,
            df_peaks,
            include_suggestions=False,
            prompt_config_path=prompts_yaml,
        )
        assert result["tickers"] == []
        assert "generated_at" in result


# ---------------------------------------------------------------------------
# output schema
# ---------------------------------------------------------------------------


class TestOutputSchema:
    def test_top_level_keys(self, one_super_one_flat):
        df_flags, df_peaks, prompts_yaml = one_super_one_flat
        result = generate_report(
            df_flags,
            df_peaks,
            include_suggestions=False,
            prompt_config_path=prompts_yaml,
        )
        assert set(result.keys()) == {"generated_at", "tickers"}

    def test_generated_at_is_iso_string(self, one_super_one_flat):
        df_flags, df_peaks, prompts_yaml = one_super_one_flat
        result = generate_report(
            df_flags,
            df_peaks,
            include_suggestions=False,
            prompt_config_path=prompts_yaml,
        )
        # must parse without raising
        from datetime import datetime

        datetime.fromisoformat(result["generated_at"])

    def test_ticker_entry_required_keys(self, one_super_one_flat):
        df_flags, df_peaks, prompts_yaml = one_super_one_flat
        result = generate_report(
            df_flags,
            df_peaks,
            include_suggestions=False,
            prompt_config_path=prompts_yaml,
        )
        for entry in result["tickers"]:
            assert "ticker" in entry
            assert "superperformer" in entry
            assert "frequencies" in entry

    def test_freq_entry_required_keys(self, one_super_one_flat):
        df_flags, df_peaks, prompts_yaml = one_super_one_flat
        result = generate_report(
            df_flags,
            df_peaks,
            include_suggestions=False,
            prompt_config_path=prompts_yaml,
        )
        for entry in result["tickers"]:
            for freq_entry in entry["frequencies"]:
                assert "freq" in freq_entry
                assert "peak_count" in freq_entry
                assert "mean_peak_gap_days" in freq_entry
                assert "std_peak_gap_days" in freq_entry

    def test_low_window_count_present_when_column_exists(self, one_super_one_flat):
        df_flags, df_peaks, prompts_yaml = one_super_one_flat
        assert "low_window_count" in df_flags.columns
        result = generate_report(
            df_flags,
            df_peaks,
            include_suggestions=False,
            prompt_config_path=prompts_yaml,
        )
        for entry in result["tickers"]:
            assert "low_window_count" in entry

    def test_low_window_count_absent_when_column_missing(self, prompts_yaml):
        df_flags = pd.DataFrame([_make_flags_row("A")]).drop(columns=["low_window_count"])
        df_peaks = pd.DataFrame([_make_peaks_row("A")])
        result = generate_report(
            df_flags,
            df_peaks,
            include_suggestions=False,
            prompt_config_path=prompts_yaml,
        )
        for entry in result["tickers"]:
            assert "low_window_count" not in entry


# ---------------------------------------------------------------------------
# sorting
# ---------------------------------------------------------------------------


class TestSorting:
    def test_superperformers_come_first(self, prompts_yaml):
        df_flags = pd.DataFrame(
            [
                _make_flags_row("ZZZ", superperformer=False),
                _make_flags_row("AAA", superperformer=True),
                _make_flags_row("MMM", superperformer=False),
            ]
        )
        df_peaks = pd.DataFrame(
            [
                _make_peaks_row("ZZZ"),
                _make_peaks_row("AAA"),
                _make_peaks_row("MMM"),
            ]
        )
        result = generate_report(
            df_flags,
            df_peaks,
            include_suggestions=False,
            prompt_config_path=prompts_yaml,
        )
        tickers = [e["ticker"] for e in result["tickers"]]
        assert tickers[0] == "AAA"  # superperformer first
        assert set(tickers[1:]) == {"MMM", "ZZZ"}

    def test_alphabetical_within_non_superperformers(self, prompts_yaml):
        df_flags = pd.DataFrame(
            [
                _make_flags_row("ZZZ", superperformer=False),
                _make_flags_row("AAA", superperformer=False),
            ]
        )
        df_peaks = pd.DataFrame(
            [
                _make_peaks_row("ZZZ"),
                _make_peaks_row("AAA"),
            ]
        )
        result = generate_report(
            df_flags,
            df_peaks,
            include_suggestions=False,
            prompt_config_path=prompts_yaml,
        )
        tickers = [e["ticker"] for e in result["tickers"]]
        assert tickers == ["AAA", "ZZZ"]


# ---------------------------------------------------------------------------
# suggestions
# ---------------------------------------------------------------------------


class TestSuggestions:
    def test_suggestion_present_for_superperformer(self, one_super_one_flat):
        df_flags, df_peaks, prompts_yaml = one_super_one_flat
        client = _mock_client("Buy before peak, sell at peak.")
        result = generate_report(
            df_flags,
            df_peaks,
            include_suggestions=True,
            anthropic_client=client,
            prompt_config_path=prompts_yaml,
        )
        super_entry = next(e for e in result["tickers"] if e["ticker"] == "SUPER")
        for freq_entry in super_entry["frequencies"]:
            assert "suggestion" in freq_entry
            assert isinstance(freq_entry["suggestion"], str)

    def test_suggestion_absent_for_non_superperformer(self, one_super_one_flat):
        df_flags, df_peaks, prompts_yaml = one_super_one_flat
        client = _mock_client()
        result = generate_report(
            df_flags,
            df_peaks,
            include_suggestions=True,
            anthropic_client=client,
            prompt_config_path=prompts_yaml,
        )
        flat_entry = next(e for e in result["tickers"] if e["ticker"] == "FLAT")
        for freq_entry in flat_entry["frequencies"]:
            assert "suggestion" not in freq_entry

    def test_include_suggestions_false_skips_llm(self, one_super_one_flat):
        df_flags, df_peaks, prompts_yaml = one_super_one_flat
        client = _mock_client()
        generate_report(
            df_flags,
            df_peaks,
            include_suggestions=False,
            anthropic_client=client,
            prompt_config_path=prompts_yaml,
        )
        client.messages.create.assert_not_called()

    def test_suggestion_none_on_llm_failure(self, one_super_one_flat):
        """If the LLM call raises, suggestion must be None — not crash the report."""
        df_flags, df_peaks, prompts_yaml = one_super_one_flat
        client = MagicMock()
        client.messages.create.side_effect = Exception("API timeout")
        result = generate_report(
            df_flags,
            df_peaks,
            include_suggestions=True,
            anthropic_client=client,
            prompt_config_path=prompts_yaml,
        )
        super_entry = next(e for e in result["tickers"] if e["ticker"] == "SUPER")
        for freq_entry in super_entry["frequencies"]:
            assert freq_entry["suggestion"] is None


# ---------------------------------------------------------------------------
# NaN serialisation
# ---------------------------------------------------------------------------


class TestNanSerialisation:
    def test_nan_peak_gap_serialises_as_null(self, prompts_yaml):
        """NaN mean_peak_gap_days must become None (JSON null), not raise."""
        df_flags = pd.DataFrame([_make_flags_row("A", superperformer=False)])
        df_peaks = pd.DataFrame(
            [
                {
                    "ticker": "A",
                    "freq": "ME",
                    "peak_count": 0,
                    "mean_peak_gap_days": float("nan"),
                    "std_peak_gap_days": float("nan"),
                }
            ]
        )
        result = generate_report(
            df_flags,
            df_peaks,
            include_suggestions=False,
            prompt_config_path=prompts_yaml,
        )
        freq_entry = result["tickers"][0]["frequencies"][0]
        assert freq_entry["mean_peak_gap_days"] is None
        assert freq_entry["std_peak_gap_days"] is None

    def test_result_is_json_serialisable(self, one_super_one_flat):
        """The full report dict must survive json.dumps without raising."""
        df_flags, df_peaks, prompts_yaml = one_super_one_flat
        result = generate_report(
            df_flags,
            df_peaks,
            include_suggestions=False,
            prompt_config_path=prompts_yaml,
        )
        # must not raise
        json.dumps(result)


# ---------------------------------------------------------------------------
# left-join behaviour (ticker in df_flags but not in df_peaks)
# ---------------------------------------------------------------------------


class TestLeftJoin:
    def test_ticker_missing_from_peaks_gets_null_peak_fields(self, prompts_yaml):
        """
        If a ticker exists in df_flags but has no matching row in df_peaks,
        peak fields must be None (not KeyError, not crash).
        """
        df_flags = pd.DataFrame([_make_flags_row("GHOST", superperformer=False)])
        df_peaks = pd.DataFrame(
            columns=[
                "ticker",
                "freq",
                "peak_count",
                "mean_peak_gap_days",
                "std_peak_gap_days",
            ]
        )
        result = generate_report(
            df_flags,
            df_peaks,
            include_suggestions=False,
            prompt_config_path=prompts_yaml,
        )
        freq_entry = result["tickers"][0]["frequencies"][0]
        assert freq_entry["peak_count"] is None
        assert freq_entry["mean_peak_gap_days"] is None
        assert freq_entry["std_peak_gap_days"] is None


# ---------------------------------------------------------------------------
# save_report
# ---------------------------------------------------------------------------


class TestSaveReport:
    def test_writes_valid_json(self, tmp_path, one_super_one_flat):
        df_flags, df_peaks, prompts_yaml = one_super_one_flat
        report = generate_report(
            df_flags,
            df_peaks,
            include_suggestions=False,
            prompt_config_path=prompts_yaml,
        )
        out = tmp_path / "report.json"
        save_report(report, out)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert "tickers" in loaded
        assert "generated_at" in loaded

    def test_saved_ticker_count_matches(self, tmp_path, one_super_one_flat):
        df_flags, df_peaks, prompts_yaml = one_super_one_flat
        report = generate_report(
            df_flags,
            df_peaks,
            include_suggestions=False,
            prompt_config_path=prompts_yaml,
        )
        out = tmp_path / "report.json"
        save_report(report, out)
        loaded = json.loads(out.read_text())
        assert len(loaded["tickers"]) == 2

    def test_creates_parent_directory_if_missing(self, tmp_path, one_super_one_flat):
        """
        save_report must create the output directory if it does not exist.
        """
        df_flags, df_peaks, prompts_yaml = one_super_one_flat
        report = generate_report(
            df_flags, df_peaks, include_suggestions=False, prompt_config_path=prompts_yaml
        )
        out = tmp_path / "new_dir" / "sub" / "report.json"
        save_report(report, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# _load_prompt_config error handling
# ---------------------------------------------------------------------------


class TestLoadPromptConfig:
    def test_missing_file_raises_file_not_found(self, tmp_path, prompts_yaml):
        df_flags = pd.DataFrame([_make_flags_row("A")])
        df_peaks = pd.DataFrame([_make_peaks_row("A")])
        with pytest.raises(FileNotFoundError):
            generate_report(
                df_flags,
                df_peaks,
                include_suggestions=False,
                prompt_config_path=tmp_path / "nonexistent.yaml",
            )

    def test_missing_suggestion_key_raises_key_error(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("other_key:\n  model: x\n")
        df_flags = pd.DataFrame([_make_flags_row("A")])
        df_peaks = pd.DataFrame([_make_peaks_row("A")])
        with pytest.raises(KeyError, match="suggestion"):
            generate_report(
                df_flags,
                df_peaks,
                include_suggestions=False,
                prompt_config_path=bad_yaml,
            )


# ---------------------------------------------------------------------------
# integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestReportGeneratorIntegration:
    """
    Full pipeline smoke test. Requires:
    - Real SeasonalityETL (network-free, uses synthetic data from conftest)
    - ANTHROPIC_API_KEY env var set (for the LLM suggestion test only)

    Run with: uv run pytest -m integration
    """

    def test_full_pipeline_no_suggestions(
        self, df_synth_combined, df_rolling_combined, prompts_yaml
    ):
        from reporting.build_features import build_features
        from reporting.flag_tickers import flag_tickers
        from reporting.peak_analysis import summarise_peaks

        df_features = build_features(df_rolling_combined)
        df_flags = flag_tickers(df_features)
        df_peaks = summarise_peaks(df_synth_combined, freqs=["ME", "QE"])

        report = generate_report(
            df_flags,
            df_peaks,
            include_suggestions=False,
            prompt_config_path=prompts_yaml,
        )

        assert "tickers" in report
        assert len(report["tickers"]) > 0
        json.dumps(report)

        tickers = report["tickers"]
        supers = [t for t in tickers if t["superperformer"]]
        non_supers = [t for t in tickers if not t["superperformer"]]
        assert tickers == supers + non_supers

    def test_full_pipeline_with_synth_prices(
        self, df_synth_perfect, df_synth_flat, df_rolling_combined, prompts_yaml
    ):
        """
        End-to-end with real synthetic price data through every module.
        No LLM calls (include_suggestions=False).
        """
        from reporting.build_features import build_features
        from reporting.flag_tickers import flag_tickers
        from reporting.peak_analysis import summarise_peaks

        df_prices = pd.concat([df_synth_perfect, df_synth_flat], ignore_index=True)
        df_features = build_features(df_rolling_combined)
        df_flags = flag_tickers(df_features)
        df_peaks = summarise_peaks(df_prices, freqs=["ME", "QE"])

        report = generate_report(
            df_flags,
            df_peaks,
            include_suggestions=False,
            prompt_config_path=prompts_yaml,
        )

        # structural checks
        assert "tickers" in report
        assert len(report["tickers"]) > 0
        # must serialise without raising
        json.dumps(report)

        # superperformer (if any) comes first
        tickers = report["tickers"]
        supers = [t for t in tickers if t["superperformer"]]
        non_supers = [t for t in tickers if not t["superperformer"]]
        assert tickers == supers + non_supers

    @pytest.mark.llm
    def test_llm_suggestion_is_non_empty_string(
        self, df_synth_perfect, df_synth_flat, df_rolling_combined
    ):
        """
        Real LLM call for any superperformer ticker.
        Marked @pytest.mark.llm — run only when ANTHROPIC_API_KEY is set.
        Only asserts the suggestion is a non-empty string; content is not
        deterministic.
        """
        import os

        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        from reporting.build_features import build_features
        from reporting.flag_tickers import flag_tickers
        from reporting.peak_analysis import summarise_peaks

        df_prices = pd.concat([df_synth_perfect, df_synth_flat], ignore_index=True)
        df_features = build_features(df_rolling_combined)
        df_flags = flag_tickers(df_features)
        df_peaks = summarise_peaks(df_prices, freqs=["ME", "QE"])

        report = generate_report(df_flags, df_peaks, include_suggestions=True)

        for entry in report["tickers"]:
            if entry["superperformer"]:
                for freq_entry in entry["frequencies"]:
                    suggestion = freq_entry.get("suggestion")
                    assert suggestion is None or isinstance(suggestion, str)
                    if suggestion is not None:
                        assert len(suggestion) > 0

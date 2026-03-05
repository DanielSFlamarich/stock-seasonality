# src/reporting/report_generator.py

"""
Assembles the final seasonality report from the three upstream modules.

Inputs
------
df_flags  : output of flag_tickers.flag_tickers()
            One row per (ticker, freq). Must contain:
              ticker, freq, superperformer_flag, stl_available,
              low_window_count (optional; present unless min_reliable_windows=None),
              acf_lag_val_mean, p2m_val_mean, stl_strength_mean,
              seasonality_score_harmonic_mean.

df_peaks  : output of peak_analysis.summarise_peaks()
            One row per (ticker, freq). Must contain:
              ticker, freq, peak_count, mean_peak_gap_days, std_peak_gap_days.

Output
------
A JSON-serialisable dict with the following shape:

    {
      "generated_at": "<ISO-8601 timestamp>",
      "tickers": [
        {
          "ticker": "AAPL",
          "superperformer": true,
          "low_window_count": false,   // omitted if low_window_count column absent
          "frequencies": [
            {
              "freq": "ME",
              "peak_count": 11,
              "mean_peak_gap_days": 31.2,
              "std_peak_gap_days": 2.8,
              "suggestion": "..."      // only present for superperformers
            },
            ...
          ]
        },
        ...
      ]
    }

Design decisions
----------------
1. Suggestions are generated only for superperformers.
   Rationale: LLM calls have cost and latency. Non-superperformers lack the
   signal quality to support a reliable buy/sell narrative — generating one
   would be misleading.

2. Prompt config lives in config/prompts.yaml, not in source code.
   Model string, max_tokens, and the prompt template are all editable without
   touching Python. The template uses {ticker}, {freq_label}, {gap_desc}, and
   {peak_count} as substitution placeholders.

3. NaN values are serialised as null in JSON (via _sanitise).
   Python's json module cannot serialise float('nan'); null is the correct
   JSON equivalent and is handled gracefully by any UI consumer.

4. Tickers are sorted: superperformers first, then alphabetically within
   each group. This makes the top of the report immediately actionable.

5. LLM calls are made per (ticker, freq), not batched, to keep retry logic
   simple and error isolation clean. If a single call fails the suggestion
   field is set to null and a warning is logged — the rest of the report
   is unaffected.

6. Output is written to reports/report_json/ by convention (see save_report).
   The directory must exist; report_generator does not create it.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# prompt config
# ---------------------------------------------------------------------------

# default location of the prompt config file, relative to the project root.
# override by passing prompt_config_path to generate_report().
_DEFAULT_PROMPT_CONFIG = Path(__file__).parents[2] / "config" / "prompts.yaml"

# frequency labels used in the prompt template — plain English, no jargon.
_FREQ_LABEL: dict[str, str] = {
    "W": "weekly",
    "ME": "monthly",
    "QE": "quarterly",
    "YE": "yearly",
}

# ---------------------------------------------------------------------------
# required columns
# ---------------------------------------------------------------------------

_REQUIRED_FLAGS_COLS: list[str] = [
    "ticker",
    "freq",
    "superperformer_flag",
    "stl_available",
    "acf_lag_val_mean",
    "p2m_val_mean",
    "stl_strength_mean",
    "seasonality_score_harmonic_mean",
]

_REQUIRED_PEAKS_COLS: list[str] = [
    "ticker",
    "freq",
    "peak_count",
    "mean_peak_gap_days",
    "std_peak_gap_days",
]


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def generate_report(
    df_flags: pd.DataFrame,
    df_peaks: pd.DataFrame,
    *,
    include_suggestions: bool = True,
    anthropic_client: Optional[anthropic.Anthropic] = None,
    prompt_config_path: Optional[Path] = None,
) -> dict:
    """
    Build the full seasonality report as a JSON-serialisable dict.

    Parameters
    ----------
    df_flags : pd.DataFrame
        Output of flag_tickers.flag_tickers(). One row per (ticker, freq).

    df_peaks : pd.DataFrame
        Output of peak_analysis.summarise_peaks(). One row per (ticker, freq).

    include_suggestions : bool
        If True (default), call the LLM to generate plain-language buy/sell
        suggestions for superperformer tickers. Set False to skip LLM calls
        entirely (useful for testing or offline use).

    anthropic_client : anthropic.Anthropic, optional
        Injected client instance. If None, a default client is created (uses
        the ANTHROPIC_API_KEY environment variable). Explicit injection makes
        the function testable without real API calls.

    prompt_config_path : Path, optional
        Path to the prompts YAML file. Defaults to config/prompts.yaml
        relative to the project root. Override for testing or multi-env use.

    Returns
    -------
    dict
        JSON-serialisable report. See module docstring for shape.

    Raises
    ------
    ValueError
        If required columns are missing from df_flags or df_peaks.
    """
    _validate_input(df_flags, _REQUIRED_FLAGS_COLS, "df_flags")
    _validate_input(df_peaks, _REQUIRED_PEAKS_COLS, "df_peaks")

    if df_flags.empty:
        logger.warning("generate_report: df_flags is empty — returning empty report.")
        return {"generated_at": _now_iso(), "tickers": []}

    prompt_cfg = _load_prompt_config(prompt_config_path or _DEFAULT_PROMPT_CONFIG)
    merged = df_flags.merge(df_peaks, on=["ticker", "freq"], how="left")
    client = anthropic_client if anthropic_client is not None else anthropic.Anthropic()

    ticker_entries = []
    for ticker, group in merged.groupby("ticker"):
        entry = _build_ticker_entry(
            ticker=ticker,
            group=group,
            include_suggestions=include_suggestions,
            client=client,
            prompt_cfg=prompt_cfg,
        )
        ticker_entries.append(entry)

    # superperformers first, then alphabetical within each group
    ticker_entries.sort(key=lambda e: (not e["superperformer"], e["ticker"]))

    return {
        "generated_at": _now_iso(),
        "tickers": ticker_entries,
    }


def save_report(report: dict, path: str | Path) -> None:
    """
    Write the report dict to a JSON file.

    By convention, path should be under reports/report_json/.
    The parent directory must already exist.

    Parameters
    ----------
    report : dict
        Output of generate_report().
    path : str or Path
        Destination file path.
    """
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Report saved to {path}")


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------


def _load_prompt_config(path: Path) -> dict:
    """
    Load and return the 'suggestion' block from prompts.yaml.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the given path.
    KeyError
        If the 'suggestion' key is missing from the YAML.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt config not found at {path}. "
            f"Expected config/prompts.yaml relative to the project root."
        )
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if "suggestion" not in raw:
        raise KeyError(f"prompts.yaml at {path} is missing the 'suggestion' key.")
    return raw["suggestion"]


def _build_ticker_entry(
    ticker: str,
    group: pd.DataFrame,
    include_suggestions: bool,
    client: anthropic.Anthropic,
    prompt_cfg: dict,
) -> dict:
    """Build the dict entry for a single ticker."""
    first = group.iloc[0]
    is_super = bool(first["superperformer_flag"])

    entry: dict = {
        "ticker": ticker,
        "superperformer": is_super,
    }

    # low_window_count is optional (absent if min_reliable_windows=None)
    if "low_window_count" in group.columns:
        entry["low_window_count"] = bool(first["low_window_count"])

    entry["frequencies"] = [
        _build_freq_entry(
            row=row,
            is_super=is_super,
            include_suggestions=include_suggestions,
            client=client,
            prompt_cfg=prompt_cfg,
        )
        for _, row in group.iterrows()
    ]
    return entry


def _build_freq_entry(
    row: pd.Series,
    is_super: bool,
    include_suggestions: bool,
    client: anthropic.Anthropic,
    prompt_cfg: dict,
) -> dict:
    """Build the dict entry for one (ticker, freq) row."""
    freq_entry: dict = {
        "freq": row["freq"],
        "peak_count": _sanitise(row.get("peak_count")),
        "mean_peak_gap_days": _sanitise(row.get("mean_peak_gap_days")),
        "std_peak_gap_days": _sanitise(row.get("std_peak_gap_days")),
    }

    if is_super and include_suggestions:
        freq_entry["suggestion"] = _get_llm_suggestion(
            ticker=row["ticker"],
            freq=row["freq"],
            peak_count=row.get("peak_count"),
            mean_gap=row.get("mean_peak_gap_days"),
            std_gap=row.get("std_peak_gap_days"),
            client=client,
            prompt_cfg=prompt_cfg,
        )

    return freq_entry


def _get_llm_suggestion(
    ticker: str,
    freq: str,
    peak_count,
    mean_gap,
    std_gap,
    client: anthropic.Anthropic,
    prompt_cfg: dict,
) -> Optional[str]:
    """
    Call the LLM to generate a plain-language buy/sell suggestion.

    Model, max_tokens, and the prompt template are read from prompt_cfg
    (loaded from config/prompts.yaml). Returns None on any failure so a
    single bad call does not invalidate the rest of the report.
    """
    freq_label = _FREQ_LABEL.get(freq, freq)

    if mean_gap is not None and not _is_nan(mean_gap):
        gap_desc = f"roughly every {mean_gap:.0f} days"
        if std_gap is not None and not _is_nan(std_gap):
            gap_desc += f" (give or take {std_gap:.0f} days)"
    else:
        gap_desc = "at irregular intervals"

    prompt = prompt_cfg["template"].format(
        ticker=ticker,
        freq_label=freq_label,
        gap_desc=gap_desc,
        peak_count=peak_count,
    )

    try:
        response = client.messages.create(
            model=prompt_cfg["model"],
            max_tokens=prompt_cfg["max_tokens"],
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as exc:
        logger.warning(f"_get_llm_suggestion: LLM call failed for {ticker}/{freq}: {exc}")
        return None


def _validate_input(df: pd.DataFrame, required_cols: list[str], name: str) -> None:
    """Raise ValueError if any required columns are missing."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"generate_report: {name} is missing required columns: {missing}")


def _sanitise(value) -> Optional[float]:
    """
    Convert a value to a JSON-safe type.
    float('nan') and float('inf') → None (JSON null).
    """
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _is_nan(value) -> bool:
    """Safe nan check that works for both float and numpy scalar."""
    try:
        return math.isnan(value)
    except (TypeError, ValueError):
        return False


def _now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

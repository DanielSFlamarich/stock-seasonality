# src/pipeline/run_pipeline.py

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

# set up logging
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

# add repo root to sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline.data_loader import DataLoader  # noqa: E402
from src.pipeline.seasonality_etl import SeasonalityETL  # noqa: E402
from src.reporting.build_features import build_features  # noqa: E402
from src.reporting.flag_tickers import flag_tickers  # noqa: E402
from src.reporting.peak_analysis import summarise_peaks  # noqa: E402
from src.reporting.report_generator import generate_report, save_report  # noqa: E402


def run_pipeline(
    config_path: str = "config/tickers_list.yaml",
    start_date: str = "2019-01-01",
    end_date: Optional[str] = None,
    intervals: Optional[List[str]] = None,
    output_dir: str = "data/processed",
    frequencies: Optional[List[str]] = None,
    use_cache: bool = True,
    force_refresh: bool = False,
    report_dir: str = "reports/report_json",
    include_suggestions: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Runs the full seasonality analysis pipeline.

    Parameters:
    ----------
    config_path : str
        Path to YAML file with tickers list
    start_date : str
        ISO date string (YYYY-MM-DD)
    end_date : str, optional
        ISO date string or None (defaults to today)
    intervals : List[str]
        Granularity for historical data (default: ['1d'])
    output_dir : str
        Directory for output CSV
    frequencies : List[str]
        Seasonality window frequencies (default: ['W', 'ME', 'QE', 'YE'])
    use_cache : bool
        Whether to use cached data from DataLoader
    force_refresh : bool
        Force re-download even if cache exists
    report_dir : str
        Directory for the JSON report output (default: reports/report_json)
    include_suggestions : bool
        If True, call the LLM to generate buy/sell suggestions for superperformers.
        Defaults to False to avoid unintentional API spend.

    Returns:
    -------
    pd.DataFrame or None
        Computed seasonality scores (df_rolling), or None on failure.
        The JSON report is written to report_dir as a side effect.
    """
    if intervals is None:
        intervals = ["1d"]
    if frequencies is None:
        frequencies = ["W", "ME", "QE", "YE"]

    # validate date format
    try:
        datetime.fromisoformat(start_date)
        if end_date:
            datetime.fromisoformat(end_date)
    except ValueError as e:
        logging.error(f"Invalid date format: {e}. Expected YYYY-MM-DD")
        return None

    # STEP 1: load data
    # downloads historical OHLCV data from yfinance for configured
    # tickers and dates (or loads from cache)
    # uses: Dataloader (internal), datetime, logging
    try:
        logging.info("=" * 60)
        logging.info("STEP 1: Loading data...")
        logging.info(f"  Config: {config_path}")
        logging.info(f"  Date range: {start_date} to {end_date or 'today'}")
        logging.info(f"  Intervals: {', '.join(intervals)}")
        logging.info(f"  Use cache: {use_cache and not force_refresh}")

        loader = DataLoader(
            config_path=config_path,
            use_cache=(use_cache and not force_refresh),
            verbose=True,
        )
        df = loader.load(start_date=start_date, end_date=end_date, intervals=intervals)

        logging.info(f"(^o^) Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
        logging.info(f"  Tickers: {df['ticker'].nunique()}")
        logging.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    except Exception as e:
        logging.error(f"(>_<) Data loading failed: {e}")
        return None

    # STEP 2: Compute seasonality metrics
    # calculates ACF, P2M, and STL metrics over rolling windows, then
    # combines them into meta-scores
    # uses: SeasonalityETL (internal), pandas
    #
    # STEP 3: Save df_rolling CSV
    # STEP 4: build_features  — aggregate df_rolling → one row per (ticker, freq)
    # STEP 5: flag_tickers    — attach superperformer flags and percentile ranks
    # STEP 6: peak_analysis   — detect peaks in raw price series (uses df from Step 1)
    # STEP 7: report_generator — merge flags + peaks into JSON-serialisable report
    # STEP 8: save_report     — write report to disk
    try:
        logging.info("=" * 60)
        logging.info("STEP 2: Computing seasonality metrics...")
        logging.info(f"  Frequencies: {', '.join(frequencies)}")

        etl = SeasonalityETL()
        df_scores = etl.fit_rolling(df, frequencies=frequencies)

        logging.info(f"(^o^) Metrics computed: {df_scores.shape[0]:,} windows")
        logging.info(f"  Columns: {', '.join(df_scores.columns)}")

        # show summary stats
        if not df_scores.empty:
            mean_linear = df_scores["seasonality_score_linear"].mean()
            mean_geom = df_scores["seasonality_score_geom"].mean()
            mean_harmonic = df_scores["seasonality_score_harmonic"].mean()

            logging.info(
                f"  Mean scores: Linear={mean_linear:.3f}, "
                f"Geom={mean_geom:.3f}, Harmonic={mean_harmonic:.3f}"
            )

        # STEP 3: Save df_rolling results
        # writes the scored DataFrame to a timestamped CSV file in the output directory
        # Uses: Path-pathlib, datetime, logging
        logging.info("=" * 60)
        logging.info("STEP 3: Saving df_rolling results...")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        date_str = datetime.today().strftime("%Y-%m-%d")
        filename = output_path / f"seasonality_scores_{date_str}.csv"
        df_scores.to_csv(filename, index=False)

        logging.info(f"(^o^) Results saved to: {filename}")
        logging.info(f"  File size: {filename.stat().st_size / 1024:.1f} KB")

        # STEP 4: Build per-(ticker, freq) feature summary
        # aggregates df_rolling into _mean/_std/_latest/_trend columns
        # uses: build_features (internal)
        logging.info("=" * 60)
        logging.info("STEP 4: Building features...")

        df_features = build_features(df_scores)

        logging.info(f"(^o^) Features built: {df_features.shape[0]} (ticker, freq) groups")

        # STEP 5: Flag tickers
        # attaches superperformer_flag, stl_available, low_window_count,
        # and universe-relative percentile ranks
        # uses: flag_tickers (internal)
        logging.info("=" * 60)
        logging.info("STEP 5: Flagging tickers...")

        df_flags = flag_tickers(df_features)

        n_super = int(df_flags["superperformer_flag"].sum())
        logging.info(
            f"(^o^) Tickers flagged: {n_super} superperformer(s) "
            f"out of {df_flags['ticker'].nunique()} total"
        )

        # STEP 6: Peak analysis on raw price series
        # detects price peaks and computes inter-peak gap statistics
        # deliberately uses df (raw OHLCV from Step 1), not df_scores
        # uses: summarise_peaks (internal)
        logging.info("=" * 60)
        logging.info("STEP 6: Peak analysis...")

        df_peaks = summarise_peaks(df, freqs=frequencies)

        logging.info(f"(^o^) Peak analysis done: {df_peaks.shape[0]} (ticker, freq) rows")

        # STEP 7: Generate report
        # merges df_flags + df_peaks into a JSON-serialisable report dict
        # LLM suggestions are only generated when include_suggestions=True
        # uses: generate_report (internal)
        logging.info("=" * 60)
        logging.info("STEP 7: Generating report...")
        logging.info(f"  LLM suggestions: {'enabled' if include_suggestions else 'disabled'}")

        report = generate_report(
            df_flags,
            df_peaks,
            include_suggestions=include_suggestions,
        )

        n_tickers = len(report.get("tickers", []))
        logging.info(f"(^o^) Report assembled: {n_tickers} ticker(s)")

        # STEP 8: Save report to disk
        # writes timestamped JSON file to report_dir
        # uses: save_report (internal), Path-pathlib
        logging.info("=" * 60)
        logging.info("STEP 8: Saving report...")

        Path(report_dir).mkdir(parents=True, exist_ok=True)
        report_path = Path(report_dir) / f"report_{date_str}.json"
        save_report(report, report_path)

        logging.info(f"(^o^) Report saved to: {report_path}")
        # logging.info(f"  File size: {report_path.stat().st_size / 1024:.1f} KB")
        logging.info("=" * 60)
        logging.info("Pipeline completed successfully! 🎉")

        return df_scores

    except Exception as e:
        logging.error(f"(>_<) ETL processing failed: {e}")
        import traceback

        logging.error(traceback.format_exc())
        return None


def main():
    """
    CLI entrypoint with argparse.
    Converts command-line strings --> Python types
    Handles user input validation
    Only runs when script is executed directly
    """
    parser = argparse.ArgumentParser(
        description="Compute seasonality scores for financial tickers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # basic usage (uses config file)
  python src/pipeline/run_pipeline.py

  # custom date range
  python src/pipeline/run_pipeline.py --start-date 2020-01-01 --end-date 2023-12-31

  # multiple intervals and frequencies
  python src/pipeline/run_pipeline.py --intervals 1d,1wk --frequencies W,ME,QE

  # force refresh (ignore cache)
  python src/pipeline/run_pipeline.py --force-refresh

  # custom output location
  python src/pipeline/run_pipeline.py --output-dir results/my_analysis

  # generate LLM suggestions for superperformers
  python src/pipeline/run_pipeline.py --include-suggestions
        """,
    )

    parser.add_argument(
        "--config",
        default="config/tickers_list.yaml",
        help="Path to tickers YAML config file (default: config/tickers_list.yaml)",
    )
    parser.add_argument(
        "--start-date",
        default="2022-01-01",
        help="Start date in YYYY-MM-DD format (default: 2022-01-01)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--intervals",
        default="1d",
        help="Comma-separated intervals (e.g., '1d,1wk,1mo'). Default: 1d",
    )
    parser.add_argument(
        "--frequencies",
        default="W,ME,QE,YE",
        help="Comma-separated pandas frequency codes (e.g., 'W,ME,QE,YE'). " "Default: W,ME,QE,YE",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for results CSV (default: data/processed)",
    )
    parser.add_argument(
        "--report-dir",
        default="reports/report_json",
        help="Output directory for JSON report (default: reports/report_json)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache - always download fresh data",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-download even if cache exists (stronger than --no-cache)",
    )
    parser.add_argument(
        "--include-suggestions",
        action="store_true",
        help="Call the LLM to generate buy/sell suggestions for superperformers "
        "(requires ANTHROPIC_API_KEY). Off by default.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )

    args = parser.parse_args()

    # adjust logging level if verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Verbose mode enabled")

    # parse comma-separated lists
    intervals = [i.strip() for i in args.intervals.split(",")]
    frequencies = [f.strip() for f in args.frequencies.split(",")]

    # run pipeline
    result = run_pipeline(
        config_path=args.config,
        start_date=args.start_date,
        end_date=args.end_date,
        intervals=intervals,
        output_dir=args.output_dir,
        frequencies=frequencies,
        use_cache=(not args.no_cache),
        force_refresh=args.force_refresh,
        report_dir=args.report_dir,
        include_suggestions=args.include_suggestions,
    )

    # exit with appropriate code
    sys.exit(0 if result is not None else 1)


if __name__ == "__main__":
    main()

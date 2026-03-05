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

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.pipeline.data_loader import DataLoader  # noqa: E402
from src.pipeline.seasonality_etl import SeasonalityETL  # noqa: E402


def run_pipeline(
    config_path: str = "config/tickers_list.yaml",
    start_date: str = "2019-01-01",
    end_date: Optional[str] = None,
    intervals: Optional[List[str]] = None,
    output_dir: str = "data/processed",
    frequencies: Optional[List[str]] = None,
    use_cache: bool = True,
    force_refresh: bool = False,
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

    Returns:
    -------
    pd.DataFrame or None
        Computed seasonality scores, or None on failure
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
    try:
        logging.info("=" * 60)
        logging.info("STEP 2: Computing seasonality metrics...")
        logging.info(f"  Frequencies: {', '.join(frequencies)}")

        etl = SeasonalityETL()
        df_scores = etl.fit_rolling(df, frequencies=frequencies)

        logging.info(f"(^o^) Metrics computed: {df_scores.shape[0]:,} windows")
        logging.info(f"  Columns: {', '.join(df_scores.columns)}")

        # Show summary stats
        if not df_scores.empty:
            mean_linear = df_scores["seasonality_score_linear"].mean()
            mean_geom = df_scores["seasonality_score_geom"].mean()
            mean_harmonic = df_scores["seasonality_score_harmonic"].mean()

            logging.info(
                f"  Mean scores: Linear={mean_linear:.3f}, "
                f"Geom={mean_geom:.3f}, Harmonic={mean_harmonic:.3f}"
            )

        # STEP 3: Save results
        # writes the scored DataFrame to a timestamped CSV file in the output directory
        # Uses: Path-pathlib, datetime, logging
        logging.info("=" * 60)
        logging.info("STEP 3: Saving results...")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        date_str = datetime.today().strftime("%Y-%m-%d")
        filename = output_path / f"seasonality_scores_{date_str}.csv"
        df_scores.to_csv(filename, index=False)

        logging.info(f"(^o^) Results saved to: {filename}")
        logging.info(f"  File size: {filename.stat().st_size / 1024:.1f} KB")
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
    )

    # exit with appropriate code
    sys.exit(0 if result is not None else 1)


if __name__ == "__main__":
    main()

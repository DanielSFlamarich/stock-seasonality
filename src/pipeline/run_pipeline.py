# src/pipeline/run_pipeline.py

import sys
from datetime import datetime
from pathlib import Path

from src.pipeline.data_loader import DataLoader
from src.pipeline.seasonality_etl import SeasonalityETL

# repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    try:
        print("Loading data...")
        loader = DataLoader(
            config_path="config/tickers_list.yaml", use_cache=True, verbose=True
        )
        df = loader.load(start_date="2022-01-01", intervals=["1d", "1wk"])
        print(f"Data loaded: {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    try:
        print("Running ETL...")
        etl = SeasonalityETL()
        etl.fit(df)
        df_scores = etl.get_scores()
        print(f"Metrics computed for {df_scores.shape[0]} rows")

        output_dir = Path("data/.cache")
        output_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.today().strftime("%Y-%m-%d")
        filename = output_dir / f"seasonality_scores_{date_str}.csv"
        df_scores.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"ETL processing failed: {e}")


if __name__ == "__main__":
    main()

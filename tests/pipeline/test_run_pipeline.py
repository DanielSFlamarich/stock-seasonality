import os
import subprocess
from datetime import datetime
from pathlib import Path

"""
Run the main pipeline script and verify that it creates a non-empty output CSV file
under the data/.cache/ directory with today's date.
"""


def test_run_pipeline_script():

    cache_path = Path("data/.cache")
    cache_path.mkdir(parents=True, exist_ok=True)

    # clean up previous output files
    for f in cache_path.glob("seasonality_scores_*.csv"):
        f.unlink()

    # locate the run_pipeline script
    script_path = (
        Path(__file__).resolve().parents[2] / "src" / "pipeline" / "run_pipeline.py"
    )

    # set up PYTHONPATH so imports work
    env = os.environ.copy()
    env["PYTHONPATH"] = str(script_path.parents[2])

    # execute the pipeline
    result = subprocess.run(
        ["python", str(script_path)], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

    # check output CSV is created and not empty
    date_str = datetime.today().strftime("%Y-%m-%d")
    expected_file = cache_path / f"seasonality_scores_{date_str}.csv"
    assert expected_file.exists(), f"Expected file not found: {expected_file}"
    assert expected_file.stat().st_size > 0, "Output file is empty"

    print("Pipeline CLI test passed.")

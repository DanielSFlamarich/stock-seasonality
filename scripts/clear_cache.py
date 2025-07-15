# scripts/clear_cache.py

import shutil
from pathlib import Path

"""
Clears the local pipeline cache for downloaded stock pipeline.

This script is intended to run as part of the pre-commit hooks. It deletes
the contents of the pipeline/.cache directory to ensure that cached pipeline does not
persist between commits in notebooks or analysis scripts.

Only the .cache directory is ever removed as a safety measure.
"""

cache_dir = Path("pipeline/.cache")

if cache_dir.exists():
    if cache_dir.is_dir():
        if cache_dir.name == ".cache":  # ensure we're only ever deleting .cache
            shutil.rmtree(cache_dir)
            print("Cleared .cache directory.")
        else:
            print("Refusing to delete: not the .cache directory.")
    else:
        print("Path exists but is not a directory.")
else:
    print("No .cache directory to clear.")

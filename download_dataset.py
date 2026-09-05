from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "creditcard.csv"


def download_via_env_vars() -> bool:
    """Download using KAGGLE_USERNAME + KAGGLE_KEY env vars."""
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")

    if not username or not key:
        print("[download] KAGGLE_USERNAME or KAGGLE_KEY not set.")
        return False

    print(f"[download] Using env var credentials (user: {username})")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    import subprocess
    env = os.environ.copy()
    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle",
            "datasets", "download",
            "-d", "mlg-ulb/creditcardfraud",
            "-p", str(RAW_DIR),
            "--unzip",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"[download] Success: {RAW_FILE}")
        return True

    print(f"[download] kaggle CLI failed:\n{result.stderr}")
    return False


def download_via_opendatasets() -> bool:
    """Fallback: use opendatasets library (prompts for credentials)."""
    try:
        import opendatasets as od
    except ImportError:
        print("[download] opendatasets not installed. Run: pip install opendatasets")
        return False

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    od.download(
        "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        data_dir=str(RAW_DIR),
    )
    # opendatasets puts it in a subdirectory
    subdir = RAW_DIR / "creditcardfraud"
    if (subdir / "creditcard.csv").exists():
        import shutil
        shutil.move(str(subdir / "creditcard.csv"), str(RAW_FILE))
    return RAW_FILE.exists()


def main() -> None:
    if RAW_FILE.exists():
        print(f"[download] Already exists: {RAW_FILE}")
        return

    print("[download] Attempting download …")

    if download_via_env_vars():
        return

    print("[download] Trying opendatasets fallback …")
    if download_via_opendatasets():
        return

    print("\n[download] MANUAL STEPS REQUIRED:")
    print("  1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("  2. Download creditcard.csv")
    print(f"  3. Place it at: {RAW_FILE}")
    print("  4. Then run: python src/data_prep.py")
    sys.exit(1)


if __name__ == "__main__":
    main()

"""
Stage 4 Forward Data Fetcher & Ingestion Pipeline.
Downloads and verifies official monthly kline archives from data.binance.vision
for UNIV_10 (DOT, ATOM, LINK, UNI) starting from 2024-10-01.
"""
import hashlib
import io
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

RAW_STORAGE_DIR = PROJECT_ROOT / "data" / "raw" / "binance" / "spot" / "monthly" / "klines"
STAGE4_PARQUET_PATH = PROJECT_ROOT / "data" / "universes" / "stage4_univ10_q4_2024.parquet"


def calculate_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def fetch_and_verify_month(symbol: str, year_month: str) -> Path:
    """
    Downloads monthly zip and .CHECKSUM from data.binance.vision and verifies cryptographic hash.
    Returns path to the verified zip file.
    """
    base_url = "https://data.binance.vision/data/spot/monthly/klines"
    filename = f"{symbol}-1h-{year_month}.zip"
    zip_url = f"{base_url}/{symbol}/1h/{filename}"
    checksum_url = f"{zip_url}.CHECKSUM"

    target_dir = RAW_STORAGE_DIR / symbol / "1h"
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_path = target_dir / filename
    checksum_path = target_dir / f"{filename}.CHECKSUM"

    # Download .CHECKSUM
    if not checksum_path.exists():
        req = urllib.request.Request(checksum_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read()
            with open(checksum_path, "wb") as f:
                f.write(content)

    with open(checksum_path, "r", encoding="utf-8") as f:
        official_hash = f.read().strip().split()[0].lower()

    # Download .zip
    if not zip_path.exists():
        print(f"Downloading {filename} from data.binance.vision...")
        req = urllib.request.Request(zip_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
            with open(zip_path, "wb") as f:
                f.write(content)

    # Verify SHA-256
    local_hash = calculate_sha256(zip_path)
    if local_hash != official_hash:
        raise ValueError(
            f"Checksum mismatch for {filename}! Local: {local_hash}, Official: {official_hash}"
        )

    return zip_path


def parse_kline_csv_from_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_names = [n for n in z.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise FileNotFoundError(f"No CSV found in {zip_path}")
        with z.open(csv_names[0]) as f:
            # Read first line to check if header is present
            first_line = f.readline().decode("utf-8")
            f.seek(0)
            has_header = "open_time" in first_line.lower() or "open" in first_line.lower()
            if has_header:
                df = pd.read_csv(f)
            else:
                col_names = [
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "count",
                    "taker_buy_volume", "taker_buy_quote_volume", "ignore"
                ]
                df = pd.read_csv(f, names=col_names)

    # Standardize column names
    col_map = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=col_map)

    # Ensure required columns
    required = ["open_time", "open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column {col} in {zip_path}")

    # Convert types
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype(np.int64)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)

    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)
    return df[required]


def build_forward_dataset(
    symbols: List[str] = ["DOTUSDT", "ATOMUSDT", "LINKUSDT", "UNIUSDT"],
    months: List[str] = ["2024-10", "2024-11", "2024-12"],
    output_path: Path = STAGE4_PARQUET_PATH,
) -> pl.DataFrame:
    """
    Downloads, verifies, and compiles synchronized multi-asset parquet table for the given symbols and months.
    """
    print(f"Building Stage 4 forward dataset for {symbols} across months {months}...")
    symbol_dfs: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        month_dfs = []
        for ym in months:
            zip_p = fetch_and_verify_month(sym, ym)
            df_m = parse_kline_csv_from_zip(zip_p)
            month_dfs.append(df_m)
        full_sym_df = pd.concat(month_dfs, ignore_index=True)
        full_sym_df = full_sym_df.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)
        symbol_dfs[sym] = full_sym_df
        print(f"  {sym}: {len(full_sym_df)} hourly bars ({pd.to_datetime(full_sym_df['open_time'].iloc[0], unit='ms')} to {pd.to_datetime(full_sym_df['open_time'].iloc[-1], unit='ms')})")

    # Align timestamps
    common_times = set(symbol_dfs[symbols[0]]["open_time"])
    for sym in symbols[1:]:
        common_times = common_times.intersection(set(symbol_dfs[sym]["open_time"]))

    sorted_times = sorted(list(common_times))
    print(f"Synchronized common hourly bars across all {len(symbols)} assets: {len(sorted_times)}")

    aligned_dict: Dict[str, np.ndarray] = {"open_time": np.array(sorted_times, dtype=np.int64)}
    for sym in symbols:
        df_sym = symbol_dfs[sym].set_index("open_time").loc[sorted_times]
        for col in ["open", "high", "low", "close", "volume"]:
            aligned_dict[f"{sym}_{col}"] = df_sym[col].to_numpy(dtype=np.float64)

    df_aligned = pl.DataFrame(aligned_dict)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_aligned.write_parquet(output_path)
    print(f"Saved synchronized forward parquet table to: {output_path} ({len(df_aligned)} bars)")

    return df_aligned


if __name__ == "__main__":
    df_fwd = build_forward_dataset()
    print("Forward dataset check completed successfully.")

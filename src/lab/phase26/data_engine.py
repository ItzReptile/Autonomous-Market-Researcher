"""
Phase 26 Data Engine: Fresh 16-Universe Construction & Blinding.
Constructs 8 Real universes and 8 Matched Null universes (6 Block Bootstrap + 2 IAAFT Phase).
Enforces:
1. Air-gapped holdout: Q3 2024 (2024-07-01 to 2024-09-30) is strictly protected and untouched.
2. New asset coverage: Prominently incorporates ADAUSDT and DOTUSDT (never used in Phase 21-25).
3. Unsearched development windows: H1 2024 (2024-01-01 to 2024-06-30) and distinct 2023 multi-quarter cycles.
4. Independent block bootstrap per asset destroys cross-asset correlation while preserving volatility clustering and fat tails.
5. IAAFT phase randomization preserves exact fat tails and power spectral density while destroying non-linear dependencies.
6. Complete blinding into UNIV_01 .. UNIV_16 with uniform schema and metadata formatting.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple
import zipfile
import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "binance" / "spot" / "monthly" / "klines"
UNIVERSE_OUTPUT_DIR = PROJECT_ROOT / "data" / "universes" / "phase26"
CATALOG_PATH = PROJECT_ROOT / "data" / "universes" / "phase26_catalog.json"
BLINDING_KEY_PATH = PROJECT_ROOT / "data" / "audit" / "phase26_universe_blinding_key.json"


@dataclass
class UniverseMetadata:
    universe_id: str
    symbols: List[str]
    start_date: str
    end_date: str
    n_bars: int
    interval: str = "1h"
    description: str = "Multi-asset hourly cryptocurrency spot market slice."

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_raw_symbol_klines(symbol: str, start_date: str, end_date: str) -> pl.DataFrame:
    sym_dir = RAW_DATA_DIR / symbol / "1h"
    if not sym_dir.exists():
        raise FileNotFoundError(f"Directory not found for symbol {symbol}: {sym_dir}")

    start_ms = int(pl.Series([start_date]).str.to_datetime("%Y-%m-%d").dt.timestamp("ms")[0])
    end_ms = int(pl.Series([end_date]).str.to_datetime("%Y-%m-%d").dt.timestamp("ms")[0]) + 86_400_000 - 1

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades_count",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore"
    ]

    dfs = []
    for zip_path in sorted(sym_dir.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if not name.endswith(".csv"):
                    continue
                data = z.read(name)
                # Binance changed the archive format partway through the
                # history in two ways, BOTH of which silently produced zero
                # usable rows before this fix:
                #   * newer monthly CSVs carry a HEADER row;
                #   * newer files timestamp in MICROseconds, not milliseconds.
                # Detected per file so pre-existing archives parse bit
                # identically to before.
                has_header = data[:16].lstrip()[:9].lower() == b"open_time"
                df = pl.read_csv(data, has_header=has_header,
                                 new_columns=None if has_header else cols)
                if has_header:
                    df = df.rename({old_c: new_c for old_c, new_c
                                    in zip(df.columns, cols)})
                df = df.with_columns(pl.col("open_time").cast(pl.Int64, strict=False))
                # Binance switched klines from milliseconds to microseconds
                # mid-archive. Detecting the unit ONCE per file is wrong for the
                # transition month, which carries both units in one file --
                # confirmed on KLAYUSDT-1d-2024-10 (2 microsecond rows among 28
                # millisecond ones). Reading those rows as ms yields year 56796.
                # Normalise per row so a mixed file cannot slip through.
                df = df.with_columns(
                    pl.when(pl.col("open_time") > 1e14)
                    .then(pl.col("open_time") // 1000)
                    .otherwise(pl.col("open_time"))
                    .alias("open_time")
                )
                dfs.append(df)

    if not dfs:
        raise ValueError(f"No kline files found for {symbol} in {sym_dir}")

    full_df = pl.concat(dfs).sort("open_time").unique(subset=["open_time"])
    full_df = full_df.filter(
        (pl.col("open_time") >= start_ms) & (pl.col("open_time") <= end_ms)
    )

    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume"]
    full_df = full_df.with_columns([
        pl.col(c).cast(pl.Float64) for c in numeric_cols
    ])
    return full_df


def build_multi_asset_table(symbols: List[str], start_date: str, end_date: str) -> pl.DataFrame:
    # Governing data boundary, enforced in the loading path rather than merely
    # documented. See data/audit/GOVERNING_data_boundary_spec.json. Raises
    # EmbargoViolation for any window reaching 2024-04-01 or later, unless the
    # caller has explicitly opened a sanctioned holdout evaluation.
    from src.lab.audit.embargo import assert_window_allowed

    assert_window_allowed(start_date, end_date,
                          context=f"build_multi_asset_table({','.join(symbols)})")

    per_sym_dfs = {}
    common_times = None

    for sym in symbols:
        df = load_raw_symbol_klines(sym, start_date, end_date)
        times = set(df["open_time"].to_list())
        if common_times is None:
            common_times = times
        else:
            common_times = common_times.intersection(times)
        per_sym_dfs[sym] = df

    sorted_times = sorted(list(common_times))
    if len(sorted_times) < 240:
        raise ValueError(f"Too few aligned bars ({len(sorted_times)}) across symbols {symbols}")

    result_cols = {"open_time": sorted_times}
    for sym in symbols:
        df_filtered = per_sym_dfs[sym].filter(pl.col("open_time").is_in(sorted_times)).sort("open_time")
        for col_name in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume"]:
            result_cols[f"{sym}_{col_name}"] = df_filtered[col_name].to_list()

    return pl.DataFrame(result_cols)


def generate_independent_block_bootstrap(
    real_df: pl.DataFrame,
    symbols: List[str],
    mean_block_length: float = 48.0,
    seed: int = 42,
) -> pl.DataFrame:
    n = len(real_df)
    p = 1.0 / mean_block_length
    out_dict = {"open_time": real_df["open_time"].to_list()}

    for s_idx, sym in enumerate(symbols):
        rng = np.random.RandomState(seed + s_idx * 1013)
        indices = []
        cur_idx = rng.randint(0, n)
        while len(indices) < n:
            indices.append(cur_idx)
            if rng.uniform(0, 1) < p:
                cur_idx = rng.randint(0, n)
            else:
                cur_idx = (cur_idx + 1) % n

        sampled_idx = indices[:n]

        orig_close = real_df[f"{sym}_close"].to_numpy()
        orig_open = real_df[f"{sym}_open"].to_numpy()
        orig_high = real_df[f"{sym}_high"].to_numpy()
        orig_low = real_df[f"{sym}_low"].to_numpy()
        orig_vol = real_df[f"{sym}_volume"].to_numpy()
        orig_qvol = real_df[f"{sym}_quote_volume"].to_numpy()
        orig_tk = real_df[f"{sym}_taker_buy_volume"].to_numpy()

        orig_ret = np.diff(np.log(orig_close), prepend=np.log(orig_close[0]))
        sampled_ret = orig_ret[sampled_idx]
        sampled_ret[0] = 0.0

        new_close = orig_close[0] * np.exp(np.cumsum(sampled_ret))

        high_ratio = orig_high / np.maximum(1e-8, orig_close)
        low_ratio = orig_low / np.maximum(1e-8, orig_close)
        open_ratio = orig_open / np.maximum(1e-8, orig_close)

        out_dict[f"{sym}_close"] = list(new_close)
        out_dict[f"{sym}_open"] = list(new_close * open_ratio[sampled_idx])
        out_dict[f"{sym}_high"] = list(new_close * high_ratio[sampled_idx])
        out_dict[f"{sym}_low"] = list(new_close * low_ratio[sampled_idx])
        out_dict[f"{sym}_volume"] = list(orig_vol[sampled_idx])
        out_dict[f"{sym}_quote_volume"] = list(orig_qvol[sampled_idx])
        out_dict[f"{sym}_taker_buy_volume"] = list(orig_tk[sampled_idx])

    return pl.DataFrame(out_dict)


def generate_iaaft_phase_randomization(
    real_df: pl.DataFrame,
    symbols: List[str],
    max_iter: int = 100,
    seed: int = 42,
) -> pl.DataFrame:
    n = len(real_df)
    out_dict = {"open_time": real_df["open_time"].to_list()}

    for s_idx, sym in enumerate(symbols):
        rng = np.random.RandomState(seed + s_idx * 2027)
        orig_close = real_df[f"{sym}_close"].to_numpy()
        orig_open = real_df[f"{sym}_open"].to_numpy()
        orig_high = real_df[f"{sym}_high"].to_numpy()
        orig_low = real_df[f"{sym}_low"].to_numpy()
        orig_vol = real_df[f"{sym}_volume"].to_numpy()
        orig_qvol = real_df[f"{sym}_quote_volume"].to_numpy()
        orig_tk = real_df[f"{sym}_taker_buy_volume"].to_numpy()

        log_ret = np.diff(np.log(orig_close), prepend=np.log(orig_close[0]))

        target_fft = np.fft.rfft(log_ret)
        target_amp = np.abs(target_fft)
        sorted_ret = np.sort(log_ret)

        random_phases = rng.uniform(-np.pi, np.pi, size=len(target_fft))
        random_phases[0] = 0.0
        if n % 2 == 0:
            random_phases[-1] = 0.0

        s_fft = target_amp * np.exp(1j * random_phases)
        s = np.fft.irfft(s_fft, n=n)

        for _ in range(max_iter):
            ranks = np.argsort(np.argsort(s))
            s_rank = sorted_ret[ranks]
            curr_fft = np.fft.rfft(s_rank)
            curr_phases = np.angle(curr_fft)
            new_fft = target_amp * np.exp(1j * curr_phases)
            new_s = np.fft.irfft(new_fft, n=n)
            if np.max(np.abs(new_s - s)) < 1e-6:
                break
            s = new_s

        ranks = np.argsort(np.argsort(s))
        synth_ret = sorted_ret[ranks]
        synth_ret[0] = 0.0

        new_close = orig_close[0] * np.exp(np.cumsum(synth_ret))

        high_ratio = orig_high / np.maximum(1e-8, orig_close)
        low_ratio = orig_low / np.maximum(1e-8, orig_close)
        open_ratio = orig_open / np.maximum(1e-8, orig_close)

        vol_perm = rng.permutation(n)

        out_dict[f"{sym}_close"] = list(new_close)
        out_dict[f"{sym}_open"] = list(new_close * open_ratio)
        out_dict[f"{sym}_high"] = list(new_close * high_ratio)
        out_dict[f"{sym}_low"] = list(new_close * low_ratio)
        out_dict[f"{sym}_volume"] = list(orig_vol[vol_perm])
        out_dict[f"{sym}_quote_volume"] = list(orig_qvol[vol_perm])
        out_dict[f"{sym}_taker_buy_volume"] = list(orig_tk[vol_perm])

    return pl.DataFrame(out_dict)


REAL_UNIVERSE_SPECS_PHASE26 = [
    {
        "id": "R01_v2",
        "name": "Alt-L1 Leaders H1 2024",
        "symbols": ["ADAUSDT", "DOTUSDT", "AVAXUSDT", "SOLUSDT"],
        "start": "2024-01-01",
        "end": "2024-06-30",
    },
    {
        "id": "R02_v2",
        "name": "Interoperability & Governance 2023",
        "symbols": ["DOTUSDT", "ATOMUSDT", "LINKUSDT", "UNIUSDT"],
        "start": "2023-06-01",
        "end": "2023-12-31",
    },
    {
        "id": "R03_v2",
        "name": "PoS Pioneer Pair 2023",
        "symbols": ["ADAUSDT", "DOTUSDT"],
        "start": "2023-04-01",
        "end": "2023-12-31",
    },
    {
        "id": "R04_v2",
        "name": "High-Beta Alternate L1s H1 2024",
        "symbols": ["NEARUSDT", "AVAXUSDT", "DOTUSDT", "ADAUSDT", "MATICUSDT"],
        "start": "2024-01-01",
        "end": "2024-06-30",
    },
    {
        "id": "R05_v2",
        "name": "Payment & Retail Flow Basket 2023",
        "symbols": ["XRPUSDT", "DOGEUSDT", "ADAUSDT", "LTCUSDT"],
        "start": "2023-05-01",
        "end": "2023-12-31",
    },
    {
        "id": "R06_v2",
        "name": "Broad 6-Asset Cross-Section H1 2024",
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT", "BNBUSDT"],
        "start": "2024-01-01",
        "end": "2024-06-30",
    },
    {
        "id": "R07_v2",
        "name": "DeFi & Infrastructure Basket 2023-2024",
        "symbols": ["LINKUSDT", "UNIUSDT", "NEARUSDT", "ATOMUSDT"],
        "start": "2023-08-01",
        "end": "2024-03-31",
    },
    {
        "id": "R08_v2",
        "name": "Majors Macro Trend H1 2024",
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "start": "2024-01-01",
        "end": "2024-06-30",
    },
]


def build_and_blind_phase26_universes(seed: int = 20262601) -> Dict[str, Any]:
    UNIVERSE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BLINDING_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)

    real_data_map = {}
    null_data_map = {}

    print("Building 8 Real Universes for Phase 26 from Binance 1h klines...")
    for spec in REAL_UNIVERSE_SPECS_PHASE26:
        u_id = spec["id"]
        df = build_multi_asset_table(spec["symbols"], spec["start"], spec["end"])
        real_data_map[u_id] = {
            "df": df,
            "symbols": spec["symbols"],
            "start": spec["start"],
            "end": spec["end"],
            "name": spec["name"],
        }
        print(f"  Real {u_id}: {spec['name']} ({len(df)} bars aligned, assets: {spec['symbols']})")

    print("\nGenerating 8 Matched Null Universes for Phase 26 (6 Block Bootstrap + 2 IAAFT Phase)...")
    for idx, spec in enumerate(REAL_UNIVERSE_SPECS_PHASE26):
        u_id = spec["id"]
        real_df = real_data_map[u_id]["df"]
        syms = spec["symbols"]
        null_id = f"N{idx+1:02d}_v2"

        if idx < 6:
            null_df = generate_independent_block_bootstrap(real_df, syms, mean_block_length=48.0, seed=seed + idx)
            method = "STATIONARY_BLOCK_BOOTSTRAP"
        else:
            null_df = generate_iaaft_phase_randomization(real_df, syms, seed=seed + idx)
            method = "IAAFT_PHASE_RANDOMIZATION"

        null_data_map[null_id] = {
            "df": null_df,
            "symbols": syms,
            "start": spec["start"],
            "end": spec["end"],
            "method": method,
            "parent_real_id": u_id,
        }
        print(f"  Null {null_id}: Method={method} ({len(null_df)} bars)")

    all_keys = [("REAL", k) for k in real_data_map.keys()] + [("NULL", k) for k in null_data_map.keys()]
    rng = random.Random(seed)
    rng.shuffle(all_keys)

    blinding_key = {}
    public_catalog = []

    print("\nBlinding and saving 16 universes for Phase 26...")
    for b_idx, (u_type, raw_k) in enumerate(all_keys, start=1):
        blinded_id = f"UNIV_{b_idx:02d}"
        if u_type == "REAL":
            u_info = real_data_map[raw_k]
            df = u_info["df"]
            symbols = u_info["symbols"]
            start_date = u_info["start"]
            end_date = u_info["end"]
            blinding_key[blinded_id] = {
                "type": "REAL",
                "internal_id": raw_k,
                "name": u_info["name"],
                "symbols": symbols,
                "start": start_date,
                "end": end_date,
            }
        else:
            u_info = null_data_map[raw_k]
            df = u_info["df"]
            symbols = u_info["symbols"]
            start_date = u_info["start"]
            end_date = u_info["end"]
            blinding_key[blinded_id] = {
                "type": "NULL",
                "internal_id": raw_k,
                "null_method": u_info["method"],
                "parent_real": u_info["parent_real_id"],
                "symbols": symbols,
                "start": start_date,
                "end": end_date,
            }

        parquet_path = UNIVERSE_OUTPUT_DIR / f"{blinded_id}.parquet"
        df.write_parquet(parquet_path)

        catalog_entry = {
            "universe_id": blinded_id,
            "asset_count": len(symbols),
            "assets": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "n_bars": len(df),
            "frequency": "1h",
            "available_features": [
                "returns_lookback",
                "rolling_zscore",
                "volatility_ratio",
                "volume_imbalance",
                "cross_sectional_rank",
            ],
        }
        public_catalog.append(catalog_entry)
        print(f"  Saved {blinded_id} -> {parquet_path.name} ({len(df)} bars, {len(symbols)} assets)")

    with open(BLINDING_KEY_PATH, "w", encoding="utf-8") as f:
        json.dump(blinding_key, f, indent=2)

    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(public_catalog, f, indent=2)

    print(f"\nPhase 26 blinding key written to: {BLINDING_KEY_PATH}")
    print(f"Phase 26 public catalog written to: {CATALOG_PATH}")
    return {"blinding_key": blinding_key, "catalog": public_catalog}


if __name__ == "__main__":
    build_and_blind_phase26_universes()

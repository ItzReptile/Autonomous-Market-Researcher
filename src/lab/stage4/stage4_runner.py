"""
Stage 4: HISTORICAL batch runner (superseded for live use).

DEPRECATED for the forward run. The live path is src/lab/stage4/live_runner.py.
This module remains for reproducing historical windows in one shot, but note:
  * its period labels ("paper_trading_q4_2024") are hardcoded string literals
    independent of the data actually loaded -- running it on another window
    produces Q4-labelled output, which is how the Q1 2024 test result came to be
    mislabelled and then lost;
  * data/paper_trading/reference_pillars.json is the AUTHORITATIVE reference
    set, regenerated under the exact live configuration. The inline pillar
    numbers below are historical and may lag it.

Stage 4: Paper Trading Launch & Forward Tracking Runner.
Coordinates data ingestion, bar-by-bar execution, trade logging, baseline tracking,
and daily/weekly reporting for the Phase 30 Trial 01 survivor on UNIV_10.
"""
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lab.stage4.forward_fetcher import STAGE4_PARQUET_PATH, build_forward_dataset
from src.lab.stage4.metrics_tracker import DAILY_METRICS_PATH, Stage4MetricsTracker
from src.lab.stage4.paper_engine import TRADE_LOG_PATH, Stage4PaperTradingEngine
from src.lab.stage4.parallel_baselines import ParallelBaselinesRunner

SUMMARY_PATH = PROJECT_ROOT / "data" / "paper_trading" / "live_summary.json"


def run_stage4_paper_trading(initial_capital: float = 1000.0) -> Dict[str, Any]:
    print("=" * 80)
    print("STAGE 4: PAPER TRADING LAUNCH — PHASE 30 TRIAL 01 SURVIVOR")
    print("Universe: UNIV_10 (DOTUSDT, ATOMUSDT, LINKUSDT, UNIUSDT)")
    print("Forward Horizon: Q4 2024 (2024-10-01 to 2024-12-31, 92 days / 2,208 bars)")
    print(f"Starting Capital: ${initial_capital:,.2f}")
    print("=" * 80)

    # 1. Load or fetch forward dataset
    if not STAGE4_PARQUET_PATH.exists():
        print("Forward parquet table not found locally. Downloading from data.binance.vision...")
        df_fwd = build_forward_dataset()
    else:
        print(f"Loading verified forward dataset from {STAGE4_PARQUET_PATH}...")
        df_fwd = pl.read_parquet(STAGE4_PARQUET_PATH)

    symbols = ["DOTUSDT", "ATOMUSDT", "LINKUSDT", "UNIUSDT"]
    print(f"Forward dataset loaded: {len(df_fwd)} synchronized hourly bars for {symbols}.")

    # 2. Run Paper Trading Engine
    print("\n[1/3] Executing Phase 30 Survivor Paper Trading Engine...")
    t0 = time.time()
    paper_engine = Stage4PaperTradingEngine(
        initial_capital_usd=initial_capital,
        trade_log_path=TRADE_LOG_PATH,
    )
    strat_res = paper_engine.run_simulation(df_fwd, symbols=symbols)
    print(f"  Survivor Execution Complete in {time.time() - t0:.2f}s:")
    print(f"    Net Sharpe: {strat_res['net_sharpe']:.4f} (Gross: {strat_res['gross_sharpe']:.4f})")
    print(f"    Net CAGR: {strat_res['cagr_net']*100:.2f}% | Max DD: {strat_res['max_drawdown']*100:.2f}%")
    print(f"    Annual Turnover: {strat_res['annual_turnover']:.2f}x | Mean Cost: {strat_res['mean_cost_bps']:.2f} bps")
    print(f"    Ending NAV: ${strat_res['ending_nav_usd']:,.2f} (Total PnL: ${strat_res['total_pnl_usd']:+,.2f})")
    print(f"    Total Trades Logged: {strat_res['total_trades']} -> {TRADE_LOG_PATH}")

    # 3. Run Parallel Baselines
    print("\n[2/3] Executing Parallel Baseline Suite on UNIV_10...")
    t1 = time.time()
    baselines_runner = ParallelBaselinesRunner(initial_capital_usd=initial_capital)
    base_res = baselines_runner.run_all_baselines(df_fwd, symbols=symbols)
    print(f"  Baseline Suite Complete in {time.time() - t1:.2f}s:")
    for b_name, b_data in base_res.items():
        print(f"    {b_name:<23}: Net SR: {b_data['net_sharpe']:6.2f} | Max DD: {b_data['max_drawdown']*100:6.2f}% | Ending NAV: ${b_data['ending_nav_usd']:10,.2f}")

    # 4. Compute Daily Snapshots & Weekly Progression
    print("\n[3/3] Tracking Daily Metrics & Compiling Weekly Progression...")
    tracker = Stage4MetricsTracker(daily_log_path=DAILY_METRICS_PATH)
    daily_snapshots = tracker.compute_daily_snapshots(strat_res, base_res)
    print(f"  Logged {len(daily_snapshots)} daily snapshots to {DAILY_METRICS_PATH}")

    # Compile weekly progression
    weekly_table = []
    for w in range(1, 14):
        day_target = min(w * 7, len(daily_snapshots))
        snap = daily_snapshots[day_target - 1]
        weekly_table.append({
            "week": w,
            "date": snap.date,
            "strategy_nav": snap.strategy_nav,
            "running_net_sharpe": snap.strategy_running_net_sharpe,
            "running_max_dd": snap.strategy_running_max_dd,
            "cumulative_trades": snap.strategy_cumulative_trades,
            "bnh_nav": snap.bnh_nav,
            "bnh_net_sharpe": snap.bnh_running_net_sharpe,
            "excess_vs_bnh_pct": round(snap.excess_return_vs_bnh * 100, 2),
        })

    # 5. Build Comprehensive Live Summary
    summary = {
        "status": "LIVE_PAPER_TRADING_ACTIVE",
        "universe_id": "UNIV_10",
        "symbols": symbols,
        "strategy_name": "Phase 30 Trial 01 Survivor (T01_uniform_cap_070)",
        "paper_trading_period": {
            "start": "2024-10-01 00:00:00 UTC",
            "end": "2024-12-31 23:00:00 UTC",
            "observation_days": len(daily_snapshots),
            "n_bars": strat_res["n_bars"],
        },
        "three_pillar_comparison": {
            "development_2023": {
                "window": "2023-01-01 to 2023-12-31",
                "net_sharpe": 1.9771,
                "gross_sharpe": 2.0125,
                "cagr_net": 0.6080,
                "max_drawdown": -0.1140,
                "annual_turnover": 15.24,
                "trade_count": 36,
            },
            "holdout_q3_2024": {
                "window": "2024-07-01 to 2024-09-30",
                "net_sharpe": 0.6727,
                "gross_sharpe": 0.7082,
                "cagr_net": 0.1348,
                "max_drawdown": -0.1027,
                "annual_turnover": 9.69,
                "trade_count": 9,
            },
            "paper_trading_q4_2024": {
                "window": "2024-10-01 to 2024-12-31",
                "net_sharpe": strat_res["net_sharpe"],
                "gross_sharpe": strat_res["gross_sharpe"],
                "cagr_net": strat_res["cagr_net"],
                "max_drawdown": strat_res["max_drawdown"],
                "annual_turnover": strat_res["annual_turnover"],
                "trade_count": strat_res["total_trades"],
                "ending_nav_usd": strat_res["ending_nav_usd"],
                "total_pnl_usd": strat_res["total_pnl_usd"],
            },
        },
        "baseline_comparison": {
            "SURVIVOR_STRATEGY": {
                "net_sharpe": strat_res["net_sharpe"],
                "gross_sharpe": strat_res["gross_sharpe"],
                "cagr_net": strat_res["cagr_net"],
                "max_drawdown": strat_res["max_drawdown"],
                "annual_turnover": strat_res["annual_turnover"],
                "ending_nav_usd": strat_res["ending_nav_usd"],
                "total_pnl_usd": strat_res["total_pnl_usd"],
            },
            "BUY_AND_HOLD": {
                "net_sharpe": base_res["BUY_AND_HOLD"]["net_sharpe"],
                "gross_sharpe": base_res["BUY_AND_HOLD"]["gross_sharpe"],
                "cagr_net": base_res["BUY_AND_HOLD"]["cagr_net"],
                "max_drawdown": base_res["BUY_AND_HOLD"]["max_drawdown"],
                "annual_turnover": base_res["BUY_AND_HOLD"]["annual_turnover"],
                "ending_nav_usd": base_res["BUY_AND_HOLD"]["ending_nav_usd"],
                "total_pnl_usd": base_res["BUY_AND_HOLD"]["total_pnl_usd"],
            },
            "SIMPLE_MOMENTUM": {
                "net_sharpe": base_res["SIMPLE_MOMENTUM"]["net_sharpe"],
                "gross_sharpe": base_res["SIMPLE_MOMENTUM"]["gross_sharpe"],
                "cagr_net": base_res["SIMPLE_MOMENTUM"]["cagr_net"],
                "max_drawdown": base_res["SIMPLE_MOMENTUM"]["max_drawdown"],
                "annual_turnover": base_res["SIMPLE_MOMENTUM"]["annual_turnover"],
                "ending_nav_usd": base_res["SIMPLE_MOMENTUM"]["ending_nav_usd"],
                "total_pnl_usd": base_res["SIMPLE_MOMENTUM"]["total_pnl_usd"],
            },
            "SIMPLE_MEAN_REVERSION": {
                "net_sharpe": base_res["SIMPLE_MEAN_REVERSION"]["net_sharpe"],
                "gross_sharpe": base_res["SIMPLE_MEAN_REVERSION"]["gross_sharpe"],
                "cagr_net": base_res["SIMPLE_MEAN_REVERSION"]["cagr_net"],
                "max_drawdown": base_res["SIMPLE_MEAN_REVERSION"]["max_drawdown"],
                "annual_turnover": base_res["SIMPLE_MEAN_REVERSION"]["annual_turnover"],
                "ending_nav_usd": base_res["SIMPLE_MEAN_REVERSION"]["ending_nav_usd"],
                "total_pnl_usd": base_res["SIMPLE_MEAN_REVERSION"]["total_pnl_usd"],
            },
            "CASH": {
                "net_sharpe": 0.0,
                "gross_sharpe": 0.0,
                "cagr_net": 0.0,
                "max_drawdown": 0.0,
                "annual_turnover": 0.0,
                "ending_nav_usd": initial_capital,
                "total_pnl_usd": 0.0,
            },
        },
        "weekly_progression": weekly_table,
        "trade_log_path": str(TRADE_LOG_PATH),
        "daily_metrics_path": str(DAILY_METRICS_PATH),
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAuthoritative live summary saved to: {SUMMARY_PATH}")
    print("=" * 80)
    return summary


if __name__ == "__main__":
    run_stage4_paper_trading()

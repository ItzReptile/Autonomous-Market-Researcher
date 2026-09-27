"""
Stage 4 Metrics Tracker & Daily/Weekly Reporting.
Tracks daily running Net Sharpe, CAGR, Max Drawdown, and comparative alpha
vs. parallel baselines since paper trading inception.
"""
from dataclasses import asdict, dataclass
import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

DAILY_METRICS_PATH = PROJECT_ROOT / "data" / "paper_trading" / "daily_metrics.jsonl"


@dataclass
class DailySnapshot:
    day_index: int
    date: str
    strategy_nav: float
    strategy_running_net_sharpe: float
    strategy_running_gross_sharpe: float
    strategy_running_cagr: float
    strategy_running_max_dd: float
    strategy_running_turnover: float
    strategy_cumulative_trades: int
    strategy_cumulative_pnl_usd: float
    bnh_nav: float
    bnh_running_net_sharpe: float
    bnh_running_max_dd: float
    mom_nav: float
    mom_running_net_sharpe: float
    mom_running_max_dd: float
    mr_nav: float
    mr_running_net_sharpe: float
    mr_running_max_dd: float
    cash_nav: float
    excess_return_vs_bnh: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Stage4MetricsTracker:
    HOURS_PER_YEAR = 8760

    def __init__(self, daily_log_path: Path = DAILY_METRICS_PATH):
        self.daily_log_path = daily_log_path

    def compute_daily_snapshots(
        self,
        strategy_res: Dict[str, Any],
        baselines_res: Dict[str, Dict[str, Any]],
    ) -> List[DailySnapshot]:
        timestamps = strategy_res["timestamps"]
        strat_net_ret = strategy_res["net_returns"]
        strat_gross_ret = strategy_res["gross_returns"]
        strat_nav = strategy_res["nav_history"]
        strat_to = strategy_res["turnovers"]

        bnh_net_ret = baselines_res["BUY_AND_HOLD"]["net_returns"]
        bnh_nav = baselines_res["BUY_AND_HOLD"]["nav_history"]

        mom_net_ret = baselines_res["SIMPLE_MOMENTUM"]["net_returns"]
        mom_nav = baselines_res["SIMPLE_MOMENTUM"]["nav_history"]

        mr_net_ret = baselines_res["SIMPLE_MEAN_REVERSION"]["net_returns"]
        mr_nav = baselines_res["SIMPLE_MEAN_REVERSION"]["nav_history"]

        cash_nav = baselines_res["CASH"]["nav_history"]

        n_bars = len(strat_net_ret)
        hours_per_day = 24
        n_days = n_bars // hours_per_day

        snapshots: List[DailySnapshot] = []
        self.daily_log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(self.daily_log_path, "w", encoding="utf-8")

        trade_tickets = strategy_res.get("trade_tickets", [])

        for day in range(1, n_days + 1):
            bar_end = day * hours_per_day
            ts_end = timestamps[min(bar_end, len(timestamps) - 1)]
            date_str = datetime.datetime.fromtimestamp(ts_end / 1000.0, tz=datetime.timezone.utc).strftime("%Y-%m-%d")

            # Strategy running metrics up to day
            sub_net = np.array(strat_net_ret[:bar_end])
            sub_gross = np.array(strat_gross_ret[:bar_end])
            sub_to = np.array(strat_to[:bar_end])
            n_obs = len(sub_net)

            mu_net = float(np.mean(sub_net)) if n_obs > 0 else 0.0
            sd_net = float(np.std(sub_net)) + 1e-8 if n_obs > 0 else 1.0
            sr_net = float(np.sqrt(self.HOURS_PER_YEAR) * (mu_net / sd_net))

            mu_gross = float(np.mean(sub_gross)) if n_obs > 0 else 0.0
            sd_gross = float(np.std(sub_gross)) + 1e-8 if n_obs > 0 else 1.0
            sr_gross = float(np.sqrt(self.HOURS_PER_YEAR) * (mu_gross / sd_gross))

            cum_net = np.cumsum(sub_net)
            peak = np.maximum.accumulate(cum_net)
            dd = cum_net - peak
            max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

            ann_to = float(np.sum(sub_to) * (self.HOURS_PER_YEAR / max(1, n_obs)))
            cagr = float(mu_net * self.HOURS_PER_YEAR)
            cum_trades = sum(1 for t in trade_tickets if t.bar_index <= bar_end)

            cur_strat_nav = strat_nav[bar_end]
            cur_pnl_usd = cur_strat_nav - strategy_res["initial_capital_usd"]

            # Baseline metrics
            cur_bnh_nav = bnh_nav[bar_end]
            sub_bnh_net = np.array(bnh_net_ret[:bar_end])
            sr_bnh = float(np.sqrt(self.HOURS_PER_YEAR) * (np.mean(sub_bnh_net) / (np.std(sub_bnh_net) + 1e-8)))
            cum_bnh = np.cumsum(sub_bnh_net)
            dd_bnh = float(np.min(cum_bnh - np.maximum.accumulate(cum_bnh)))

            cur_mom_nav = mom_nav[bar_end]
            sub_mom_net = np.array(mom_net_ret[:bar_end])
            sr_mom = float(np.sqrt(self.HOURS_PER_YEAR) * (np.mean(sub_mom_net) / (np.std(sub_mom_net) + 1e-8)))
            cum_mom = np.cumsum(sub_mom_net)
            dd_mom = float(np.min(cum_mom - np.maximum.accumulate(cum_mom)))

            cur_mr_nav = mr_nav[bar_end]
            sub_mr_net = np.array(mr_net_ret[:bar_end])
            sr_mr = float(np.sqrt(self.HOURS_PER_YEAR) * (np.mean(sub_mr_net) / (np.std(sub_mr_net) + 1e-8)))
            cum_mr = np.cumsum(sub_mr_net)
            dd_mr = float(np.min(cum_mr - np.maximum.accumulate(cum_mr)))

            cur_cash_nav = cash_nav[bar_end]
            excess_bnh = (cur_strat_nav - cur_bnh_nav) / strategy_res["initial_capital_usd"]

            snap = DailySnapshot(
                day_index=day,
                date=date_str,
                strategy_nav=round(cur_strat_nav, 2),
                strategy_running_net_sharpe=round(sr_net, 4),
                strategy_running_gross_sharpe=round(sr_gross, 4),
                strategy_running_cagr=round(cagr, 4),
                strategy_running_max_dd=round(max_dd, 4),
                strategy_running_turnover=round(ann_to, 2),
                strategy_cumulative_trades=cum_trades,
                strategy_cumulative_pnl_usd=round(cur_pnl_usd, 2),
                bnh_nav=round(cur_bnh_nav, 2),
                bnh_running_net_sharpe=round(sr_bnh, 4),
                bnh_running_max_dd=round(dd_bnh, 4),
                mom_nav=round(cur_mom_nav, 2),
                mom_running_net_sharpe=round(sr_mom, 4),
                mom_running_max_dd=round(dd_mom, 4),
                mr_nav=round(cur_mr_nav, 2),
                mr_running_net_sharpe=round(sr_mr, 4),
                mr_running_max_dd=round(dd_mr, 4),
                cash_nav=round(cur_cash_nav, 2),
                excess_return_vs_bnh=round(excess_bnh, 4),
            )
            snapshots.append(snap)
            log_file.write(json.dumps(snap.to_dict()) + "\n")

        log_file.close()
        return snapshots

"""
Stage 4 Live Parallel Baseline Tracker.

The existing parallel_baselines.py is a vectorised BATCH backtester: it needs
the whole price matrix up front and produces one final answer. It cannot run
forward in real time, so it is unusable for live comparison.

This module is the streaming equivalent. Each baseline advances one bar at a
time on the same live feed, the same $1,000 basis and the same launch timestamp
as the survivor strategy, persisting its own NAV / Sharpe / drawdown atomically
so the comparison survives a crash exactly as the strategy state does.

Baselines (definitions preserved from parallel_baselines.py):
  1. BUY_AND_HOLD          equal 25% weights, entered at launch, held
  2. SIMPLE_MOMENTUM       480h lookback, rebalance every 168h to the leader
  3. SIMPLE_MEAN_REVERSION 480h z-score, long at z <= -2, flat at z >= 0
  4. CASH                  flat, zero return
"""
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np

from src.lab.stage4.execution_model import (
    BASE_FEE_RATE,
    BASE_SLIPPAGE_RATE,
    DEFAULT_FILTERS,
    SymbolFilters,
    quantize_order,
)

HOURS_PER_YEAR = 8760
MOM_LOOKBACK = 480
MOM_REBALANCE = 168
MR_LOOKBACK = 480
HISTORY_CAP = 520  # enough for the 480h signals plus slack

BASELINE_NAMES = ["BUY_AND_HOLD", "SIMPLE_MOMENTUM", "SIMPLE_MEAN_REVERSION", "CASH"]


@dataclass
class BaselineBook:
    """Mutable book for a single baseline."""
    name: str
    nav: float
    peak_nav: float
    max_drawdown: float = 0.0
    weights: Dict[str, float] = field(default_factory=dict)
    net_returns: List[float] = field(default_factory=list)
    total_trades: int = 0
    bars_since_rebalance: int = 0
    mom_leader: Optional[str] = None
    mr_active: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "nav": self.nav,
            "peak_nav": self.peak_nav,
            "max_drawdown": self.max_drawdown,
            "weights": self.weights,
            "net_returns": self.net_returns,
            "total_trades": self.total_trades,
            "bars_since_rebalance": self.bars_since_rebalance,
            "mom_leader": self.mom_leader,
            "mr_active": self.mr_active,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BaselineBook":
        return cls(
            name=d["name"],
            nav=d["nav"],
            peak_nav=d["peak_nav"],
            max_drawdown=d.get("max_drawdown", 0.0),
            weights=d.get("weights", {}),
            net_returns=d.get("net_returns", []),
            total_trades=d.get("total_trades", 0),
            bars_since_rebalance=d.get("bars_since_rebalance", 0),
            mom_leader=d.get("mom_leader"),
            mr_active=d.get("mr_active", {}),
        )

    def sharpe(self) -> float:
        if len(self.net_returns) < 2:
            return 0.0
        arr = np.asarray(self.net_returns, dtype=np.float64)
        sd = float(np.std(arr)) + 1e-12
        return float(math.sqrt(HOURS_PER_YEAR) * (float(np.mean(arr)) / sd))

    def cagr(self) -> float:
        if not self.net_returns:
            return 0.0
        return float(np.mean(self.net_returns) * HOURS_PER_YEAR)


class LiveBaselineTracker:
    """Streaming, crash-safe tracker for the four reference baselines."""

    def __init__(
        self,
        symbols: List[str],
        initial_capital_usd: float,
        state_path: Path,
        filters: Optional[Dict[str, SymbolFilters]] = None,
    ):
        self.symbols = list(symbols)
        self.initial_capital_usd = float(initial_capital_usd)
        self.state_path = Path(state_path)
        self.filters = filters if filters is not None else dict(DEFAULT_FILTERS)

        self.books: Dict[str, BaselineBook] = {}
        self.close_history: List[Dict[str, float]] = []
        self.anchor_timestamp_utc: Optional[int] = None
        self.last_processed_timestamp_utc: Optional[int] = None
        self.last_prices: Dict[str, float] = {}

        self._load()

    # ------------------------------------------------------------------
    # Persistence (same atomic discipline as the strategy state)
    # ------------------------------------------------------------------

    def _save(self) -> None:
        payload = {
            "symbols": self.symbols,
            "initial_capital_usd": self.initial_capital_usd,
            "anchor_timestamp_utc": self.anchor_timestamp_utc,
            "last_processed_timestamp_utc": self.last_processed_timestamp_utc,
            "last_prices": self.last_prices,
            "close_history": self.close_history[-HISTORY_CAP:],
            "books": {k: v.to_dict() for k, v in self.books.items()},
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=self.state_path.parent, delete=False, encoding="utf-8", suffix=".tmp"
        ) as tf:
            tmp_name = tf.name
            json.dump(payload, tf)
            tf.flush()
            os.fsync(tf.fileno())
        os.replace(tmp_name, self.state_path)

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.anchor_timestamp_utc = d.get("anchor_timestamp_utc")
            self.last_processed_timestamp_utc = d.get("last_processed_timestamp_utc")
            self.last_prices = d.get("last_prices", {})
            self.close_history = d.get("close_history", [])
            self.books = {k: BaselineBook.from_dict(v) for k, v in d.get("books", {}).items()}
        except Exception as err:
            print(f"[LiveBaselines] Warning: could not load baseline state: {err}")

    @property
    def initialized(self) -> bool:
        return bool(self.books) and self.anchor_timestamp_utc is not None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(
        self,
        anchor_timestamp_utc: int,
        prices: Dict[str, float],
        warmup_close_history: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        """
        Start every baseline at the same instant, on the same capital, as the
        survivor strategy.

        `warmup_close_history` seeds the 480h signal windows from real bars that
        precede launch, so momentum and mean-reversion are live from bar one
        rather than idling in cash for twenty days. Money still starts at launch;
        only the signal lookback uses pre-launch data, exactly as the survivor's
        24h factor does.
        """
        self.anchor_timestamp_utc = anchor_timestamp_utc
        self.last_processed_timestamp_utc = anchor_timestamp_utc
        self.last_prices = dict(prices)
        self.close_history = list(warmup_close_history or [])[-HISTORY_CAP:]
        if not self.close_history or self.close_history[-1] != prices:
            self.close_history.append(dict(prices))

        n = len(self.symbols)
        self.books = {
            "BUY_AND_HOLD": BaselineBook(
                name="BUY_AND_HOLD",
                nav=self.initial_capital_usd,
                peak_nav=self.initial_capital_usd,
                weights={s: 1.0 / n for s in self.symbols},
            ),
            "SIMPLE_MOMENTUM": BaselineBook(
                name="SIMPLE_MOMENTUM",
                nav=self.initial_capital_usd,
                peak_nav=self.initial_capital_usd,
                weights={s: 0.0 for s in self.symbols},
            ),
            "SIMPLE_MEAN_REVERSION": BaselineBook(
                name="SIMPLE_MEAN_REVERSION",
                nav=self.initial_capital_usd,
                peak_nav=self.initial_capital_usd,
                weights={s: 0.0 for s in self.symbols},
                mr_active={s: 0.0 for s in self.symbols},
            ),
            "CASH": BaselineBook(
                name="CASH",
                nav=self.initial_capital_usd,
                peak_nav=self.initial_capital_usd,
                weights={s: 0.0 for s in self.symbols},
            ),
        }

        # Buy & Hold pays its entry friction once, at launch.
        bnh = self.books["BUY_AND_HOLD"]
        entry_cost = 0.0
        for s in self.symbols:
            notional = bnh.nav * (1.0 / n)
            entry_cost += abs(notional) * (BASE_FEE_RATE + BASE_SLIPPAGE_RATE)
            bnh.total_trades += 1
        bnh.nav -= entry_cost
        bnh.peak_nav = max(bnh.peak_nav, bnh.nav)

        self._save()
        print(
            f"[LiveBaselines] Initialised 4 baselines at {anchor_timestamp_utc} "
            f"on ${self.initial_capital_usd:,.2f} "
            f"({len(self.close_history)} warm-up bars seeded)."
        )

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def _closes(self, symbol: str) -> np.ndarray:
        return np.asarray([h[symbol] for h in self.close_history if symbol in h], dtype=np.float64)

    def _momentum_target(self) -> Dict[str, float]:
        """480h lookback; concentrate in the strongest asset."""
        best, best_ret = None, -np.inf
        for s in self.symbols:
            c = self._closes(s)
            if len(c) <= MOM_LOOKBACK:
                return {}
            r = (c[-1] - c[-1 - MOM_LOOKBACK]) / max(1e-8, c[-1 - MOM_LOOKBACK])
            if r > best_ret:
                best, best_ret = s, r
        if best is None:
            return {}
        return {s: (1.0 if s == best else 0.0) for s in self.symbols}

    def _mean_reversion_target(self, book: BaselineBook) -> Dict[str, float]:
        """480h z-score; long below -2 sigma, flat at or above the mean."""
        n = len(self.symbols)
        for s in self.symbols:
            c = self._closes(s)
            if len(c) <= MR_LOOKBACK:
                return {}
            window = c[-1 - MR_LOOKBACK : -1]
            mu = float(np.mean(window))
            sd = float(np.std(window)) + 1e-8
            z = (c[-1] - mu) / sd
            if z <= -2.0:
                book.mr_active[s] = 1.0
            elif z >= 0.0:
                book.mr_active[s] = 0.0
        active = sum(book.mr_active.get(s, 0.0) for s in self.symbols)
        if active <= 0:
            return {s: 0.0 for s in self.symbols}
        return {s: book.mr_active.get(s, 0.0) / float(n) for s in self.symbols}

    # ------------------------------------------------------------------
    # Advance
    # ------------------------------------------------------------------

    def advance(self, bar_open_time: int, prices: Dict[str, float]) -> None:
        """Advance all baselines by exactly one closed bar."""
        if not self.initialized:
            return
        if (
            self.last_processed_timestamp_utc is not None
            and bar_open_time <= self.last_processed_timestamp_utc
        ):
            return  # idempotent: never double-count a bar

        rets = {
            s: (prices[s] - self.last_prices.get(s, prices[s])) / (self.last_prices.get(s, prices[s]) + 1e-8)
            for s in self.symbols
        }

        self.close_history.append(dict(prices))
        if len(self.close_history) > HISTORY_CAP:
            self.close_history = self.close_history[-HISTORY_CAP:]

        for name, book in self.books.items():
            if name == "CASH":
                book.net_returns.append(0.0)
                continue

            # 1. Mark to market on existing weights.
            gross = sum(book.weights.get(s, 0.0) * rets[s] for s in self.symbols)

            # 2. Determine this bar's target weights.
            target: Dict[str, float] = {}
            if name == "BUY_AND_HOLD":
                target = dict(book.weights)  # hold
            elif name == "SIMPLE_MOMENTUM":
                book.bars_since_rebalance += 1
                if book.bars_since_rebalance >= MOM_REBALANCE:
                    t = self._momentum_target()
                    if t:
                        target = t
                        book.bars_since_rebalance = 0
                    else:
                        target = dict(book.weights)
                else:
                    target = dict(book.weights)
            elif name == "SIMPLE_MEAN_REVERSION":
                t = self._mean_reversion_target(book)
                target = t if t else dict(book.weights)

            # 3. Cost the weight changes, with exchange quantization.
            cost_usd = 0.0
            nav_pre = book.nav * (1.0 + gross)
            for s in self.symbols:
                dw = target.get(s, 0.0) - book.weights.get(s, 0.0)
                if abs(dw) <= 1e-6:
                    continue
                requested = nav_pre * dw
                filt = self.filters.get(s)
                if filt is not None:
                    fill = quantize_order(requested, prices[s], filt)
                    if fill.rejected_min_notional:
                        target[s] = book.weights.get(s, 0.0)  # order did not go through
                        continue
                    filled = fill.filled_notional
                    target[s] = book.weights.get(s, 0.0) + filled / max(1e-12, nav_pre)
                else:
                    filled = requested
                cost_usd += abs(filled) * (BASE_FEE_RATE + BASE_SLIPPAGE_RATE)
                book.total_trades += 1

            cost_frac = cost_usd / max(1e-12, book.nav)
            net = gross - cost_frac

            book.nav = book.nav * (1.0 + gross) - cost_usd
            book.weights = target
            book.net_returns.append(net)
            book.peak_nav = max(book.peak_nav, book.nav)
            if book.peak_nav > 0:
                dd = (book.nav - book.peak_nav) / book.peak_nav
                book.max_drawdown = min(book.max_drawdown, dd)

        self.last_prices = dict(prices)
        self.last_processed_timestamp_utc = bar_open_time
        self._save()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Current live figures for every baseline, for the dashboard."""
        out: Dict[str, Dict[str, Any]] = {}
        for name in BASELINE_NAMES:
            book = self.books.get(name)
            if book is None:
                continue
            out[name] = {
                "net_sharpe": round(book.sharpe(), 4),
                "cagr_net": round(book.cagr(), 4),
                "max_drawdown": round(book.max_drawdown, 4),
                "ending_nav_usd": round(book.nav, 2),
                "total_pnl_usd": round(book.nav - self.initial_capital_usd, 2),
                "total_trades": book.total_trades,
                "bars_tracked": len(book.net_returns),
            }
        return out

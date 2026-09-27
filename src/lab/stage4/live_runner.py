"""
Stage 4 Hardened Live Paper Trading Daemon.
Executes the locked Phase 30 survivor (T01_uniform_cap_070) on UNIV_10 in real-time.

Design guarantees:
1. Transactional atomic state writes surviving abrupt SIGKILL / power loss.
2. Backfill-on-resume that replays each missed bar using the factor as it stood
   AT THAT BAR, reconstructed from historical klines -- never the present-time
   factor.
3. Friction identical to the historical engine (shared execution_model), plus
   exchange LOT_SIZE / MIN_NOTIONAL quantization.
4. Feed-health discrimination: a downtime gap and an unreachable API are
   different failures with different recovery behaviour, and are never
   conflated.
5. Every unclean resume writes a durable, timestamped gap event that the
   dashboard surfaces.
"""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lab.stage4.atomic_state import AtomicStateManager, LivePortfolioState
from src.lab.stage4.execution_model import (
    BASE_FEE_RATE,
    BASE_SLIPPAGE_RATE,
    FRICTION_BASELINE_WINDOW,
    SymbolFilters,
    causal_friction_ratios,
    fetch_exchange_filters,
    quantize_order,
    slippage_rate,
)
from src.lab.stage4.live_baselines import LiveBaselineTracker
from src.lab.stage4.live_feed import KlineBar, LivePriceFeed
from src.lab.stage4.paper_engine import TradeTicket

STATE_PATH = PROJECT_ROOT / "data" / "paper_trading" / "live_state.json"
TRADE_LOG_PATH = PROJECT_ROOT / "data" / "paper_trading" / "trade_log.jsonl"
DAILY_METRICS_PATH = PROJECT_ROOT / "data" / "paper_trading" / "daily_metrics.jsonl"
GAP_EVENTS_PATH = PROJECT_ROOT / "data" / "paper_trading" / "gap_events.jsonl"
HEALTH_PATH = PROJECT_ROOT / "data" / "paper_trading" / "feed_health.json"
NAV_HISTORY_PATH = PROJECT_ROOT / "data" / "paper_trading" / "nav_history.jsonl"

# How many historical bars to pull so both the 24h factor and the 720h causal
# friction baseline are fully warm.
HISTORY_LIMIT = 1000
FACTOR_WARMUP_BARS = 25


def _iso(ms: int) -> str:
    return datetime.datetime.fromtimestamp(ms / 1000.0, tz=datetime.timezone.utc).isoformat()


def _utc_str(ms: int) -> str:
    return datetime.datetime.fromtimestamp(ms / 1000.0, tz=datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def _now_ms() -> int:
    return int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)


class FeedUnreachable(RuntimeError):
    """The price feed could not be reached. Distinct from 'we were down'."""


class LivePaperTradingDaemon:
    SYMBOLS = ["DOTUSDT", "ATOMUSDT", "LINKUSDT", "UNIUSDT"]
    CAPITAL_WEIGHTS = {"ATOMUSDT": 0.15, "DOTUSDT": 0.15, "LINKUSDT": 0.35, "UNIUSDT": 0.35}
    POSITION_CAPS = {"ATOMUSDT": 0.70, "DOTUSDT": 0.70, "LINKUSDT": 0.70, "UNIUSDT": 0.70}
    BASE_FEE_RATE = BASE_FEE_RATE
    BASE_SLIPPAGE_RATE = BASE_SLIPPAGE_RATE
    HOURS_PER_YEAR = 8760
    INITIAL_CAPITAL_USD = 1000.0

    # Feed health thresholds
    STALE_FEED_HOURS = 3          # exchange reachable but bars not advancing
    MAX_CONSECUTIVE_FAILURES = 5  # before escalating to CRITICAL

    def __init__(
        self,
        state_path: Path = STATE_PATH,
        trade_log_path: Path = TRADE_LOG_PATH,
        daily_metrics_path: Path = DAILY_METRICS_PATH,
        gap_events_path: Path = GAP_EVENTS_PATH,
        health_path: Path = HEALTH_PATH,
        nav_history_path: Path = NAV_HISTORY_PATH,
        initial_capital_usd: Optional[float] = None,
        filters: Optional[Dict[str, SymbolFilters]] = None,
        enable_baselines: bool = True,
    ):
        self.state_manager = AtomicStateManager(state_path)
        self.feed = LivePriceFeed()
        self.trade_log_path = Path(trade_log_path)
        self.daily_metrics_path = Path(daily_metrics_path)
        self.gap_events_path = Path(gap_events_path)
        self.health_path = Path(health_path)
        self.nav_history_path = Path(nav_history_path)
        self.initial_capital_usd = (
            initial_capital_usd if initial_capital_usd is not None else self.INITIAL_CAPITAL_USD
        )
        self.filters = filters if filters is not None else fetch_exchange_filters()
        self.state: Optional[LivePortfolioState] = None
        self.consecutive_feed_failures = 0
        self.baselines: Optional[LiveBaselineTracker] = None
        if enable_baselines:
            self.baselines = LiveBaselineTracker(
                symbols=self.SYMBOLS,
                initial_capital_usd=self.initial_capital_usd,
                state_path=Path(state_path).parent / "live_baselines_state.json",
            )

    # ------------------------------------------------------------------
    # Durable event logging
    # ------------------------------------------------------------------

    def _log_event(self, event_type: str, **fields: Any) -> Dict[str, Any]:
        """Append a durable, timestamped operational event."""
        record = {
            "event_type": event_type,
            "logged_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **fields,
        }
        self.gap_events_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.gap_events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[LiveDaemon] [EVENT:{event_type}] {fields.get('detail', '')}")
        return record

    def _write_health(self, status: str, detail: str, **extra: Any) -> None:
        """Overwrite the current feed-health snapshot for the dashboard."""
        payload = {
            "status": status,
            "detail": detail,
            "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "consecutive_feed_failures": self.consecutive_feed_failures,
            **extra,
        }
        self.health_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.health_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp.replace(self.health_path)

    # ------------------------------------------------------------------
    # Feed access with explicit reachability semantics
    # ------------------------------------------------------------------

    def probe_feed(self) -> Tuple[bool, str]:
        """
        Determine whether the price feed is reachable RIGHT NOW.

        This is deliberately separate from gap detection: 'we were offline' and
        'the exchange is unreachable' demand different responses, and treating a
        network outage as a downtime gap would backfill against a feed that is
        not actually answering.
        """
        try:
            bars = self.feed.fetch_recent_klines(self.SYMBOLS[0], interval="1h", limit=3)
            if not bars:
                return False, "Feed returned an empty kline set"
            now = _now_ms()
            # Staleness must be measured against the newest CLOSED bar. The
            # final element is the bar currently being built, whose close_time
            # lies in the future and would report a negative age.
            closed = [b for b in bars if b.close_time <= now]
            if not closed:
                return False, "Feed returned no closed bars"
            age_h = (now - max(b.close_time for b in closed)) / 3_600_000.0
            if age_h > self.STALE_FEED_HOURS:
                return False, (
                    f"Feed reachable but STALE: newest bar closed {age_h:.1f}h ago "
                    f"(threshold {self.STALE_FEED_HOURS}h)"
                )
            return True, f"Feed healthy (newest bar {age_h:.2f}h old)"
        except Exception as err:
            return False, f"Feed unreachable: {type(err).__name__}: {err}"

    def _fetch_closed_bars(self, symbol: str, limit: int = HISTORY_LIMIT) -> List[KlineBar]:
        """
        Fetch recent klines and drop the still-forming final bar.

        The last element returned by the exchange is the bar currently being
        built. Including it would compute signals on incomplete data.
        """
        bars = self.feed.fetch_recent_klines(symbol, interval="1h", limit=limit)
        now = _now_ms()
        closed = [b for b in bars if b.close_time <= now]
        return sorted(closed, key=lambda b: b.open_time)

    def _fetch_aligned_history(self, limit: int = HISTORY_LIMIT) -> Tuple[List[int], Dict[str, List[KlineBar]]]:
        """Fetch closed bars for every symbol and align on common timestamps."""
        per_symbol: Dict[str, List[KlineBar]] = {}
        for sym in self.SYMBOLS:
            try:
                per_symbol[sym] = self._fetch_closed_bars(sym, limit=limit)
            except Exception as err:
                raise FeedUnreachable(f"Could not fetch history for {sym}: {err}") from err

        common = set(b.open_time for b in per_symbol[self.SYMBOLS[0]])
        for sym in self.SYMBOLS[1:]:
            common &= set(b.open_time for b in per_symbol[sym])
        times = sorted(common)

        aligned = {
            sym: [next(b for b in per_symbol[sym] if b.open_time == t) for t in times]
            for sym in self.SYMBOLS
        }
        return times, aligned

    # ------------------------------------------------------------------
    # Signal + friction reconstruction (historically correct)
    # ------------------------------------------------------------------

    def _build_matrices(self, aligned: Dict[str, List[KlineBar]]) -> Dict[str, np.ndarray]:
        close = np.column_stack([[b.close for b in aligned[s]] for s in self.SYMBOLS])
        high = np.column_stack([[b.high for b in aligned[s]] for s in self.SYMBOLS])
        low = np.column_stack([[b.low for b in aligned[s]] for s in self.SYMBOLS])
        vol = np.column_stack([[b.volume for b in aligned[s]] for s in self.SYMBOLS])
        ret = np.diff(close, axis=0) / np.maximum(1e-8, close[:-1])
        ret = np.vstack([np.zeros((1, close.shape[1])), ret])
        return {"close": close, "high": high, "low": low, "volume": vol, "ret": ret}

    def _compute_factor_matrix(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """
        Per-bar raw factor, computed identically to the historical engine:
            term_mom = (close_t - close_{t-24}) / close_{t-24}
            term_vol = mean(vol_{t-15..t}) / mean(vol_{t-23..t})
            S_t      = tanh(0.60*term_mom + 0.40*term_vol)

        Every value at index t uses only bars at or before t.
        """
        n_bars, n_assets = close.shape
        factors = np.zeros((n_bars, n_assets), dtype=np.float64)
        for t in range(n_bars):
            if t < 24:
                continue
            mom = (close[t] - close[t - 24]) / np.maximum(1e-8, close[t - 24])
            v16 = np.mean(volume[t - 15 : t + 1], axis=0)
            v24 = np.mean(volume[t - 23 : t + 1], axis=0)
            vr = v16 / (v24 + 1e-8)
            factors[t] = np.tanh(0.60 * mom + 0.40 * vr)
        return factors

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize_or_resume(self) -> LivePortfolioState:
        """Load existing state, or initialise fresh. Backfills any gap on resume."""
        reachable, detail = self.probe_feed()
        if not reachable:
            self.consecutive_feed_failures += 1
            self._log_event(
                "FEED_UNREACHABLE_ON_START",
                detail=detail,
                action="Startup deferred; state NOT advanced.",
            )
            self._write_health("FEED_UNREACHABLE", detail)
            raise FeedUnreachable(detail)

        self.consecutive_feed_failures = 0
        self.state = self.state_manager.load_state()

        if self.state is None:
            self._initialize_fresh()
        else:
            print(f"[LiveDaemon] Loaded state. Last processed: {self.state.last_processed_datetime_str}")
            print(f"[LiveDaemon] NAV ${self.state.portfolio_nav_usd:,.2f} | Positions: {self.state.positions}")
            self.backfill_if_needed()

        self._write_health("HEALTHY", detail, last_processed_utc=self.state.last_processed_datetime_str)
        return self.state

    def _initialize_fresh(self) -> None:
        """Seed a brand-new portfolio at the first fully-closed bar."""
        print("[LiveDaemon] No existing state. Initialising fresh forward portfolio...")
        times, aligned = self._fetch_aligned_history()
        if len(times) < FACTOR_WARMUP_BARS:
            raise RuntimeError(f"Insufficient warm-up history: {len(times)} bars")

        anchor_ms = times[-1]
        prices = {sym: aligned[sym][-1].close for sym in self.SYMBOLS}

        self.state = LivePortfolioState(
            strategy_name="Phase 30 Trial 01 Survivor (T01_uniform_cap_070)",
            universe_id="UNIV_10",
            symbols=self.SYMBOLS,
            last_processed_timestamp_utc=anchor_ms,
            last_processed_datetime_str=_utc_str(anchor_ms),
            portfolio_nav_usd=self.initial_capital_usd,
            cash_usd=self.initial_capital_usd,
            positions={sym: 0.0 for sym in self.SYMBOLS},
            position_caps=self.POSITION_CAPS,
            capital_weights=self.CAPITAL_WEIGHTS,
            total_trades_count=0,
            running_net_sharpe=0.0,
            running_max_drawdown=0.0,
            peak_nav_usd=self.initial_capital_usd,
            last_bar_close_prices=prices,
        )
        self.state_manager.save_state(self.state)

        if self.baselines is not None:
            # Seed the baselines' 480h signal windows from the same real
            # pre-launch bars, so they are live from bar one rather than idle.
            warmup_history = [
                {sym: aligned[sym][i].close for sym in self.SYMBOLS}
                for i in range(len(times))
            ]
            self.baselines.initialize(anchor_ms, prices, warmup_close_history=warmup_history)

        self._log_event(
            "LIVE_CLOCK_START",
            detail=(
                f"Live forward paper trading initialised at bar {_utc_str(anchor_ms)} "
                f"with ${self.initial_capital_usd:,.2f}"
            ),
            anchor_bar_utc=_iso(anchor_ms),
            initial_capital_usd=self.initial_capital_usd,
            symbols=self.SYMBOLS,
            position_caps=self.POSITION_CAPS,
        )
        print(f"[LiveDaemon] Live clock anchored at {self.state.last_processed_datetime_str}.")

    def backfill_if_needed(self) -> int:
        """
        Replay every completed bar missed since the last processed timestamp.

        Cause-agnostic by design: power loss, reboot and manual shutdown are all
        handled identically. What matters is that each replayed bar is evaluated
        with the factor AS OF THAT BAR.
        """
        if self.state is None:
            return 0

        last_ms = self.state.last_processed_timestamp_utc
        now_ms = _now_ms()
        # Measure the gap from the last processed bar's CLOSE, not its OPEN.
        # A bar with open time T only finishes at T+1h, so measuring from the
        # open reports an hour of lag that never existed and made routine
        # operation look like an unclean resume.
        last_close_ms = last_ms + 3_600_000
        gap_hours = max(0.0, (now_ms - last_close_ms) / 3_600_000.0)

        try:
            times, aligned = self._fetch_aligned_history()
        except FeedUnreachable as err:
            self.consecutive_feed_failures += 1
            self._log_event(
                "BACKFILL_ABORTED_FEED_UNREACHABLE",
                detail=str(err),
                gap_hours=round(gap_hours, 3),
                action="State left untouched; will retry next cycle.",
            )
            self._write_health("FEED_UNREACHABLE", str(err))
            raise

        # The strategy state and the baseline state are committed in sequence,
        # so a crash between the two can leave the baselines one bar behind.
        # Replay from whichever is furthest back; each component independently
        # ignores bars it has already applied, so nothing is double-counted.
        replay_from = last_ms
        if self.baselines is not None and self.baselines.last_processed_timestamp_utc is not None:
            baseline_last = self.baselines.last_processed_timestamp_utc
            if baseline_last < replay_from:
                self._log_event(
                    "BASELINE_LAG_DETECTED",
                    detail=(
                        f"Baselines trail the strategy by "
                        f"{(replay_from - baseline_last) / 3_600_000.0:.2f}h; "
                        f"replaying from the earlier point to resynchronise."
                    ),
                    strategy_last_utc=_iso(replay_from),
                    baselines_last_utc=_iso(baseline_last),
                )
                replay_from = baseline_last

        missing_idx = [i for i, t in enumerate(times) if t > replay_from]
        if not missing_idx:
            print(f"[LiveDaemon] Up to date (gap {gap_hours:.2f}h, no unprocessed bars).")
            return 0

        expected = int(gap_hours)

        # Exactly one newly closed bar is ordinary forward progress, not a
        # resume after downtime. Only a genuine shortfall -- more than one bar
        # owed -- is an unclean resume. Without this the event fires every hour
        # for the whole run and buries real downtime in thousands of false
        # positives, defeating the point of gap visibility.
        is_unclean_resume = len(missing_idx) > 1
        if is_unclean_resume:
            self._log_event(
                "UNCLEAN_RESUME_GAP_DETECTED",
                detail=(
                    f"Resumed after a {gap_hours:.2f} hour gap since the "
                    f"{self.state.last_processed_datetime_str} bar closed. "
                    f"{len(missing_idx)} missed bar(s) queued for backfill."
                ),
                cause_known=False,
                last_processed_utc=_iso(last_ms),
                gap_hours=round(gap_hours, 3),
                gap_bars_expected=expected,
                gap_bars_available=len(missing_idx),
            )
        else:
            print(
                f"[LiveDaemon] New closed bar available "
                f"({len(missing_idx)} bar, gap {gap_hours:.2f}h) - normal forward advance."
            )

        mats = self._build_matrices(aligned)
        factors = self._compute_factor_matrix(mats["close"], mats["volume"])
        vol_ratio, spread_ratio = causal_friction_ratios(
            mats["close"], mats["high"], mats["low"], mats["ret"]
        )

        processed = 0
        for i in missing_idx:
            if i < FACTOR_WARMUP_BARS:
                continue
            self._process_bar(
                bar_open_time=times[i],
                prices={s: float(mats["close"][i, j]) for j, s in enumerate(self.SYMBOLS)},
                factors={s: float(factors[i, j]) for j, s in enumerate(self.SYMBOLS)},
                vol_ratio_row=vol_ratio[i],
                spread_ratio_row=spread_ratio[i],
                is_backfill=True,
            )
            processed += 1

        # Paired with the event above: only report completion for a real
        # backfill, so an ordinary hourly advance produces no events at all.
        if is_unclean_resume:
            truncated = len(missing_idx) < expected - 1
            self._log_event(
                "BACKFILL_COMPLETE",
                detail=(
                    f"Backfilled {processed} bar(s); state resynchronised to "
                    f"{self.state.last_processed_datetime_str}."
                ),
                bars_backfilled=processed,
                gap_hours=round(gap_hours, 3),
                backfill_complete=not truncated,
                history_truncated=truncated,
                note=(
                    "Gap exceeded the available kline history window; the earliest "
                    "missed bars could not be recovered."
                    if truncated else ""
                ),
            )
        else:
            print(
                f"[LiveDaemon] Advanced {processed} bar(s) to "
                f"{self.state.last_processed_datetime_str}."
            )
        return processed

    # ------------------------------------------------------------------
    # Core bar processing
    # ------------------------------------------------------------------

    def _process_bar(
        self,
        bar_open_time: int,
        prices: Dict[str, float],
        factors: Dict[str, float],
        vol_ratio_row: np.ndarray,
        spread_ratio_row: np.ndarray,
        is_backfill: bool = False,
    ) -> None:
        """Apply one closed bar: mark to market, evaluate hysteresis, trade, persist."""
        if self.state is None:
            return

        # Idempotence: a bar the strategy has already applied must never be
        # applied twice, but the baselines may still owe this bar (see the
        # baseline-lag reconciliation in backfill_if_needed), so fall through
        # to advance them.
        if bar_open_time <= self.state.last_processed_timestamp_utc:
            if self.baselines is not None:
                self.baselines.advance(bar_open_time, prices)
            return

        dt_str = _utc_str(bar_open_time)
        nav = self.state.portfolio_nav_usd

        # 1. Mark existing positions to the new prices.
        for sym in self.SYMBOLS:
            p_old = self.state.last_bar_close_prices.get(sym, prices[sym])
            p_new = prices[sym]
            ret = (p_new - p_old) / (p_old + 1e-8)
            nav += nav * self.CAPITAL_WEIGHTS[sym] * self.state.positions.get(sym, 0.0) * ret

        # 2. Hysteresis transitions on the factor as of THIS bar.
        new_positions = dict(self.state.positions)
        tickets: List[TradeTicket] = []
        slip_row = slippage_rate(vol_ratio_row, spread_ratio_row)

        for j, sym in enumerate(self.SYMBOLS):
            raw_s = factors[sym]
            cap = self.POSITION_CAPS[sym]
            curr_p = new_positions.get(sym, 0.0)
            target_p = curr_p

            if abs(raw_s) > 0.50:
                target_p = float(np.sign(raw_s) * cap)
            elif abs(raw_s) < 0.20:
                target_p = 0.0

            if abs(target_p - curr_p) <= 1e-4:
                continue

            delta = target_p - curr_p
            weight = self.CAPITAL_WEIGHTS[sym]
            requested_notional = nav * weight * delta
            price = prices[sym]

            filt = self.filters.get(sym)
            if filt is not None:
                fill = quantize_order(requested_notional, price, filt)
                if fill.rejected_min_notional or abs(fill.filled_notional) < 1e-12:
                    self._log_event(
                        "ORDER_REJECTED_MIN_NOTIONAL",
                        detail=(
                            f"{sym} order of ${abs(requested_notional):.2f} rejected "
                            f"(min notional ${filt.min_notional:.2f}); position unchanged."
                        ),
                        symbol=sym,
                        bar_utc=_iso(bar_open_time),
                        requested_notional_usd=round(requested_notional, 4),
                    )
                    continue
                filled_notional = fill.filled_notional
                filled_qty = fill.filled_qty
                quant_err = fill.quantization_error_usd
            else:
                filled_notional = requested_notional
                filled_qty = requested_notional / max(1e-12, price)
                quant_err = 0.0

            achieved_delta = filled_notional / max(1e-12, nav * weight)
            achieved_p = curr_p + achieved_delta

            fee_rate = self.BASE_FEE_RATE
            slip_rate = float(slip_row[j])
            cost_usd = abs(filled_notional) * (fee_rate + slip_rate)

            nav_before = nav
            nav -= cost_usd

            tickets.append(
                TradeTicket(
                    trade_id=self.state.total_trades_count + len(tickets) + 1,
                    timestamp=_iso(bar_open_time),
                    bar_index=self.state.total_trades_count + len(tickets) + 1,
                    symbol=sym,
                    side="BUY" if achieved_delta > 0 else "SELL",
                    traded_weight_delta=round(achieved_delta, 6),
                    position_before=round(curr_p, 6),
                    position_after=round(achieved_p, 6),
                    fill_price=round(price, 6),
                    fee_bps=round(fee_rate * 10000, 2),
                    slippage_bps=round(slip_rate * 10000, 2),
                    total_cost_bps=round((fee_rate + slip_rate) * 10000, 2),
                    trade_cost_usd=round(cost_usd, 4),
                    portfolio_nav_before=round(nav_before, 2),
                    portfolio_nav_after=round(nav, 2),
                    requested_notional_usd=round(requested_notional, 4),
                    filled_notional_usd=round(filled_notional, 4),
                    filled_qty=round(filled_qty, 8),
                    quantization_error_usd=round(quant_err, 6),
                )
            )
            new_positions[sym] = achieved_p

        # 3. Append trade tickets.
        if tickets:
            self.trade_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.trade_log_path, "a", encoding="utf-8") as f:
                for t in tickets:
                    f.write(json.dumps(t.to_dict()) + "\n")
            tag = "BACKFILL" if is_backfill else "LIVE"
            print(f"[LiveDaemon] [{tag}] {len(tickets)} trade ticket(s) at {dt_str}.")

        # 4. Drawdown bookkeeping.
        peak_nav = max(self.state.peak_nav_usd, nav)
        running_dd = (nav - peak_nav) / peak_nav if peak_nav > 0 else 0.0

        # 5. Commit atomically.
        self.state = LivePortfolioState(
            strategy_name=self.state.strategy_name,
            universe_id=self.state.universe_id,
            symbols=self.SYMBOLS,
            last_processed_timestamp_utc=bar_open_time,
            last_processed_datetime_str=dt_str,
            portfolio_nav_usd=round(nav, 2),
            cash_usd=round(
                nav * (1.0 - sum(abs(new_positions[s]) * self.CAPITAL_WEIGHTS[s] for s in self.SYMBOLS)),
                2,
            ),
            positions=new_positions,
            position_caps=self.POSITION_CAPS,
            capital_weights=self.CAPITAL_WEIGHTS,
            total_trades_count=self.state.total_trades_count + len(tickets),
            running_net_sharpe=self.state.running_net_sharpe,
            running_max_drawdown=round(min(self.state.running_max_drawdown, running_dd), 4),
            peak_nav_usd=round(peak_nav, 2),
            last_bar_close_prices=prices,
        )
        self.state_manager.save_state(self.state)

        # 6. Advance the parallel baselines on the SAME bar, same feed, same basis.
        if self.baselines is not None:
            self.baselines.advance(bar_open_time, prices)

        # 7. Append the joint NAV series so the dashboard's chart is live and
        #    moving rather than replaying a frozen backtest.
        self._append_nav_history(bar_open_time, prices)

    def _append_nav_history(self, bar_open_time: int, prices: Dict[str, float]) -> None:
        """One row per processed bar: strategy NAV alongside every baseline NAV."""
        if self.state is None:
            return
        row: Dict[str, Any] = {
            "timestamp_utc": _iso(bar_open_time),
            "bar_open_time": bar_open_time,
            "strategy_nav": round(self.state.portfolio_nav_usd, 4),
            "strategy_max_dd": self.state.running_max_drawdown,
            "strategy_trades": self.state.total_trades_count,
            "prices": {k: round(v, 6) for k, v in prices.items()},
        }
        if self.baselines is not None:
            for name, snap in self.baselines.snapshot().items():
                row[f"{name}_nav"] = snap["ending_nav_usd"]
                row[f"{name}_max_dd"] = snap["max_drawdown"]
        self.nav_history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.nav_history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def run_live_cycle(self) -> int:
        """Process any newly closed bars. Returns the number of bars processed."""
        if self.state is None:
            self.initialize_or_resume()
            return 0

        reachable, detail = self.probe_feed()
        if not reachable:
            self.consecutive_feed_failures += 1
            severity = (
                "CRITICAL" if self.consecutive_feed_failures >= self.MAX_CONSECUTIVE_FAILURES else "DEGRADED"
            )
            self._log_event(
                "FEED_UNREACHABLE",
                detail=f"{detail} (consecutive failures: {self.consecutive_feed_failures})",
                severity=severity,
                action="No bars processed; state and positions left untouched.",
            )
            self._write_health(f"FEED_UNREACHABLE_{severity}", detail)
            return 0

        if self.consecutive_feed_failures > 0:
            self._log_event(
                "FEED_RECOVERED",
                detail=f"Feed reachable again after {self.consecutive_feed_failures} failed probe(s).",
                failures_cleared=self.consecutive_feed_failures,
            )
            self.consecutive_feed_failures = 0

        processed = self.backfill_if_needed()
        self._write_health(
            "HEALTHY", detail, last_processed_utc=self.state.last_processed_datetime_str
        )
        if processed == 0:
            print(f"[LiveDaemon] No new closed bar. Last: {self.state.last_processed_datetime_str}")
        return processed

    def _acquire_single_instance_lock(self) -> bool:
        """
        Ensure only one daemon writes the live record.

        Auto-start fires at logon, so a manually started daemon plus a logon
        could otherwise leave two processes appending to the same trade log and
        state file. Per-bar idempotence limits the damage, but two writers
        racing on the same files is not a condition worth relying on for 90
        days.
        """
        lock_path = self.state_manager.state_path.parent / "daemon.lock"
        if lock_path.exists():
            try:
                existing = int(lock_path.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                existing = -1
            if existing > 0 and existing != os.getpid():
                alive = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {existing}", "/FO", "CSV", "/NH"],
                    capture_output=True, text=True,
                ).stdout
                if f'"{existing}"' in alive:
                    print(
                        f"[LiveDaemon] Another daemon is already running (PID {existing}). "
                        f"Exiting so the live record has exactly one writer."
                    )
                    return False
            print(f"[LiveDaemon] Clearing stale lock from PID {existing} (no longer running).")

        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(str(os.getpid()), encoding="utf-8")
        self._lock_path = lock_path
        return True

    def _release_single_instance_lock(self) -> None:
        path = getattr(self, "_lock_path", None)
        if path is not None and path.exists():
            try:
                if path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                    path.unlink()
            except OSError:
                pass

    def run_forever(self, poll_seconds: int = 300) -> None:
        """
        Unattended supervision loop.

        Polls on a fixed interval rather than sleeping until the next hour, so a
        missed wake-up, a suspend/resume, or a clock change self-corrects on the
        following poll instead of stalling for an hour.
        """
        if not self._acquire_single_instance_lock():
            return

        print("=" * 80)
        print("STAGE 4 LIVE PAPER TRADING DAEMON — SUPERVISION LOOP ACTIVE")
        print(f"Polling every {poll_seconds}s | Capital ${self.initial_capital_usd:,.2f}")
        print(f"PID {os.getpid()} holds the single-instance lock.")
        print("=" * 80)

        # The interrupt handler and the lock release both sit OUTSIDE the loop
        # deliberately.
        #
        # They did not, and both were defects found when this daemon was
        # actually stopped. The KeyboardInterrupt branch used to live inside the
        # loop's try, while time.sleep() sat outside it -- and the daemon spends
        # almost its entire life inside that sleep, so an interrupt virtually
        # never reached the clean-shutdown path. No DAEMON_STOPPED event was
        # logged on the one real shutdown this daemon ever had. The lock had no
        # try/finally at all and was cleared by hand.
        try:
            while True:
                try:
                    if self.state is None:
                        self.initialize_or_resume()
                    else:
                        self.run_live_cycle()
                except FeedUnreachable:
                    pass  # already logged; retry on the next poll
                except Exception as err:
                    self._log_event(
                        "UNEXPECTED_ERROR",
                        detail=f"{type(err).__name__}: {err}",
                        action="Cycle skipped; loop continues.",
                    )
                # Inside the outer try: an interrupt here must be caught too.
                # KeyboardInterrupt derives from BaseException, so the inner
                # `except Exception` above never swallows it.
                time.sleep(poll_seconds)
        except KeyboardInterrupt:
            self._log_event("DAEMON_STOPPED", detail="Manual interrupt (clean shutdown).")
            print("\n[LiveDaemon] Stopped by user.")
        finally:
            self._release_single_instance_lock()


if __name__ == "__main__":
    daemon = LivePaperTradingDaemon()
    if "--once" in sys.argv:
        daemon.initialize_or_resume()
        daemon.run_live_cycle()
    else:
        daemon.run_forever()

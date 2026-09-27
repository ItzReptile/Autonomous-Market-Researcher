from typing import Dict, List, Optional, Tuple
from .types import Bar, BarState, CostModel, ExecutionPolicy, PortfolioState


class DeterministicSimulator:
    """
    Deterministic spot portfolio accounting simulator.
    Strictly enforces:
      1. Spot-only (w_i >= 0, sum(w_i) <= 1.0).
      2. 1-bar execution delay: targets set at bar t execute at bar t+1 Open.
      3. Zero-liquidity protection: NO_TRADES bars defer execution up to h_max bars.
      4. Explicit per-side fee and slippage models.
      5. Automatic delisting liquidation with penalty.
    """
    def __init__(self, initial_cash: float = 100_000.0, policy: Optional[ExecutionPolicy] = None):
        self.initial_cash = float(initial_cash)
        self.policy = policy or ExecutionPolicy()
        self.cash = float(initial_cash)
        self.positions: Dict[str, float] = {}       # symbol -> units
        self.deferred_targets: Dict[str, float] = {}  # symbol -> pending target weight
        self.deferral_counts: Dict[str, int] = {}    # symbol -> count of deferred bars
        self.last_known_close: Dict[str, float] = {}
        self.history: List[PortfolioState] = []
        self.total_fees_paid = 0.0
        self.total_slippage_paid = 0.0

    def step(
        self,
        current_bars: Dict[str, Bar],
        target_weights: Dict[str, float],
    ) -> PortfolioState:
        """
        Executes one time step t+1 given target_weights decided at bar t.
        current_bars: Dict[symbol, Bar] at step t+1.
        target_weights: Dict[symbol, float] decided at step t.
        """
        # 1. Validate target weights for spot rules
        total_target_asset_weight = sum(target_weights.values())
        for sym, w in target_weights.items():
            if w < -1e-12:
                raise ValueError(f"Spot violation: Negative/short weight {w} for {sym}")
        if total_target_asset_weight > 1.0 + 1e-12:
            raise ValueError(
                f"Leverage violation: Total target weight {total_target_asset_weight:.6f} > 1.0"
            )

        # Merge pending deferred targets from previous steps if not explicitly overridden
        active_targets = dict(self.deferred_targets)
        active_targets.update(target_weights)

        # 2. Update last known close prices
        for sym, bar in current_bars.items():
            if bar.state != BarState.DATA_GAP and bar.close > 0:
                self.last_known_close[sym] = bar.close

        # 3. Handle Delisted assets with open positions
        delisted_symbols = [
            sym for sym, bar in current_bars.items() if bar.state == BarState.DELISTED
        ]
        for sym in delisted_symbols:
            if self.positions.get(sym, 0.0) > 1e-12:
                units = self.positions[sym]
                ref_price = self.last_known_close.get(sym, current_bars[sym].close)
                # Delisting penalty slippage
                penalty = ref_price * (self.policy.cost_model.delisting_slippage_bps / 10000.0)
                exec_price = max(0.0, ref_price - penalty)
                recovered_cash = units * exec_price
                self.cash += recovered_cash
                self.positions[sym] = 0.0
                self.total_slippage_paid += units * penalty
                if sym in active_targets:
                    active_targets[sym] = 0.0
                if sym in self.deferred_targets:
                    del self.deferred_targets[sym]
                if sym in self.deferral_counts:
                    del self.deferral_counts[sym]

        # 4. Pre-rebalance portfolio valuation at step t+1 Open
        open_prices: Dict[str, float] = {}
        for sym in list(self.positions.keys()) + list(active_targets.keys()):
            if sym in current_bars and current_bars[sym].open > 0:
                open_prices[sym] = current_bars[sym].open
            elif sym in self.last_known_close:
                open_prices[sym] = self.last_known_close[sym]
            else:
                open_prices[sym] = 0.0

        pre_portfolio_value = self.cash
        for sym, units in self.positions.items():
            if units > 0 and sym in open_prices:
                pre_portfolio_value += units * open_prices[sym]

        if pre_portfolio_value <= 0.0:
            raise RuntimeError("Portfolio bankruptcy: value <= 0")

        # Compute pre-rebalance weights
        pre_weights: Dict[str, float] = {}
        for sym, units in self.positions.items():
            if units > 0 and sym in open_prices:
                pre_weights[sym] = (units * open_prices[sym]) / pre_portfolio_value
            else:
                pre_weights[sym] = 0.0

        # 5. Process target weights and rebalance
        two_way_turnover = 0.0
        step_fees = 0.0
        step_slippage = 0.0

        # Assets involved in rebalance
        all_symbols = sorted(list(set(list(pre_weights.keys()) + list(active_targets.keys()))))

        for sym in all_symbols:
            target_w = active_targets.get(sym, 0.0)
            current_w = pre_weights.get(sym, 0.0)
            weight_delta = target_w - current_w

            if abs(weight_delta) < 1e-12:
                continue

            bar = current_bars.get(sym)

            # Check liquidity: Cannot execute on NO_TRADES or DELISTED or missing bar
            if bar is None or bar.state == BarState.NO_TRADES or bar.state == BarState.DELISTED:
                # Defer execution
                count = self.deferral_counts.get(sym, 0) + 1
                self.deferral_counts[sym] = count
                if count <= self.policy.h_max_deferred_bars:
                    self.deferred_targets[sym] = target_w
                else:
                    # Exceeded H_max: abort deferred order
                    if sym in self.deferred_targets:
                        del self.deferred_targets[sym]
                    self.deferral_counts[sym] = 0
                continue

            # If bar is normal, clear deferral tracking
            if sym in self.deferred_targets:
                del self.deferred_targets[sym]
            if sym in self.deferral_counts:
                del self.deferral_counts[sym]

            # Execute trade at Open
            p_open = bar.open
            trade_value = abs(weight_delta) * pre_portfolio_value

            fee_rate = self.policy.cost_model.fee_bps_per_side / 10000.0
            slip_rate = self.policy.cost_model.slippage_bps_per_side / 10000.0

            if weight_delta > 0:  # BUY
                p_exec = p_open * (1.0 + slip_rate)
                fee = trade_value * fee_rate
                slippage_cost = trade_value * slip_rate
                # Cost paid from cash
                total_cost = trade_value + fee
                if total_cost > self.cash + 1e-6:
                    # Clip to available cash
                    trade_value = max(0.0, (self.cash) / (1.0 + fee_rate))
                    fee = trade_value * fee_rate
                    total_cost = trade_value + fee

                units_bought = trade_value / p_exec
                self.cash -= total_cost
                self.positions[sym] = self.positions.get(sym, 0.0) + units_bought

            else:  # SELL
                p_exec = p_open * (1.0 - slip_rate)
                fee = trade_value * fee_rate
                slippage_cost = trade_value * slip_rate
                units_to_sell = min(self.positions.get(sym, 0.0), trade_value / p_open)
                proceeds = (units_to_sell * p_exec) - fee
                self.cash += proceeds
                self.positions[sym] = max(0.0, self.positions.get(sym, 0.0) - units_to_sell)

            two_way_turnover += abs(weight_delta)
            step_fees += fee
            step_slippage += slippage_cost

        self.total_fees_paid += step_fees
        self.total_slippage_paid += step_slippage
        one_way_turnover = 0.5 * two_way_turnover

        # 6. Post-rebalance portfolio valuation at step t+1 Close
        close_prices: Dict[str, float] = {}
        for sym, units in self.positions.items():
            if units > 0:
                if sym in current_bars and current_bars[sym].close > 0:
                    close_prices[sym] = current_bars[sym].close
                elif sym in self.last_known_close:
                    close_prices[sym] = self.last_known_close[sym]
                else:
                    close_prices[sym] = 0.0

        post_portfolio_value = self.cash
        for sym, units in self.positions.items():
            if units > 0 and sym in close_prices:
                post_portfolio_value += units * close_prices[sym]

        final_weights: Dict[str, float] = {}
        for sym, units in self.positions.items():
            if units > 0 and sym in close_prices and post_portfolio_value > 0:
                final_weights[sym] = (units * close_prices[sym]) / post_portfolio_value
            else:
                final_weights[sym] = 0.0

        cash_weight = self.cash / post_portfolio_value if post_portfolio_value > 0 else 1.0

        # Invariant check: cash_weight + sum(final_weights) == 1.0
        total_w = cash_weight + sum(final_weights.values())
        if abs(total_w - 1.0) > 1e-8:
            raise AssertionError(f"Cash conservation invariant violated: total weight = {total_w}")

        current_timestamp = max(b.timestamp_close for b in current_bars.values())

        state = PortfolioState(
            timestamp=current_timestamp,
            cash=self.cash,
            positions={k: v for k, v in self.positions.items() if v > 1e-12},
            prices=close_prices,
            portfolio_value=post_portfolio_value,
            weights={k: v for k, v in final_weights.items() if v > 1e-12},
            cash_weight=cash_weight,
            one_way_turnover=one_way_turnover,
            two_way_turnover=two_way_turnover,
            total_fees_paid=self.total_fees_paid,
            total_slippage_paid=self.total_slippage_paid,
        )
        self.history.append(state)
        return state

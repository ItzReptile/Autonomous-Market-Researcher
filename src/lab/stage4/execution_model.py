"""
Stage 4 Shared Execution Model.

Single source of truth for friction and order quantization, used by BOTH the
historical paper engine and the live daemon so the two cannot diverge.

Two concerns live here:

1. CAUSAL FRICTION
   The original backtest friction normalised vol/spread by the mean over the
   *entire* sample, which is not computable in real time (it peeks at the
   future). This module replaces that with a trailing-window baseline that is
   identical in intent and strictly causal, so the live daemon can reproduce it
   bar for bar.

2. ORDER QUANTIZATION
   Exchange LOT_SIZE / MIN_NOTIONAL filters, which are immaterial at $100k but
   become measurable at a $1,000 basis. Filters are the real Binance.US spot
   values fetched from /api/v3/exchangeInfo.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple
import urllib.request

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FILTERS_CACHE_PATH = PROJECT_ROOT / "data" / "paper_trading" / "exchange_filters.json"

# Trailing window for the causal vol/spread baseline (30 days of hourly bars).
FRICTION_BASELINE_WINDOW = 720
# Minimum observations before the trailing baseline is considered meaningful.
FRICTION_MIN_WARMUP = 48

# ---------------------------------------------------------------------------
# Fee schedule -- THE canonical source. Nothing should define its own.
# ---------------------------------------------------------------------------
#
# PROVENANCE, stated plainly because this number is load-bearing:
#   Supplied by the account holder from their own Coinbase Advanced fee tier
#   (US account, under $10k 30-day volume) on 2026-09-22. It could NOT be
#   independently verified from a public page -- Coinbase gates the tier table
#   behind sign-in, and signing into the user's account to read it is not
#   something this project does. So it is recorded as user-supplied, not as
#   verified. A user can read their own tier and that is authoritative for
#   their account; if the tier is wrong, every number downstream moves, which
#   is exactly why these live in ONE place.
#
# HISTORY. The project ran on 5 bps for most of its life, raised to 10 bps in
# execution_model only, before live launch. Both were far too low: they were
# guesses at "typical retail", not a real published schedule. The research
# engine meanwhile stayed at 5 bps, so research was pricing fills at a
# NINETEENTH of the real entry-tier taker cost while deciding which candidates
# to promote.
MAKER_FEE_RATE = 0.0050      # 50 bps, Coinbase Advanced entry tier
TAKER_FEE_RATE = 0.0090      # 90 bps, Coinbase Advanced entry tier

# Superseded. Kept named so an old result can be reproduced deliberately rather
# than by a magic number reappearing somewhere.
LEGACY_FEE_RATE_PHASE1_TO_30 = 0.0005    # 5 bps
LEGACY_FEE_RATE_STAGE4_LIVE = 0.0010     # 10 bps

# WHICH SIDE APPLIES: taker. See FEE_SIDE_RATIONALE.
#
# This is not a conservative default, it is forced by an assumption already
# baked into the engine. The backtest fills every position at the close of the
# bar whose signal fired, with certainty. Maker fees are the price of NOT
# having that certainty: a passive order fills only if the market comes back to
# you. For a p97-percentile breakout entry, the times it comes back to you are
# disproportionately the times the signal was wrong -- so a maker assumption
# with guaranteed fills claims the cheap fee AND the certain execution, which
# are not jointly available. Modelling maker fees honestly would require a fill
# probability model the engine does not have.
FEE_SIDE = "taker"
BASE_FEE_RATE = TAKER_FEE_RATE
BASE_SLIPPAGE_RATE = 0.0005  # 5 bps base slippage coefficient (frozen spec)

FEE_SIDE_RATIONALE = (
    "Taker. The engine assumes a certain fill at the signal bar's close; maker "
    "fees are precisely the price of surrendering that certainty. Combining "
    "passive fees with guaranteed execution would be having it both ways, and "
    "the adverse selection runs the wrong way for a breakout entry: a resting "
    "limit order fills mostly when the move fails. Both sides are reported "
    "wherever it matters, but taker is the modelling assumption."
)


# ---------------------------------------------------------------------------
# Exchange filters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SymbolFilters:
    """LOT_SIZE / MIN_NOTIONAL constraints for one symbol."""
    symbol: str
    step_size: float
    min_qty: float
    min_notional: float
    tick_size: float

    def quantize_qty(self, raw_qty: float) -> float:
        """Round an order quantity DOWN to the exchange lot step."""
        if self.step_size <= 0:
            return raw_qty
        sign = -1.0 if raw_qty < 0 else 1.0
        steps = math.floor(abs(raw_qty) / self.step_size + 1e-9)
        return sign * steps * self.step_size


# Fallback values verified against Binance.US /api/v3/exchangeInfo.
# All four UNIV_10 symbols currently share identical filters.
DEFAULT_FILTERS: Dict[str, SymbolFilters] = {
    sym: SymbolFilters(sym, step_size=0.01, min_qty=0.01, min_notional=1.00, tick_size=0.001)
    for sym in ("DOTUSDT", "ATOMUSDT", "LINKUSDT", "UNIUSDT")
}


def fetch_exchange_filters(
    symbols: Tuple[str, ...] = ("DOTUSDT", "ATOMUSDT", "LINKUSDT", "UNIUSDT"),
    cache_path: Path = FILTERS_CACHE_PATH,
    timeout: int = 15,
) -> Dict[str, SymbolFilters]:
    """
    Fetch live LOT_SIZE / MIN_NOTIONAL filters, caching to disk.

    Falls back to the cache, then to DEFAULT_FILTERS, so a network failure can
    never block startup with a hard error.
    """
    quoted = ",".join(f'"{s}"' for s in symbols)
    url = f"https://api.binance.us/api/v3/exchangeInfo?symbols=%5B{quoted}%5D".replace('"', "%22")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 Stage4PaperTrading/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)

        parsed: Dict[str, SymbolFilters] = {}
        for entry in payload.get("symbols", []):
            sym = entry["symbol"]
            step = min_q = min_not = tick = None
            for f in entry.get("filters", []):
                if f["filterType"] == "LOT_SIZE":
                    step = float(f["stepSize"])
                    min_q = float(f["minQty"])
                elif f["filterType"] in ("MIN_NOTIONAL", "NOTIONAL"):
                    min_not = float(f.get("minNotional", f.get("minNotional", 0.0)))
                elif f["filterType"] == "PRICE_FILTER":
                    tick = float(f["tickSize"])
            if step is not None:
                parsed[sym] = SymbolFilters(
                    symbol=sym,
                    step_size=step,
                    min_qty=min_q if min_q is not None else step,
                    min_notional=min_not if min_not is not None else 0.0,
                    tick_size=tick if tick is not None else 0.0,
                )

        if parsed:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({k: vars(v) for k, v in parsed.items()}, f, indent=2)
            return {**DEFAULT_FILTERS, **parsed}
    except Exception as err:
        print(f"[ExecutionModel] Live filter fetch failed ({err}); falling back to cache/defaults.")

    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return {**DEFAULT_FILTERS, **{k: SymbolFilters(**v) for k, v in raw.items()}}
        except Exception:
            pass

    return dict(DEFAULT_FILTERS)


# ---------------------------------------------------------------------------
# Causal friction
# ---------------------------------------------------------------------------

def causal_friction_ratios(
    close_mat: np.ndarray,
    high_mat: np.ndarray,
    low_mat: np.ndarray,
    ret_mat: np.ndarray,
    baseline_window: int = FRICTION_BASELINE_WINDOW,
    min_warmup: int = FRICTION_MIN_WARMUP,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute vol_ratio and spread_ratio using only information available at or
    before each bar.

    The original implementation divided by the full-sample mean. Here the
    denominator is the mean over a trailing window (expanding until it fills),
    which is what a live process can actually observe.
    """
    n_bars, n_assets = close_mat.shape

    vol_24h = np.zeros_like(ret_mat)
    for t in range(24, n_bars):
        vol_24h[t] = np.std(ret_mat[t - 24 : t], axis=0)

    spread_proxy = (high_mat - low_mat) / np.maximum(1e-8, close_mat)

    vol_ratio = np.ones_like(vol_24h)
    spread_ratio = np.ones_like(spread_proxy)

    for t in range(n_bars):
        lo = max(0, t - baseline_window + 1)
        n_obs = t - lo + 1
        if n_obs < min_warmup:
            # Not enough trailing history: assume neutral (ratio 1.0).
            continue
        vol_base = np.maximum(1e-6, np.mean(vol_24h[lo : t + 1], axis=0))
        spread_base = np.maximum(1e-6, np.mean(spread_proxy[lo : t + 1], axis=0))
        vol_ratio[t] = vol_24h[t] / vol_base
        spread_ratio[t] = spread_proxy[t] / spread_base

    return vol_ratio, spread_ratio


def slippage_rate(vol_ratio_t: np.ndarray, spread_ratio_t: np.ndarray) -> np.ndarray:
    """State-dependent slippage rate for one bar (frozen spec formula)."""
    return BASE_SLIPPAGE_RATE * (0.5 * vol_ratio_t + 0.5 * spread_ratio_t)


# ---------------------------------------------------------------------------
# Quantized fills
# ---------------------------------------------------------------------------

@dataclass
class QuantizedFill:
    """Outcome of applying exchange filters to an intended order."""
    requested_notional: float
    filled_notional: float
    requested_qty: float
    filled_qty: float
    rejected_min_notional: bool
    quantization_error_usd: float

    @property
    def fill_ratio(self) -> float:
        if abs(self.requested_notional) < 1e-12:
            return 1.0
        return self.filled_notional / self.requested_notional


def quantize_order(
    requested_notional: float,
    price: float,
    filters: SymbolFilters,
) -> QuantizedFill:
    """
    Apply LOT_SIZE and MIN_NOTIONAL to an intended dollar order.

    Quantity is rounded DOWN to the lot step (the conservative direction: never
    trade more than intended). An order whose surviving notional falls below
    MIN_NOTIONAL is rejected outright, matching exchange behaviour.
    """
    if price <= 0:
        return QuantizedFill(requested_notional, 0.0, 0.0, 0.0, True, abs(requested_notional))

    raw_qty = requested_notional / price
    filled_qty = filters.quantize_qty(raw_qty)

    if abs(filled_qty) < filters.min_qty - 1e-12:
        return QuantizedFill(requested_notional, 0.0, raw_qty, 0.0, True, abs(requested_notional))

    filled_notional = filled_qty * price

    if abs(filled_notional) < filters.min_notional - 1e-12:
        return QuantizedFill(requested_notional, 0.0, raw_qty, 0.0, True, abs(requested_notional))

    return QuantizedFill(
        requested_notional=requested_notional,
        filled_notional=filled_notional,
        requested_qty=raw_qty,
        filled_qty=filled_qty,
        rejected_min_notional=False,
        quantization_error_usd=abs(requested_notional - filled_notional),
    )

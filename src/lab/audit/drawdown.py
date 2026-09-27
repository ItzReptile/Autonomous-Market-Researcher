"""
Drawdown conventions, named explicitly so the choice is never implicit.

WHY THIS MODULE EXISTS
----------------------
Until Phase 39 the research engine measured drawdown on the CUMULATIVE SUM of
per-bar returns. That quantity is not a drawdown of anything an account can
hold. It is unbounded below, it has no denominator, and -- decisively -- it is
proportional to gross exposure.

Phase 38 measured the consequence directly on the best candidate in the
project. Scaling every position by a constant, which changes no decision the
strategy makes:

    gross 1.00x -> Sharpe +2.1733  cumsum DD -39.07%  DSR 0.8686
    gross 0.77x -> Sharpe +2.1733  cumsum DD -30.09%  DSR 0.8686
    gross 0.50x -> Sharpe +2.1733  cumsum DD -19.54%  DSR 0.8686

Sharpe and DSR are invariant to four decimals; the "risk" number moves
wherever you want it. A -30% ceiling measured this way is therefore not a risk
constraint at all -- it is a leverage constraint wearing a risk label, and any
candidate with a real edge can satisfy it for free. Phases 33-38 spent roughly
160 trials optimising against it.

TRUE NAV DRAWDOWN is the replacement: compound the returns into an equity
curve and measure the worst peak-to-trough decline as a FRACTION of the peak.
It is the number an account holder actually experiences, it is bounded in
(-1, 0], it is comparable across strategies, and it is the convention the live
paper-trading daemon has used since launch (live_runner.py:
(nav - peak_nav) / peak_nav), so the change aligns research with live.

WHAT THIS CHANGE DOES **NOT** FIX
---------------------------------
It does not close the de-leveraging loophole. Measured on the same candidate:

    gross   Sharpe    cumsum DD    NAV DD     CAGR    Calmar(NAV)
    1.00   +2.1733     -39.07%    -36.87%   102.05%      2.768
    0.77   +2.1733     -30.09%    -28.96%    78.58%      2.713
    0.50   +2.1733     -19.54%    -19.16%    51.02%      2.663
    0.25   +2.1733      -9.77%     -9.70%    25.51%      2.630
    0.10   +2.1733      -3.91%     -3.90%    10.20%      2.617

NAV drawdown falls almost as fast as the cumsum form. Compounding makes it
mildly convex -- -36.87% vs the -39.07% a linear measure gives at 1.00x, and
the two converge as exposure shrinks -- but any ABSOLUTE drawdown ceiling
remains satisfiable by trading smaller, with Sharpe unchanged to four
decimals. The -30% gate is still a leverage constraint.

The quantity that is far more scale-stable is the RATIO. Measured across a
tenfold de-leverage, Calmar drifts 2.768 -> 2.617 on the parent candidate
(5.5%) and 2.896 -> 2.344 on a weaker Phase 38 variant (19.1%), against a
drawdown that moves by a factor of ten over the same range. It is not
perfectly invariant -- compounding makes CAGR mildly convex in exposure, so
the ratio drifts UPWARD with leverage rather than downward -- but it cannot be
gamed by trading smaller, which is the property the gate needs. Switching the GATE to that basis is
a policy decision and has not been taken here; `scale_invariant_calmar` is
computed and reported alongside every result so the gap is visible rather than
implicit.

So this change buys three things -- a real quantity instead of an artificial
one, a bounded and cross-comparable number, and parity with live -- and it
does not buy immunity to de-leveraging.

ONE-TIME COST, ACCEPTED
-----------------------
Every drawdown figure reported in Phases 22-38 was computed under the cumsum
convention and is NOT comparable to anything scored afterwards. The two are
different quantities, not two estimates of one quantity, so there is no
conversion factor. Historical numbers stay in the ledgers as they were
recorded; they must be read with their convention, which is why
`max_drawdown_cumsum` is retained rather than deleted -- replaying a
historical result requires reproducing its arithmetic exactly.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# The convention used by every gate scored from Phase 39 onward.
CURRENT_CONVENTION = "nav"
LEGACY_CONVENTION = "cumsum"
CONVENTION_CHANGED_AT_PHASE = 39


def max_drawdown_nav(returns: Sequence[float]) -> float:
    """
    True peak-to-trough decline of the compounded equity curve, as a fraction
    of the running peak. Returns a value in (-1.0, 0.0].

    Not de-leverageable: scaling the return stream shrinks the trough and the
    peak together, so the ratio is preserved to first order.
    """
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    # Guard the pathological case of a bar that wipes the account out. Equity
    # is floored just above zero so the curve stays defined and the drawdown
    # saturates at -100% instead of producing a nan that a gate would read as
    # "no drawdown".
    equity = np.cumprod(1.0 + arr)
    equity = np.maximum(equity, 1e-12)
    peak = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    dd = (equity - peak) / peak
    return float(np.min(dd)) if dd.size else 0.0


def max_drawdown_cumsum(returns: Sequence[float]) -> float:
    """
    LEGACY. Worst decline of the running SUM of returns, in return units.

    Retained only to reproduce results recorded before Phase 39. Do not use it
    for any new gate: it is proportional to gross exposure, so it can be
    satisfied by de-leveraging without changing the strategy.
    """
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(np.min(dd)) if dd.size else 0.0


def max_drawdown(returns: Sequence[float], convention: str = CURRENT_CONVENTION) -> float:
    """Dispatch by name. Unknown conventions raise rather than defaulting."""
    if convention == "nav":
        return max_drawdown_nav(returns)
    if convention == "cumsum":
        return max_drawdown_cumsum(returns)
    raise ValueError(
        f"Unknown drawdown convention {convention!r}; expected 'nav' or 'cumsum'."
    )


def scale_invariant_calmar(returns: Sequence[float], periods_per_year: int = 8760) -> float:
    """
    Annualised return per unit of NAV drawdown.

    Reported, not gated. Unlike an absolute drawdown ceiling, this cannot be
    improved by trading smaller: measured drift over a tenfold de-leverage is
    5.5% on the parent candidate and 19.1% on a weaker one, and in both cases
    the drift is DOWNWARD as exposure falls, so de-leveraging hurts it. A
    drawdown ceiling over the same range moves by a factor of ten. Positive is
    better; 0.0 when there is no drawdown to divide by.
    """
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    dd = max_drawdown_nav(arr)
    if abs(dd) < 1e-12:
        return 0.0
    equity = float(np.prod(1.0 + arr))
    if equity <= 0.0:
        return -1.0 / abs(dd)
    cagr = equity ** (periods_per_year / arr.size) - 1.0
    return float(cagr / abs(dd))


def geometric_cagr(returns: Sequence[float], periods_per_year: int = 8760) -> float:
    """
    Compounded annual growth rate of the NAV curve.

    The Phase 30 engine's `cagr_net` field is ARITHMETIC (mean * periods), which
    does not describe the curve NAV drawdown is measured on. On the project's
    best candidate the two differ materially: arithmetic 102.05% against a real
    compounded 148.54% (equity 15.29x over 3.00 years). Pairing arithmetic CAGR
    with NAV drawdown would give Calmar 2.768 where the coherent geometric
    pairing gives 4.029. Calmar must use this one.
    """
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    equity = float(np.prod(1.0 + arr))
    if equity <= 0.0:
        return -1.0
    return float(equity ** (periods_per_year / arr.size) - 1.0)


def required_gross_for_drawdown_budget(
    returns: Sequence[float], budget: float = -0.30, tol: float = 1e-4
) -> float:
    """
    Gross multiplier that brings NAV drawdown to `budget`, capped at 1.0.

    This is what the retired -30% ceiling actually measured. Because Calmar is
    near scale-invariant while drawdown is not, position size is a DEPLOYMENT
    decision rather than a gate: a candidate earns its place on efficiency, and
    this number says how large it may then be traded for a given risk appetite.
    Reported, never gated.
    """
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size == 0:
        return 1.0
    if abs(max_drawdown_nav(arr)) <= abs(budget):
        return 1.0
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if abs(max_drawdown_nav(arr * mid)) > abs(budget):
            hi = mid
        else:
            lo = mid
    return float(lo)

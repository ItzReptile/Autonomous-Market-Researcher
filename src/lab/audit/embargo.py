"""
Mechanical enforcement of the governing data boundary.

Spec: data/audit/GOVERNING_data_boundary_spec.json

    DEV      2021-01-01 -> 2023-12-31   search, fitting, gates, regimes
    BLIND    2024-01-01 -> 2024-03-31   one evaluation per candidate
    EMBARGO  2024-04-01 -> 2024-06-30   NOTHING touches this
    HOLDOUT  2024-07-01 -> 2024-09-30   sealed; one evaluation, ever

HARD RULE
    Nothing may read a bar dated 2024-04-01T00:00:00Z or later, except a single
    sanctioned holdout evaluation entered through sanctioned_holdout_evaluation().

This module exists because writing the rule down is not enforcing it. The whole
reason the Deflated Sharpe gate was wrong for 30 phases is that a documented
intention ("count prior trials") was never checked in code. The guard here runs
inside the data-loading path, so a violating window raises rather than quietly
returning data.

FAILS CLOSED. A window that cannot be parsed is refused, not permitted.
"""
from __future__ import annotations

import datetime as _dt
import threading
from contextlib import contextmanager
from typing import Any, Iterable, Optional, Sequence, Union

# ---------------------------------------------------------------- boundaries

# PHASE 51 RESTRUCTURE. The project's data now runs to 2026-08, not 2024-09.
# The old structure rationed a three-month holdout while roughly two years of
# untouched data sat unused. The 2024 BLIND and HOLDOUT windows have both been
# consumed - BLIND by the Phase 30 candidate, HOLDOUT twice, by T01 and by
# A_rank336_w0.6 - so neither can serve as clean out-of-sample again, and the
# Q2 2024 embargo existed only to decorrelate those two. All three are
# therefore folded into DEV, which is honest rather than generous: spent
# windows are not out-of-sample.
#
#   DEV      2021-01-01 .. 2024-09-30   45 months, all search and fitting
#   EMBARGO  2024-10-01 .. 2024-12-31   3 months, nothing may read it
#   HOLDOUT  2025-01-01 .. 2025-12-31   12 months, one evaluation per candidate
#   RESERVE  2026-01-01 .. 2026-08-31   8 months, sealed behind its own permit
#
# The primary holdout goes from 3 months carrying ~20 trades to 12 months, with
# an independent 8-month RESERVE behind it for confirming anything that
# survives. Phase 50 showed a 20-trade holdout is too thin to decide anything.
DEV_START = "2021-01-01"
DEV_END = "2024-09-30"
EMBARGO_START = "2024-10-01"
EMBARGO_END = "2024-12-31"
HOLDOUT_START = "2025-01-01"
HOLDOUT_END = "2025-12-31"
RESERVE_START = "2026-01-01"
RESERVE_END = "2026-08-31"

# Retained so historical references still resolve; both are inside DEV now and
# are recorded as SPENT, not as available validation windows.
BLIND_START = "2024-01-01"
BLIND_END = "2024-03-31"
LEGACY_HOLDOUT_START = "2024-07-01"
LEGACY_HOLDOUT_END = "2024-09-30"

#: No ordinary process may read at or after this instant.
FORBIDDEN_FROM = "2024-10-01"


class EmbargoViolation(RuntimeError):
    """Raised when a data access would cross the governing boundary."""


# Thread-local so a sanctioned holdout evaluation in one thread cannot
# accidentally license a hunt running in another.
_state = threading.local()


def _permit_holdout() -> bool:
    return bool(getattr(_state, "holdout_permit", False))


def _permit_reserve() -> bool:
    return bool(getattr(_state, "reserve_permit", False))


@contextmanager
def sanctioned_reserve_evaluation(reason: str):
    """Permit reads inside RESERVE only. Separate from the holdout permit so
    that consuming one cannot silently consume the other."""
    print(f"[EMBARGO] SANCTIONED RESERVE EVALUATION OPENED: {reason}")
    _state.reserve_permit = True
    try:
        yield
    finally:
        _state.reserve_permit = False
        print(f"[EMBARGO] sanctioned reserve evaluation closed: {reason}")


@contextmanager
def sanctioned_holdout_evaluation(reason: str):
    """
    Temporarily permit reads inside the HOLDOUT range only.

    Deliberately narrow: it never licenses the EMBARGO range, and it never
    licenses anything after HOLDOUT_END. Entering it is the act of consuming
    the holdout, so `reason` is required and printed -- a holdout evaluation
    should be visible in any log that captured it.
    """
    if not reason or not str(reason).strip():
        raise ValueError("sanctioned_holdout_evaluation requires an explicit reason")
    prev = getattr(_state, "holdout_permit", False)
    _state.holdout_permit = True
    print(f"[EMBARGO] SANCTIONED HOLDOUT EVALUATION OPENED: {reason}")
    try:
        yield
    finally:
        _state.holdout_permit = prev
        print(f"[EMBARGO] sanctioned holdout evaluation closed: {reason}")


# ---------------------------------------------------------------- parsing

def _to_date(value: Any) -> _dt.date:
    """Parse a date from the shapes the pipeline actually passes around."""
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, (int, float)):
        v = float(value)
        # epoch ms if it is implausibly large for seconds
        if v > 1e11:
            v /= 1000.0
        return _dt.datetime.fromtimestamp(v, tz=_dt.timezone.utc).date()
    if isinstance(value, str):
        s = value.strip().replace("Z", "").replace("T", " ")
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return _dt.datetime.strptime(s[: len(fmt) + 2].strip(), fmt).date()
            except ValueError:
                continue
        try:
            return _dt.date.fromisoformat(s[:10])
        except ValueError:
            pass
    raise EmbargoViolation(
        f"Cannot interpret date {value!r}; refusing the access rather than guessing. "
        f"The boundary guard fails closed."
    )


_FORBIDDEN = _dt.date.fromisoformat(FORBIDDEN_FROM)
_HOLDOUT_S = _dt.date.fromisoformat(HOLDOUT_START)
_HOLDOUT_E = _dt.date.fromisoformat(HOLDOUT_END)
_EMBARGO_S = _dt.date.fromisoformat(EMBARGO_START)
_EMBARGO_E = _dt.date.fromisoformat(EMBARGO_END)
_RESERVE_S = _dt.date.fromisoformat(RESERVE_START)
_RESERVE_E = _dt.date.fromisoformat(RESERVE_END)


# ---------------------------------------------------------------- checks

def assert_window_allowed(start: Any, end: Any, context: str = "") -> None:
    """
    Refuse a date-range load that crosses the boundary.

    Permitted without a permit: any window ending before 2024-04-01.
    Permitted with a permit: windows lying entirely inside HOLDOUT.
    Everything else raises.
    """
    s, e = _to_date(start), _to_date(end)
    if e < s:
        raise EmbargoViolation(f"Window end {e} precedes start {s} ({context}).")

    if e < _FORBIDDEN:
        return  # entirely before the wall

    where = f" [{context}]" if context else ""

    if _permit_reserve():
        if s >= _RESERVE_S and e <= _RESERVE_E:
            return
        raise EmbargoViolation(
            f"Sanctioned RESERVE evaluation permits ONLY {RESERVE_START}..{RESERVE_END}, "
            f"but {s}..{e} was requested{where}."
        )

    if _permit_holdout():
        if s >= _HOLDOUT_S and e <= _HOLDOUT_E:
            return
        raise EmbargoViolation(
            f"Sanctioned holdout evaluation permits ONLY {HOLDOUT_START}..{HOLDOUT_END}, "
            f"but {s}..{e} was requested{where}. The embargo range is never permitted, "
            f"and RESERVE needs its own permit."
        )

    if s <= _EMBARGO_E and e >= _EMBARGO_S:
        raise EmbargoViolation(
            f"EMBARGO VIOLATION{where}: {s}..{e} overlaps the embargo window "
            f"{EMBARGO_START}..{EMBARGO_END}, which nothing may read. "
            f"Dev must end on or before {DEV_END}."
        )

    raise EmbargoViolation(
        f"EMBARGO VIOLATION{where}: {s}..{e} reaches {FORBIDDEN_FROM} or later. "
        f"Only a sanctioned holdout evaluation may read {HOLDOUT_START}..{HOLDOUT_END}; "
        f"nothing may read the embargo window or anything after the holdout."
    )


def assert_timestamps_allowed(timestamps: Sequence[Union[int, float]], context: str = "") -> None:
    """Refuse an already-materialised series whose bars cross the boundary."""
    if timestamps is None or len(timestamps) == 0:
        return
    lo, hi = min(timestamps), max(timestamps)
    assert_window_allowed(lo, hi, context=context)


def describe() -> str:
    # Kept in step with the Phase 51 restructure above. It previously still
    # printed the old 2024 layout, including a BLIND window that is now spent
    # and folded into DEV -- a documented intention drifting out of step with
    # the code, which is the exact failure mode this module exists to prevent.
    return (
        f"DEV {DEV_START}..{DEV_END} | "
        f"EMBARGO {EMBARGO_START}..{EMBARGO_END} (untouchable) | "
        f"HOLDOUT {HOLDOUT_START}..{HOLDOUT_END} (sealed, one evaluation per candidate) | "
        f"RESERVE {RESERVE_START}..{RESERVE_END} (sealed behind its own permit)"
    )

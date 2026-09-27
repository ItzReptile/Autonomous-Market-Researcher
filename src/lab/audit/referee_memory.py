"""
Cross-phase referee memory.

WHY THIS EXISTS
---------------
Phase 30's referee was constructed with an empty `prior_trial_returns`, so the
multiple-testing counter restarted at zero. Its trial 1 was therefore scored at
K_eff = 1.0 -- the weakest possible Deflated Sharpe test -- while the same
universe had already absorbed 19 trials in Phase 29 (and 19 each in Phases 26
and 27). Whichever variant happened to be evaluated first got the easiest gate,
which makes the DSR result order-dependent rather than a property of the
strategy.

This module loads the return series of every prior trial on a given universe,
across phases, so a referee can be seeded with the real history before scoring
anything new.

IMPORTANT: universe IDs were REDEFINED at Phase 26.
    phase21_catalog: UNIV_10 = BTC, ETH, LTC, XRP, BNB
    phase26_catalog: UNIV_10 = DOT, ATOM, LINK, UNI
They are different universes sharing a label. Matching is done on the ASSET SET,
never on the ID, so pre-Phase-26 trials are not silently folded in.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = PROJECT_ROOT / "data" / "audit"
# Retained for callers that still import it. NOT a default any more: resolving a
# bare label against this one catalog attached the wrong basket for 16 of 18 ids.
CATALOG_PATH = PROJECT_ROOT / "data" / "universes" / "phase26_catalog.json"


CATALOG_DIR = PROJECT_ROOT / "data" / "universes"


class AmbiguousUniverseLabel(KeyError):
    """A universe id means different asset sets in different catalogs."""


def universe_assets(universe_id: str, catalog_path: Optional[Path] = None) -> List[str]:
    """
    Resolve a universe label to its asset set.

    THE LABEL IS NOT THE IDENTITY. 16 of this project's 18 universe ids denote
    a DIFFERENT basket in phase21_catalog than in every later catalog -- e.g.
    UNIV_10 was {BNB,BTC,ETH,LTC,XRP} and is now {ATOM,DOT,LINK,UNI}. Resolving
    a bare label against whichever catalog happened to be the default is how a
    search history gets attached to the wrong basket.

    With catalog_path given, that catalog is authoritative. Without one, every
    catalog is consulted and the label must mean the SAME asset set in all of
    them; otherwise this raises rather than guessing. Failing closed is the
    point: the caller has to say which basket it means.
    """
    if catalog_path is not None:
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        entry = next((c for c in catalog if c["universe_id"] == universe_id), None)
        if entry is None:
            raise KeyError(f"{universe_id} not present in {catalog_path.name}")
        return list(entry["assets"])

    found: Dict[Tuple[str, ...], List[str]] = {}
    for cat in sorted(CATALOG_DIR.glob("*_catalog.json")):
        try:
            catalog = json.load(cat.open("r", encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        entry = next((c for c in catalog if c.get("universe_id") == universe_id), None)
        if entry is not None:
            found.setdefault(tuple(sorted(entry["assets"])), []).append(cat.name)
    if not found:
        raise KeyError(f"{universe_id} not present in any catalog under {CATALOG_DIR}")
    if len(found) > 1:
        detail = "; ".join(f"{','.join(k)} (from {', '.join(v)})" for k, v in found.items())
        raise AmbiguousUniverseLabel(
            f"{universe_id} denotes {len(found)} different asset sets: {detail}. "
            f"Pass catalog_path explicitly to say which one is meant."
        )
    return list(next(iter(found)))


def _row_universe_and_symbols(rec: Dict[str, Any]) -> Tuple[Optional[str], Optional[List[str]]]:
    """Ledger schemas differ by phase; look in both the row and its proposal."""
    uid = rec.get("universe_id")
    syms = rec.get("symbols")
    prop = rec.get("proposal") or {}
    if uid is None:
        uid = prop.get("universe_id")
    if syms is None:
        syms = prop.get("symbols")
    return uid, syms


def collect_stored_returns(
    universe_id: str,
    ledger_glob: str = "ledger_*.jsonl",
    catalog_path: Optional[Path] = None,
    include_subsets: bool = True,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    Return every stored net-return series relevant to this universe.

    include_subsets (default True)
        Also count searches run on a SUBSET of this universe's assets. A hunt
        over {BTC, ETH} is unquestionably multiple testing against
        {BTC, ETH, SOL}: same model, same prompts, same window, overlapping
        assets. Matching on the exact asset set alone let the 22-trial NOSOL
        diagnostic hide from UNIV_05's counter purely because it carried a
        different label -- precisely the "artificially low K from a fresh
        label" failure this module exists to prevent.

        Counting them can only raise K, never lower it, so the direction of
        the error is conservative.

    Rows that do not persist `net_returns` are reported by `audit_coverage`
    rather than silently dropped.
    """
    assets = set(universe_assets(universe_id, catalog_path))
    series: List[np.ndarray] = []
    provenance: List[Dict[str, Any]] = []

    for path in sorted(AUDIT_DIR.glob(ledger_glob)):
        for line in path.open("r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid, syms = _row_universe_and_symbols(rec)
            row_syms = set(syms) if syms is not None else None
            if row_syms is not None:
                # Match on assets, never on the label: universe ids were
                # redefined at Phase 26, so the same id can mean two different
                # baskets.
                if include_subsets:
                    if not row_syms or not row_syms.issubset(assets):
                        continue
                elif row_syms != assets:
                    continue
            else:
                # No symbols recorded. Matching on the label here would attach
                # this row to whatever basket currently owns the id, and 16 of
                # 18 ids denote different baskets in different catalogs. Skip
                # it instead; audit_coverage reports the gap. Measured on the
                # current ledgers this discards nothing: every row that stores
                # net_returns also stores symbols (0 affected rows).
                continue
            bt = rec.get("backtest") or {}
            nr = bt.get("net_returns") if isinstance(bt, dict) else None
            if not nr:
                continue
            arr = np.asarray(nr, dtype=np.float64)
            if arr.size < 2:
                continue
            series.append(arr)
            provenance.append({
                "ledger": path.name,
                "trial_index": rec.get("trial_index"),
                "variant_name": rec.get("variant_name"),
                "n_obs": int(arr.size),
                "symbols": sorted(row_syms) if row_syms else None,
                "subset_match": bool(row_syms is not None and row_syms != assets),
            })
    return series, provenance


def audit_coverage(universe_id: str, catalog_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Per-ledger count of this universe's trials vs how many persist returns.

    Coverage gaps matter: a phase whose returns were never stored cannot
    contribute to the correlation estimate, so K_eff computed from stored
    history is a LOWER BOUND on the true multiple-testing burden.
    """
    assets = set(universe_assets(universe_id, catalog_path))
    out: List[Dict[str, Any]] = []
    for path in sorted(AUDIT_DIR.glob("ledger_*.jsonl")):
        total = with_ret = 0
        for line in path.open("r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid, syms = _row_universe_and_symbols(rec)
            if uid != universe_id:
                continue
            if syms is not None and set(syms) != assets:
                continue
            total += 1
            bt = rec.get("backtest") or {}
            if isinstance(bt, dict) and bt.get("net_returns"):
                with_ret += 1
        if total:
            out.append({
                "ledger": path.name,
                "trials": total,
                "with_returns": with_ret,
                "missing_returns": total - with_ret,
            })
    return out


# Two trials belong to the same search only if their return series are this
# correlated. Set HIGH on purpose: a loose threshold merges genuinely different
# strategies into one cluster and hands back a lenient penalty, which is the
# failure mode this estimator exists to remove. Measured on real UNIV_10 data,
# 0.95 leaves 95% of the diverse Phase 29 pairs in separate clusters while still
# collapsing true near-duplicates.
CLUSTER_RHO_THRESHOLD = 0.95


def cluster_count_k(
    new_returns: np.ndarray,
    prior_returns: Sequence[np.ndarray],
    threshold: float = CLUSTER_RHO_THRESHOLD,
) -> Tuple[int, int, float]:
    """
    K_eff as the number of DISTINCT search clusters, not the raw trial count.

    Why this replaces the rho-average estimator
    -------------------------------------------
    K_eff = k/(1+(k-1)*rho) is bounded above by 1/rho, so flooding the family
    with near-duplicates raises rho and LOWERS the penalty. Measured on this
    project's own data, 50 near-duplicates moved the live candidate from
    DSR 0.7683 (fail) to 0.9725 (pass). More searching made the gate easier.

    Clustering removes that. A duplicate either joins an existing cluster (count
    unchanged) or forms a new one (count rises). It can never lower the count.

    COMPLETE LINKAGE is required for that guarantee. A cluster forms only when
    EVERY pair inside it clears the threshold, so adding a trial can never merge
    two previously separate clusters. Average or single linkage allow a new
    point to bridge two clusters and reduce the count -- reintroducing exactly
    the exploit being removed.

    Returns (n_clusters, n_trials, max_offdiag_rho).
    """
    series = list(prior_returns) + [new_returns]
    k = len(series)
    if k <= 1:
        return 1, k, 0.0

    m = min(int(np.asarray(s).size) for s in series)
    mat = np.column_stack([np.asarray(s)[:m] for s in series])
    corr = np.nan_to_num(np.corrcoef(mat, rowvar=False), nan=0.0)
    np.fill_diagonal(corr, 1.0)

    iu = np.triu_indices(k, 1)
    max_off = float(corr[iu].max()) if iu[0].size else 0.0

    # Complete-linkage agglomeration on correlation directly. Raw rho, not
    # |rho|: an anti-correlated strategy is treated as a separate search, which
    # is the conservative reading.
    clusters: List[List[int]] = [[i] for i in range(k)]
    merged = True
    while merged and len(clusters) > 1:
        merged = False
        best = None
        best_rho = threshold
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                # complete linkage: the WEAKEST cross pair governs the merge
                worst = min(corr[i, j] for i in clusters[a] for j in clusters[b])
                if worst >= best_rho:
                    best_rho = worst
                    best = (a, b)
        if best is not None:
            a, b = best
            clusters[a] = clusters[a] + clusters[b]
            del clusters[b]
            merged = True

    return len(clusters), k, max_off


def compute_k_eff(
    new_returns: np.ndarray,
    prior_returns: Sequence[np.ndarray],
) -> Tuple[float, float, int]:
    """
    K_eff over a correlated trial family.

    Identical formula to Phase30Referee.compute_effective_k, lifted here so the
    seeded and unseeded paths cannot drift apart:
        K_eff = k / (1 + (k-1) * rho_avg)
    Highly correlated variants add little; independent trials add nearly one
    each. Returns (k_eff, rho_avg, k_total).
    """
    k = len(prior_returns) + 1
    if k <= 1:
        return 1.0, 0.0, 1

    min_len = min(int(new_returns.size), min(int(r.size) for r in prior_returns))
    mat = np.column_stack([np.asarray(r)[:min_len] for r in prior_returns]
                          + [np.asarray(new_returns)[:min_len]])
    corr = np.nan_to_num(np.corrcoef(mat, rowvar=False), nan=0.0)
    iu = np.triu_indices(k, k=1)
    pair = corr[iu]
    rho = float(np.mean(pair)) if pair.size else 0.0
    rho = max(0.0, min(0.999, rho))
    k_eff = float(k / (1.0 + (k - 1) * rho))
    return max(1.0, min(float(k), k_eff)), rho, k


# ---------------------------------------------------------------------------
# Project-wide search footprint
# ---------------------------------------------------------------------------
#
# K_eff is necessarily PER-UNIVERSE: the Deflated Sharpe correction asks how
# many trials competed for the result being scored, and a search on a disjoint
# basket is not competing for this one. That is statistically right and it is
# also, on its own, misleading -- because the project chooses WHICH universe to
# report from.
#
# UNIV_05 now carries 141 clusters, so a candidate there needs Sharpe ~2.45. An
# untouched basket would start near K=1 and need ~1.0. Nothing about the second
# strategy is better; it is being graded on a curve the first one earned. With
# 18 baskets available, "hunt until one clears" is a selection effect operating
# one level above the one DSR corrects for, and it is exactly the mechanism
# that produced this project's fake winner: K_eff reset to 1 turned a
# DSR 0.4331 failure into a 0.9725 pass.
#
# There is no honest way to fold that into a single K without inventing a
# cross-universe dependence structure the data cannot support. So it is
# REPORTED instead: every result carries the number of baskets ever searched
# project-wide, alongside its own universe's count. A reader can then see that
# a "fresh universe, K=3" result is the 19th basket tried, not the first.


def project_search_footprint(ledger_glob: str = "ledger_*.jsonl") -> Dict[str, Any]:
    """
    How much searching this project has done, across every basket.

    Identity is the ASSET SET, never the label: 16 of 18 universe ids denote
    different baskets in different catalogs, so counting labels would both
    double-count and under-count.

    Returns counts plus a per-basket breakdown, sorted by trials descending.
    """
    by_assets: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    ledgers: set = set()
    rows_total = 0
    rows_without_symbols = 0
    rows_with_returns = 0

    for path in sorted(AUDIT_DIR.glob(ledger_glob)):
        for line in path.open("r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows_total += 1
            ledgers.add(path.name)
            uid, syms = _row_universe_and_symbols(rec)
            bt = rec.get("backtest") or {}
            has_ret = bool(isinstance(bt, dict) and bt.get("net_returns"))
            rows_with_returns += int(has_ret)
            if not syms:
                rows_without_symbols += 1
                continue
            key = tuple(sorted(syms))
            slot = by_assets.setdefault(key, {
                "assets": list(key), "trials": 0, "with_returns": 0,
                "labels": set(), "ledgers": set()})
            slot["trials"] += 1
            slot["with_returns"] += int(has_ret)
            if uid:
                slot["labels"].add(uid)
            slot["ledgers"].add(path.name)

    baskets = sorted(by_assets.values(), key=lambda d: -d["trials"])
    for b in baskets:
        b["labels"] = sorted(b["labels"])
        b["ledgers"] = sorted(b["ledgers"])

    return {
        "universes_ever_searched": len(by_assets),
        "total_trials_logged": rows_total,
        "trials_with_stored_returns": rows_with_returns,
        "rows_without_symbols_unattributable": rows_without_symbols,
        "ledgers": len(ledgers),
        "per_basket": [
            {"assets": b["assets"], "labels": b["labels"],
             "trials": b["trials"], "with_returns": b["with_returns"],
             "n_ledgers": len(b["ledgers"])}
            for b in baskets
        ],
        "reading": (
            "K_eff is per-universe by construction. This is the level above it: "
            "the number of DISTINCT BASKETS ever searched. A result on a basket "
            "with a low K is not a first look if this count is high -- it is the "
            "Nth basket tried, and that selection is not corrected by DSR."),
    }

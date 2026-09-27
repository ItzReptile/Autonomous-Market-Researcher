"""
Stage 4 Atomic State Storage Manager.
Guarantees transactional integrity of strategy state across sudden power loss,
hard kills, or OS crashes by writing to a temporary file, fsyncing to physical disk,
and performing an atomic filesystem rename.
"""
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STATE_PATH = PROJECT_ROOT / "data" / "paper_trading" / "live_state.json"


@dataclass
class LivePortfolioState:
    strategy_name: str
    universe_id: str
    symbols: List[str]
    last_processed_timestamp_utc: int  # ms epoch
    last_processed_datetime_str: str   # ISO-8601 UTC
    portfolio_nav_usd: float
    cash_usd: float
    positions: Dict[str, float]        # symbol -> current position [-0.70, +0.70]
    position_caps: Dict[str, float]
    capital_weights: Dict[str, float]
    total_trades_count: int
    running_net_sharpe: float
    running_max_drawdown: float
    peak_nav_usd: float
    last_bar_close_prices: Dict[str, float]
    created_at_utc: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LivePortfolioState":
        return cls(
            strategy_name=data["strategy_name"],
            universe_id=data["universe_id"],
            symbols=data["symbols"],
            last_processed_timestamp_utc=data["last_processed_timestamp_utc"],
            last_processed_datetime_str=data["last_processed_datetime_str"],
            portfolio_nav_usd=data["portfolio_nav_usd"],
            cash_usd=data["cash_usd"],
            positions=data["positions"],
            position_caps=data["position_caps"],
            capital_weights=data["capital_weights"],
            total_trades_count=data["total_trades_count"],
            running_net_sharpe=data["running_net_sharpe"],
            running_max_drawdown=data["running_max_drawdown"],
            peak_nav_usd=data["peak_nav_usd"],
            last_bar_close_prices=data["last_bar_close_prices"],
            created_at_utc=data.get("created_at_utc", ""),
            version=data.get("version", 1),
        )


class AtomicStateManager:
    """
    Atomic read/write manager for strategy execution state.
    Uses temporary files and atomic os.replace to eliminate partial-write corruptions.
    """

    # Temp files older than this are considered abandoned by a killed process.
    STALE_TEMP_AGE_SEC = 300

    def __init__(self, state_path: Path = DEFAULT_STATE_PATH):
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_path = self.state_path.with_suffix(".json.bak")
        self.cleanup_stale_temp_files()

    def cleanup_stale_temp_files(self, max_age_sec: Optional[float] = None) -> int:
        """
        Remove temp files abandoned by a killed process.

        A hard kill between NamedTemporaryFile creation and os.replace leaves
        the temp file behind. The state itself is never corrupted -- that is the
        point of the atomic rename -- but across months of crash/restart cycles
        the orphans accumulate in the data directory. Only files older than
        STALE_TEMP_AGE_SEC are removed, so a temp file being written right now
        by a concurrent process is never touched.
        """
        threshold = self.STALE_TEMP_AGE_SEC if max_age_sec is None else max_age_sec
        removed = 0
        now = time.time()
        try:
            for tmp in self.state_path.parent.glob("*.tmp"):
                try:
                    if now - tmp.stat().st_mtime >= threshold:
                        tmp.unlink()
                        removed += 1
                except OSError:
                    continue
        except OSError:
            pass
        if removed:
            print(f"[AtomicStateManager] Swept {removed} orphaned temp file(s) from a prior hard kill.")
        return removed

    def save_state(self, state: LivePortfolioState) -> None:
        """
        Atomically saves state to disk.
        1. Serializes to temporary file in the same directory (guarantees same filesystem for atomic rename).
        2. Flushes buffers and executes os.fsync to force physical flush to storage.
        3. Creates a backup copy of existing state if present.
        4. Atomically replaces target file via os.replace.
        """
        state_dict = state.to_dict()
        dir_path = self.state_path.parent
        dir_path.mkdir(parents=True, exist_ok=True)

        # Create temporary file in target directory
        with tempfile.NamedTemporaryFile("w", dir=dir_path, delete=False, encoding="utf-8", suffix=".tmp") as tf:
            temp_name = tf.name
            json.dump(state_dict, tf, indent=2)
            tf.flush()
            os.fsync(tf.fileno())

        # Back up the previous state by COPYING, never by moving.
        #
        # An earlier implementation used os.replace(state_path -> backup_path),
        # which moves the primary file away and leaves a window in which
        # state_path does not exist at all. A hard kill inside that window left
        # no primary state file on disk. Recovery still succeeded via the backup
        # fallback, but the window is avoidable and was observed to be hit under
        # a tight write loop. Copying keeps the primary present at every instant.
        if self.state_path.exists():
            try:
                shutil.copy2(self.state_path, self.backup_path)
            except Exception:
                pass  # Non-fatal if backup fails

        # Atomic replacement: state_path goes from old content to new content
        # with no intermediate state in which it is missing or partial.
        os.replace(temp_name, self.state_path)

    def load_state(self) -> Optional[LivePortfolioState]:
        """
        Loads state from disk with fallback to backup if primary is corrupted or missing.
        Returns None if neither exists.
        """
        for candidate_path in [self.state_path, self.backup_path]:
            if not candidate_path.exists():
                continue
            try:
                with open(candidate_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return LivePortfolioState.from_dict(data)
            except (json.JSONDecodeError, KeyError, ValueError) as err:
                print(f"[AtomicStateManager] Warning: Corrupted state file at {candidate_path}: {err}")
                continue

        return None

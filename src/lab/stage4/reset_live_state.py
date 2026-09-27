"""
Stage 4 Live Record Reset.

The working tree's paper_trading/ directory is contaminated: trade_log.jsonl
holds the Q1 2024 blind-test run (the engine opens it in "w" mode, so that test
overwrote the live log), while live_summary.json and daily_metrics.jsonl hold
Q4 2024 backtest output. None of it belongs in a forward record that is meant
to start today.

This script archives everything to a timestamped folder and leaves a clean
directory, so the live run starts genuinely fresh rather than layered on top of
backtest artifacts. Nothing is deleted -- the prior files remain inspectable
under data/paper_trading/_archive/.
"""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PAPER_DIR = PROJECT_ROOT / "data" / "paper_trading"
ARCHIVE_ROOT = PAPER_DIR / "_archive"

# Files that constitute the live record and must be clear before launch.
LIVE_RECORD_FILES = [
    "trade_log.jsonl",
    "daily_metrics.jsonl",
    "live_summary.json",
    "live_state.json",
    "live_state.json.bak",
    "live_baselines_state.json",
    "nav_history.jsonl",
    "gap_events.jsonl",
    "feed_health.json",
    "daemon.log",
]


def reset_live_record(dry_run: bool = False, label: str = "") -> Path:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{label}" if label else ""
    archive_dir = ARCHIVE_ROOT / f"pre_launch_{stamp}{suffix}"

    print("=" * 78)
    print("STAGE 4 LIVE RECORD RESET")
    print(f"Source : {PAPER_DIR}")
    print(f"Archive: {archive_dir}")
    print("=" * 78)

    moved, absent = [], []
    for name in LIVE_RECORD_FILES:
        src = PAPER_DIR / name
        if src.exists():
            moved.append(name)
        else:
            absent.append(name)

    for name in moved:
        src = PAPER_DIR / name
        size = src.stat().st_size
        print(f"  ARCHIVE  {name:<28} ({size:,} bytes)")
    for name in absent:
        print(f"  (absent) {name}")

    if dry_run:
        print("\nDRY RUN — nothing moved.")
        return archive_dir

    archive_dir.mkdir(parents=True, exist_ok=True)
    for name in moved:
        shutil.move(str(PAPER_DIR / name), str(archive_dir / name))

    manifest = {
        "archived_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reason": (
            "Pre-launch reset. trade_log.jsonl contained the Q1 2024 blind-test run; "
            "daily_metrics.jsonl and live_summary.json contained Q4 2024 backtest output. "
            "Archived so the forward record starts empty."
        ),
        "files_archived": moved,
        "files_absent": absent,
    }
    with open(archive_dir / "ARCHIVE_MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nArchived {len(moved)} file(s) -> {archive_dir}")
    print("Live record directory is now clean.")
    return archive_dir


def verify_clean() -> bool:
    """Confirm no live-record file survives in the working directory."""
    leftovers = [n for n in LIVE_RECORD_FILES if (PAPER_DIR / n).exists()]
    if leftovers:
        print("NOT CLEAN — still present:")
        for n in leftovers:
            print(f"  {n}")
        return False
    print("VERIFIED CLEAN — no live-record files present; forward run will start empty.")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    if args.verify_only:
        sys.exit(0 if verify_clean() else 1)

    reset_live_record(dry_run=args.dry_run, label=args.label)
    if not args.dry_run:
        sys.exit(0 if verify_clean() else 1)

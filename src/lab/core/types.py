from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class BarState(str, Enum):
    NORMAL = "NORMAL"
    NO_TRADES = "NO_TRADES"
    DATA_GAP = "DATA_GAP"
    EXCHANGE_OUTAGE = "EXCHANGE_OUTAGE"
    DELISTED = "DELISTED"


@dataclass(frozen=True)
class Bar:
    timestamp_open: int  # ms or us
    timestamp_close: int
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades_count: int
    state: BarState = BarState.NORMAL


@dataclass
class CostModel:
    fee_bps_per_side: float = 5.0          # 5 basis points = 0.0005
    slippage_bps_per_side: float = 5.0     # 5 basis points = 0.0005
    delisting_slippage_bps: float = 200.0  # 200 basis points = 0.02


@dataclass
class ExecutionPolicy:
    h_max_deferred_bars: int = 3
    cost_model: CostModel = field(default_factory=CostModel)


@dataclass
class PortfolioState:
    timestamp: int
    cash: float
    positions: Dict[str, float]          # symbol -> units held
    prices: Dict[str, float]             # symbol -> current valuation price
    portfolio_value: float
    weights: Dict[str, float]            # symbol -> portfolio weight
    cash_weight: float
    one_way_turnover: float = 0.0
    two_way_turnover: float = 0.0
    total_fees_paid: float = 0.0
    total_slippage_paid: float = 0.0

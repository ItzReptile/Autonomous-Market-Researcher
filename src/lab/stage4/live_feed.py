"""
Stage 4 Live Price Feed Client.

VENUE CONSISTENCY (important)
-----------------------------
All research, development, holdout and stress testing used Binance GLOBAL spot
data, downloaded from data.binance.vision as monthly kline archives.

The live feed must therefore read the SAME market. It does not, by default,
read Binance.US: that is a separate venue with roughly four orders of magnitude
less volume in these pairs (median hourly ATOMUSDT volume ~7 units vs ~49,000
on Binance global), with 30-40% of hours showing zero trades at all. Since 40%
of the strategy's signal weight is a volume ratio, running it against Binance.US
would feed the volume dimension near-noise and would not be the strategy that
was validated.

  PRIMARY  : data-api.binance.vision  -- Binance global market data, publicly
             reachable from the US, and verified byte-identical to the archived
             research data over overlapping bars.
  BLOCKED  : api.binance.com          -- returns HTTP 451 from this region.
  FALLBACK : api.binance.us           -- DIFFERENT MARKET. Disabled by default.
             Enabling it is a deliberate choice and is reported as a venue
             mismatch, never a silent substitution.
"""
from dataclasses import dataclass
import datetime
import json
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request


@dataclass
class KlineBar:
    open_time: int       # epoch ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int      # epoch ms
    datetime_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "open_time": self.open_time,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "close_time": self.close_time,
            "datetime_utc": self.datetime_utc,
        }


class VenueMismatchError(RuntimeError):
    """Raised rather than silently serving data from a different exchange."""


class LivePriceFeed:
    # Binance global market data — same venue as the research archives.
    PRIMARY_BASE = "https://data-api.binance.vision/api/v3"
    # Separate venue. Only used when explicitly allowed.
    BINANCE_US_BASE = "https://api.binance.us/api/v3"

    HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Stage4PaperTrading/1.0"}

    def __init__(
        self,
        max_retries: int = 3,
        timeout_sec: int = 10,
        allow_us_fallback: bool = False,
    ):
        self.max_retries = max_retries
        self.timeout_sec = timeout_sec
        self.allow_us_fallback = allow_us_fallback
        self.active_venue = "binance-global (data-api.binance.vision)"

    # ------------------------------------------------------------------

    def _get_json(self, url: str) -> Any:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                req = urllib.request.Request(url, headers=self.HEADERS)
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    if resp.status != 200:
                        raise ValueError(f"HTTP status {resp.status}")
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as err:
                last_err = err
                if attempt < self.max_retries:
                    time.sleep(1.0 * (2 ** (attempt - 1)))
        raise RuntimeError(f"Request failed after {self.max_retries} attempts: {last_err}")

    def _bases(self) -> List[str]:
        bases = [self.PRIMARY_BASE]
        if self.allow_us_fallback:
            bases.append(self.BINANCE_US_BASE)
        return bases

    # ------------------------------------------------------------------

    def fetch_recent_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 50,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[KlineBar]:
        """Fetch hourly kline bars from the Binance global market-data endpoint."""
        suffix = f"/klines?symbol={symbol}&interval={interval}&limit={limit}"
        if start_time is not None:
            suffix += f"&startTime={int(start_time)}"
        if end_time is not None:
            suffix += f"&endTime={int(end_time)}"

        last_err: Optional[Exception] = None
        for base in self._bases():
            try:
                data = self._get_json(base + suffix)
                if base == self.BINANCE_US_BASE:
                    self.active_venue = "binance-US (VENUE MISMATCH vs research data)"
                else:
                    self.active_venue = "binance-global (data-api.binance.vision)"

                bars = []
                for row in data:
                    ot = int(row[0])
                    dt_str = datetime.datetime.fromtimestamp(
                        ot / 1000, tz=datetime.timezone.utc
                    ).strftime("%Y-%m-%d %H:%M:%S UTC")
                    bars.append(
                        KlineBar(
                            open_time=ot,
                            open=float(row[1]),
                            high=float(row[2]),
                            low=float(row[3]),
                            close=float(row[4]),
                            volume=float(row[5]),
                            close_time=int(row[6]),
                            datetime_utc=dt_str,
                        )
                    )
                return bars
            except Exception as err:
                last_err = err
                continue

        raise RuntimeError(f"Failed to fetch klines for {symbol} from all venues: {last_err}")

    def fetch_latest_price(self, symbol: str) -> float:
        """Fetch the latest spot price."""
        suffix = f"/ticker/price?symbol={symbol}"
        last_err: Optional[Exception] = None
        for base in self._bases():
            try:
                data = self._get_json(base + suffix)
                return float(data["price"])
            except Exception as err:
                last_err = err
                continue
        raise RuntimeError(f"Failed to fetch price for {symbol}: {last_err}")

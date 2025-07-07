from __future__ import annotations
import math
import datetime as dt
from typing import List, Tuple, Optional

from bot.polygon_client import PolygonClient
from bot.pricing import bs_price

__all__ = ["IronCondor"]

class IronCondor:
    """30‑day delta‑based iron‑condor P/L calculator for a single symbol."""

    def __init__(self, client: PolygonClient, risk_free: float = 0.02):
        self.client = client
        self.r = risk_free

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _occ(root: str, expiry: dt.date, strike: float, call: bool) -> Optional[str]:
        """Polygon OCC option ticker. Returns **None** when *strike* is NaN/None."""
        if strike is None or math.isnan(strike):
            return None
        side = "C" if call else "P"
        return f"O:{root.upper()}{expiry.strftime('%y%m%d')}{side}{int(strike * 1000):08d}"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def pl_for_date(self, sym: str, trade_date: dt.date) -> float:
        """Compute P/L for a 30‑day iron‑condor opened on *trade_date*."""
        # 1) Fetch EOD spot
        spot = self.client.spot(sym, trade_date)
        if spot is None:
            print(f"[DBG] {trade_date} no spot price")
            return 0.0

        # 2) Pick expiry = next monthly Friday ≥ 21 days out
        expiry_sel = self._next_expiry(trade_date)
        print(f"[DBG] {trade_date} expiry {expiry_sel}")

        # 3) Grab snapshot chain
        chain = self.client.snapshot_chain(sym, trade_date)
        print(f"[DBG] {trade_date} chain columns: {list(chain.columns)}")

        # --- PATCH: Robust option type and strike selection ---
        # Option type
        if "contract_type" in chain.columns:
            otype_col = "contract_type"
        elif "option_type" in chain.columns:
            otype_col = "option_type"
        elif "type" in chain.columns:
            otype_col = "type"
        else:
            raise ValueError(
                f"Option chain missing option type column: {list(chain.columns)}"
            )
        # Strike price
        if "strike_price" in chain.columns:
            strike_col = "strike_price"
        elif "strike" in chain.columns:
            strike_col = "strike"
        else:
            raise ValueError(
                f"Option chain missing strike column: {list(chain.columns)}"
            )

        calls = chain[chain[otype_col] == "call"]
        puts = chain[chain[otype_col] == "put"]
        print(f"[DBG] {trade_date} {len(calls)} calls / {len(puts)} puts")
        print(f"[DBG] {trade_date} available call strikes: {list(calls[strike_col])[:10]}")
        print(f"[DBG] {trade_date} available put strikes: {list(puts[strike_col])[:10]}")

        if calls.empty or puts.empty:
            return 0.0

        # --- STRICT STRIKE SELECTION PATCH ---
        # Use sorted available strikes from the chain
        call_strikes = calls[strike_col].sort_values().to_numpy()
        put_strikes = puts[strike_col].sort_values().to_numpy()

        # Use ATM (closest to spot) for shorts, with fallback to nearest neighbor for longs
        short_call_k = call_strikes[(abs(call_strikes - spot)).argmin()]
        short_put_k = put_strikes[(abs(put_strikes - spot)).argmin()]

        # Long call = next higher strike, long put = next lower strike
        long_call_k = next((k for k in call_strikes if k > short_call_k), short_call_k)
        long_put_k = next((k for k in reversed(put_strikes) if k < short_put_k), short_put_k)

        ks = [short_call_k, short_put_k, long_call_k, long_put_k]
        if any(k is None or math.isnan(k) for k in ks):
            print(f"[DBG] {trade_date} strike selection failed → skip")
            return 0.0

        print(f"[DBG] {trade_date} strikes {long_call_k} {short_call_k} {short_put_k} {long_put_k}")

        # 6) Build leg tuples (ticker, strike, call?)
        legs: List[Tuple[str, float, bool]] = [
            (self._occ(sym, expiry_sel, short_call_k, True), short_call_k, True),   # sell call
            (self._occ(sym, expiry_sel, long_call_k, True), long_call_k, True),     # buy further call
            (self._occ(sym, expiry_sel, short_put_k, False), short_put_k, False),   # sell put
            (self._occ(sym, expiry_sel, long_put_k, False), long_put_k, False),     # buy further put
        ]

        # PATCH: Print OCC tickers used
        print(f"[DBG] {trade_date} OCC tickers: {[t for t, *_ in legs]}")

        # If any OCC ticker is None (because of NaN), abandon trade.
        if any(t is None for t, *_ in legs):
            print(f"[DBG] {trade_date} OCC generation failed → skip")
            return 0.0

        # 7) Open @ trade_date close, close @ expiry close.
        open_px = [self.client.agg_close(tkr, trade_date) for tkr, *_ in legs]
        close_px = [self.client.agg_close(tkr, expiry_sel) for tkr, *_ in legs]

        print(f"[DBG] {trade_date} prices {open_px} {close_px}")

        if any(px is None for px in open_px + close_px):
            return 0.0  # incomplete data → skip

        # 8) P/L = credit received – debit paid back (calls positive, puts positive)
        credit = open_px[0] + open_px[2] - open_px[1] - open_px[3]
        debit = close_px[1] + close_px[3] - close_px[0] - close_px[2]
        return credit - debit

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _next_expiry(trade_date: dt.date) -> dt.date:
        """Next Friday ≥ 21 calendar days away."""
        d = trade_date + dt.timedelta(days=21)
        # roll forward to Friday
        while d.weekday() != 4:  # 0=Mon … 4=Fri
            d += dt.timedelta(days=1)
        return d

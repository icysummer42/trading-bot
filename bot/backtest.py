"""Backtest driver that rolls weekly 30‑day iron condors."""
from __future__ import annotations
import datetime as dt
import time

from bot.polygon_client import PolygonClient
from bot.strategy.iron_condor import IronCondor


class OptionBacktester:
    def __init__(self, cfg, dp, sg):
        self.cfg, self.dp, self.sg = cfg, dp, sg
        self.poly = PolygonClient(cfg.polygon_key)

    def run(self):
        start_date = dt.date.today() - dt.timedelta(days=730)
        for sym in self.cfg.symbols_equity:
            pnl = 0.0
            d = start_date
            strat = IronCondor(self.poly, self.dp)
            while d < dt.date.today() - dt.timedelta(days=30):
                if d.weekday() < 5:  # Only run for Mon-Fri
                    pnl += strat.pl_for_date(sym, d)
                d += dt.timedelta(days=1)
            print(f"{sym}: total P/L ${pnl:,.0f}")
        print("Option back‑test complete ✓")

"""Backtest driver that rolls weekly 30‑day iron condors."""
from __future__ import annotations
import datetime as dt
import concurrent.futures
import time
from tqdm import tqdm
from bot.polygon_client import PolygonClient
from bot.strategy.iron_condor import IronCondor


class OptionBacktester:
    def __init__(self, cfg, dp, sg):
        self.cfg, self.dp, self.sg = cfg, dp, sg
        self.poly = PolygonClient(cfg.polygon_key)

    def run(self):
        start_date = dt.date.today() - dt.timedelta(days=730)
        end_date = dt.date.today() - dt.timedelta(days=30)

        for sym in self.cfg.symbols_equity:
            dates = [
                start_date + dt.timedelta(days=i)
                for i in range((end_date - start_date).days)
                if (start_date + dt.timedelta(days=i)).weekday() < 5
            ]
            strat = IronCondor(self.poly, self.dp)

            def pl_for(d):
                return strat.pl_for_date(sym, d)

            pnl = 0.0
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                # executor.map returns results in order of 'dates'
                results = list(executor.map(pl_for, dates))
                pnl = sum(results)
            print(f"{sym}: total P/L ${pnl:,.0f}")
        print("Option back‑test complete ✓")

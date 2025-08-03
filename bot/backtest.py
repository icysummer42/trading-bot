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

    def run(self, use_parallel=True, max_workers=4):
        start_date = dt.date.today() - dt.timedelta(days=730)
        end_date = dt.date.today() - dt.timedelta(days=30)
        total_trades = 0
        start_wall = time.time()

        for sym in self.cfg.symbols_equity:
            # Build a list of all valid trading dates (Mon-Fri)
            dates = [
                start_date + dt.timedelta(days=i)
                for i in range((end_date - start_date).days)
                if (start_date + dt.timedelta(days=i)).weekday() < 5
            ]
            strat = IronCondor(self.poly, self.dp)
            pnl = 0.0

            if use_parallel:
                # Parallel execution with progress bar
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(tqdm(
                        executor.map(lambda d: strat.pl_for_date(sym, d), dates),
                        total=len(dates), desc=f"{sym} backtest", unit="trades"
                    ))
                    pnl = sum(results)
                    total_trades += len(results)
            else:
                # Serial execution with progress bar
                for d in tqdm(dates, desc=f"{sym} backtest", unit="trades"):
                    pnl += strat.pl_for_date(sym, d)
                    total_trades += 1

            print(f"{sym}: total P/L ${pnl:,.0f}")

        elapsed = time.time() - start_wall
        print("\n========== BACKTEST SUMMARY ==========")
        print(f"Total trades processed: {total_trades:,}")
        print(f"Total wall time: {elapsed:.1f} sec ({elapsed/60:.1f} min)")
        print("Option back‑test complete ✓")

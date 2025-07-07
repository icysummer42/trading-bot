"""CLI entry point for Quant‑Options‑Bot (slim version)."""
from __future__ import annotations

import argparse

from bot.backtest import OptionBacktester
from bot.engine import ExecutionEngine, RiskManager, StrategyEngine
from config import Config
from data_pipeline import DataPipeline
from signal_generator import SignalGenerator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["backtest", "live"], default="backtest")
    ap.add_argument("--once", action="store_true", help="run one live iteration and exit")
    args = ap.parse_args()

    cfg = Config()
    dp = DataPipeline(cfg)
    sg = SignalGenerator(cfg, dp)

    if args.mode == "backtest":
        OptionBacktester(cfg, dp, sg).run()
        return

    # --- basic live loop (placeholder) ---
    strat = StrategyEngine(cfg)
    exe = ExecutionEngine()
    risk = RiskManager()

    edge = 0.0  # placeholder edge score
    for sym in cfg.symbols_equity:
        for trade in strat.generate(edge, sym):
            size = risk.size(edge, capital=1_000_000)
            exe.place({**trade, "size": size})
        if args.once:
            break


if __name__ == "__main__":
    main()

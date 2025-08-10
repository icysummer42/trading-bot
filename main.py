"""CLI entry point for Quant‑Options‑Bot (enhanced with advanced risk management)."""
from __future__ import annotations

import argparse
import numpy as np

from bot.backtest import OptionBacktester
from bot.engine import ExecutionEngine, RiskManager, StrategyEngine
from bot.enhanced_engine import StrategyEngine as EnhancedStrategyEngine, ExecutionEngine as EnhancedExecutionEngine
from bot.risk_manager import AdvancedRiskManager
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

    # --- enhanced live loop with advanced risk management ---
    strat = EnhancedStrategyEngine(cfg)
    exe = EnhancedExecutionEngine(cfg)
    risk_manager = AdvancedRiskManager(cfg)

    for sym in cfg.symbols_equity:
        # Get current market data and signal
        close_series = dp.get_close_series(sym)
        if close_series.empty:
            print(f"[WARN] No market data for {sym}, skipping")
            continue
            
        edge = sg.get_signal_score(sym, close_series)
        if not isinstance(edge, (int, float)) or np.isnan(edge):
            print(f"[WARN] Invalid edge score for {sym}: {edge}")
            continue
        
        print(f"[INFO] {sym} edge score: {edge:.3f}")
        
        # Prepare market data
        current_price = float(close_series.iloc[-1])
        volatility = sg.forecast_volatility(close_series)
        
        market_data = {
            'spot_price': current_price,
            'volatility': volatility,
            'last_close': current_price
        }
        
        # Generate and execute trades with enhanced risk management
        trades = strat.generate(edge, sym, market_data)
        for trade in trades:
            exe.place(trade)
        
        # Update portfolio positions
        exe.update_positions({sym: current_price})
        
        # Generate risk report
        if args.once:
            risk_report = risk_manager.generate_risk_report()
            print(f"\n=== RISK REPORT ===")
            print(f"Portfolio Value: ${risk_report['portfolio_value']:,.0f}")
            print(f"VaR (95%): {risk_report['risk_metrics']['var_95']:.3f}")
            print(f"Current Drawdown: {risk_report['risk_metrics']['current_drawdown']:.2%}")
            print(f"Concentration Risk: {risk_report['risk_metrics']['concentration_risk']:.3f}")
            break


if __name__ == "__main__":
    main()

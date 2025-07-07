"""Execution‑layer stubs split from main.py to keep each file short."""

from __future__ import annotations
from typing import Any, Dict


class StrategyEngine:
    """Very simple edge‑to‑strategy mapper."""

    def __init__(self, cfg):
        self.cfg = cfg

    def generate(self, edge: float, sym: str):
        if abs(edge) < 0.2:
            return [{"strategy": "iron_condor", "symbol": sym}]
        return [{"strategy": "covered_call" if edge > 0 else "put_spread", "symbol": sym}]


class ExecutionEngine:
    """Placeholder that simply prints the trade dict."""

    def place(self, trade: Dict[str, Any]):
        print("[EXECUTE]", trade)


class RiskManager:
    """Calculates position size via a fixed 1 % capital rule."""

    def size(self, edge: float, capital: float):
        return capital * 0.01

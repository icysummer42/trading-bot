from __future__ import annotations
import math
from statistics import NormalDist

_N = NormalDist().cdf

def bs_price(spot: float, strike: float, iv: float, tau: float, r: float, call: bool) -> float:
    """Black‑Scholes option price (continuous compounding, no dividend).
    Falls back to intrinsic value when time‑to‑expiry is zero."""
    if tau <= 0.0:
        return max(0.0, spot - strike) if call else max(0.0, strike - spot)
    iv = max(iv, 1e-6)  # guard against division by zero
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * tau) / (iv * math.sqrt(tau))
    d2 = d1 - iv * math.sqrt(tau)
    if call:
        return spot * _N(d1) - strike * math.exp(-r * tau) * _N(d2)
    return strike * math.exp(-r * tau) * _N(-d2) - spot * _N(-d1)

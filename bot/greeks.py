"""
Options Greeks calculation module for risk management.

Calculates Delta, Gamma, Theta, Vega, and Rho for options positions.
"""

from __future__ import annotations
import math
from typing import Dict, Tuple
from statistics import NormalDist
from bot.pricing import bs_price

_N = NormalDist().cdf
_n = NormalDist().pdf

class GreeksCalculator:
    """Calculate Greeks for options positions."""
    
    def __init__(self):
        self.epsilon = 1e-6  # Small value for numerical differentiation
    
    def _d1_d2(self, spot: float, strike: float, iv: float, tau: float, r: float) -> Tuple[float, float]:
        """Calculate d1 and d2 for Black-Scholes formula."""
        if tau <= 0:
            return 0.0, 0.0
        
        iv = max(iv, self.epsilon)  # Avoid division by zero
        d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * tau) / (iv * math.sqrt(tau))
        d2 = d1 - iv * math.sqrt(tau)
        return d1, d2
    
    def delta(self, spot: float, strike: float, iv: float, tau: float, 
              r: float, is_call: bool) -> float:
        """
        Calculate Delta (price sensitivity to underlying price change).
        
        Args:
            spot: Current underlying price
            strike: Strike price
            iv: Implied volatility (annual)
            tau: Time to expiration (years)
            r: Risk-free rate
            is_call: True for call, False for put
        
        Returns:
            Delta value
        """
        if tau <= 0:
            if is_call:
                return 1.0 if spot > strike else 0.0
            else:
                return -1.0 if spot < strike else 0.0
        
        d1, _ = self._d1_d2(spot, strike, iv, tau, r)
        
        if is_call:
            return _N(d1)
        else:
            return _N(d1) - 1.0
    
    def gamma(self, spot: float, strike: float, iv: float, tau: float, r: float) -> float:
        """
        Calculate Gamma (rate of change of Delta).
        
        Returns:
            Gamma value (same for calls and puts)
        """
        if tau <= 0:
            return 0.0
        
        d1, _ = self._d1_d2(spot, strike, iv, tau, r)
        iv = max(iv, self.epsilon)
        
        return _n(d1) / (spot * iv * math.sqrt(tau))
    
    def theta(self, spot: float, strike: float, iv: float, tau: float, 
              r: float, is_call: bool) -> float:
        """
        Calculate Theta (time decay, negative for long positions).
        
        Returns:
            Daily theta (divide annual theta by 365)
        """
        if tau <= 0:
            return 0.0
        
        d1, d2 = self._d1_d2(spot, strike, iv, tau, r)
        iv = max(iv, self.epsilon)
        
        # Common term
        term1 = -(spot * _n(d1) * iv) / (2 * math.sqrt(tau))
        
        if is_call:
            term2 = -r * strike * math.exp(-r * tau) * _N(d2)
            theta_annual = term1 + term2
        else:
            term2 = r * strike * math.exp(-r * tau) * _N(-d2)
            theta_annual = term1 + term2
        
        # Convert to daily theta
        return theta_annual / 365.0
    
    def vega(self, spot: float, strike: float, iv: float, tau: float, r: float) -> float:
        """
        Calculate Vega (sensitivity to volatility change).
        
        Returns:
            Vega per 1% change in volatility
        """
        if tau <= 0:
            return 0.0
        
        d1, _ = self._d1_d2(spot, strike, iv, tau, r)
        
        # Vega is the same for calls and puts
        vega_per_100bp = spot * _n(d1) * math.sqrt(tau)
        
        # Return vega per 1% (100 basis points) change
        return vega_per_100bp / 100.0
    
    def rho(self, spot: float, strike: float, iv: float, tau: float, 
            r: float, is_call: bool) -> float:
        """
        Calculate Rho (sensitivity to interest rate change).
        
        Returns:
            Rho per 1% change in interest rates
        """
        if tau <= 0:
            return 0.0
        
        _, d2 = self._d1_d2(spot, strike, iv, tau, r)
        
        if is_call:
            rho_per_100bp = strike * tau * math.exp(-r * tau) * _N(d2)
        else:
            rho_per_100bp = -strike * tau * math.exp(-r * tau) * _N(-d2)
        
        # Return rho per 1% change in rates
        return rho_per_100bp / 100.0
    
    def calculate_all_greeks(self, spot: float, strike: float, iv: float, 
                           tau: float, r: float, is_call: bool) -> Dict[str, float]:
        """
        Calculate all Greeks at once for efficiency.
        
        Returns:
            Dictionary with all Greeks
        """
        if tau <= 0:
            return {
                'delta': 1.0 if (is_call and spot > strike) or (not is_call and spot < strike) else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
        
        d1, d2 = self._d1_d2(spot, strike, iv, tau, r)
        iv = max(iv, self.epsilon)
        
        # Delta
        if is_call:
            delta_val = _N(d1)
        else:
            delta_val = _N(d1) - 1.0
        
        # Gamma (same for calls and puts)
        gamma_val = _n(d1) / (spot * iv * math.sqrt(tau))
        
        # Theta
        term1 = -(spot * _n(d1) * iv) / (2 * math.sqrt(tau))
        if is_call:
            term2 = -r * strike * math.exp(-r * tau) * _N(d2)
        else:
            term2 = r * strike * math.exp(-r * tau) * _N(-d2)
        theta_val = (term1 + term2) / 365.0  # Daily theta
        
        # Vega (same for calls and puts)
        vega_val = (spot * _n(d1) * math.sqrt(tau)) / 100.0
        
        # Rho
        if is_call:
            rho_val = (strike * tau * math.exp(-r * tau) * _N(d2)) / 100.0
        else:
            rho_val = (-strike * tau * math.exp(-r * tau) * _N(-d2)) / 100.0
        
        return {
            'delta': delta_val,
            'gamma': gamma_val,
            'theta': theta_val,
            'vega': vega_val,
            'rho': rho_val
        }
    
    def portfolio_greeks(self, positions: List[Dict]) -> Dict[str, float]:
        """
        Calculate portfolio-level Greeks by summing position Greeks.
        
        Args:
            positions: List of position dictionaries with:
                - quantity: Number of contracts (positive for long, negative for short)
                - spot: Current underlying price
                - strike: Strike price
                - iv: Implied volatility
                - tau: Time to expiration (years)
                - r: Risk-free rate
                - is_call: Boolean for call/put
        
        Returns:
            Portfolio Greeks dictionary
        """
        portfolio_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
        
        for position in positions:
            quantity = position.get('quantity', 0)
            if quantity == 0:
                continue
            
            greeks = self.calculate_all_greeks(
                spot=position['spot'],
                strike=position['strike'],
                iv=position['iv'],
                tau=position['tau'],
                r=position.get('r', 0.02),
                is_call=position['is_call']
            )
            
            # Scale by quantity and add to portfolio
            for greek_name, greek_value in greeks.items():
                portfolio_greeks[greek_name] += quantity * greek_value
        
        return portfolio_greeks
    
    def implied_volatility(self, market_price: float, spot: float, strike: float,
                         tau: float, r: float, is_call: bool, 
                         initial_guess: float = 0.2, tolerance: float = 1e-6,
                         max_iterations: int = 100) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            market_price: Observed market price
            spot: Current underlying price  
            strike: Strike price
            tau: Time to expiration (years)
            r: Risk-free rate
            is_call: True for call, False for put
            initial_guess: Starting volatility guess
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
        
        Returns:
            Implied volatility
        """
        if tau <= 0:
            return 0.0
        
        iv = initial_guess
        
        for _ in range(max_iterations):
            # Calculate price and vega at current iv
            theoretical_price = bs_price(spot, strike, iv, tau, r, is_call)
            price_diff = theoretical_price - market_price
            
            if abs(price_diff) < tolerance:
                return iv
            
            # Calculate vega for Newton-Raphson step
            vega_val = self.vega(spot, strike, iv, tau, r) * 100  # Vega per 100bp
            
            if abs(vega_val) < self.epsilon:
                break  # Avoid division by zero
            
            # Newton-Raphson update
            iv = iv - price_diff / vega_val
            
            # Ensure iv stays positive
            iv = max(iv, self.epsilon)
        
        return iv
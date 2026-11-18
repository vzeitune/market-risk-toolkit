"""Black-Scholes pricing, Greeks and a discrete delta-hedging simulation."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bs_call(S, K, T, r, sigma):
    """European call price and delta. Works elementwise on arrays."""
    S, T = np.asarray(S, dtype=float), np.asarray(T, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    expired = T <= 0
    price = np.where(expired, np.maximum(S - K, 0.0), price)
    delta = np.where(expired, (S > K).astype(float), delta)
    return price, delta


def bs_greeks(S, K, T, r, sigma) -> dict:
    """Delta, gamma, vega (per 1.00 vol) and theta (per year) of a European call."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return {
        "delta": norm.cdf(d1),
        "gamma": norm.pdf(d1) / (S * sigma * np.sqrt(T)),
        "vega": S * norm.pdf(d1) * np.sqrt(T),
        "theta": -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2),
    }


def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, seed=42) -> np.ndarray:
    """GBM price paths, shape (n_paths, n_steps + 1)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_paths = np.cumsum((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z, axis=1)
    return S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))


def delta_hedge_pnl(paths, K, T, r, sigma, rebalance_every: int = 1) -> np.ndarray:
    """P&L of selling one call at the BS price and delta-hedging it.

    The hedge is rebalanced every ``rebalance_every`` steps and financed at ``r``.
    With continuous rebalancing and the true volatility the P&L tends to zero;
    the spread of the P&L shows the cost of discrete hedging.
    Returns P&L per option as a fraction of the initial option premium.
    """
    n_paths, n_cols = paths.shape
    n_steps = n_cols - 1
    dt = T / n_steps
    premium, delta = bs_call(paths[:, 0], K, T, r, sigma)
    cash = premium - delta * paths[:, 0]
    shares = delta
    for i in range(1, n_steps + 1):
        cash = cash * np.exp(r * dt)
        if i % rebalance_every == 0 and i < n_steps:
            _, new_delta = bs_call(paths[:, i], K, T - i * dt, r, sigma)
            cash -= (new_delta - shares) * paths[:, i]
            shares = new_delta
    payoff = np.maximum(paths[:, -1] - K, 0.0)
    pnl = cash + shares * paths[:, -1] - payoff
    return pnl / premium

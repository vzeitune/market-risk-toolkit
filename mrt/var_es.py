"""Value at Risk and Expected Shortfall.

Conventions
-----------
* Inputs are returns (e.g. daily log returns), as a 1-D array.
* VaR and ES are reported as positive numbers = losses, as a fraction of
  portfolio value. Multiply by the portfolio value for a currency amount.
* ``alpha`` is the tail probability, so ``alpha=0.01`` is 99% VaR.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def historical_var_es(returns, alpha: float = 0.01) -> tuple[float, float]:
    """Historical simulation: empirical quantile and mean of the tail beyond it."""
    r = np.asarray(returns, dtype=float)
    q = np.quantile(r, alpha)
    tail = r[r <= q]
    return -q, -tail.mean()


def normal_var_es(returns=None, alpha: float = 0.01, mu: float | None = None,
                  sigma: float | None = None) -> tuple[float, float]:
    """Parametric VaR/ES under a normal distribution.

    Either pass ``returns`` (mean and sample std are estimated) or ``mu``/``sigma``.
    ES uses the closed form  ES = -(mu - sigma * phi(z) / alpha).
    """
    if returns is not None:
        r = np.asarray(returns, dtype=float)
        mu, sigma = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(alpha)
    var = -(mu + z * sigma)
    es = -(mu - sigma * stats.norm.pdf(z) / alpha)
    return var, es


def student_t_var_es(returns=None, alpha: float = 0.01, mu: float | None = None,
                     sigma: float | None = None, nu: float | None = None) -> tuple[float, float]:
    """Parametric VaR/ES under a Student-t with ``nu`` degrees of freedom.

    ``sigma`` is the standard deviation of returns (not the t scale). If
    ``returns`` is given and ``nu`` is None, ``nu`` is fitted by maximum likelihood.
    """
    if returns is not None:
        r = np.asarray(returns, dtype=float)
        mu, sigma = r.mean(), r.std(ddof=1)
        if nu is None:
            nu, _, _ = stats.t.fit(r)
    if nu is None or nu <= 2:
        raise ValueError("nu must be > 2 for a finite variance")
    scale = sigma * np.sqrt((nu - 2) / nu)          # t scale giving std = sigma
    t_a = stats.t.ppf(alpha, nu)
    var = -(mu + scale * t_a)
    es = -mu + scale * stats.t.pdf(t_a, nu) / alpha * (nu + t_a ** 2) / (nu - 1)
    return var, es


def monte_carlo_var_es(returns=None, alpha: float = 0.01, n_sims: int = 100_000,
                       dist: str = "normal", seed: int | None = 42,
                       mu: float | None = None, sigma: float | None = None) -> tuple[float, float]:
    """Monte Carlo VaR/ES: simulate returns from a fitted normal or Student-t."""
    rng = np.random.default_rng(seed)
    nu = None
    if returns is not None:
        r = np.asarray(returns, dtype=float)
        mu, sigma = r.mean(), r.std(ddof=1)
        if dist == "t":
            nu, _, _ = stats.t.fit(r)
    if dist == "normal":
        sims = rng.normal(mu, sigma, n_sims)
    elif dist == "t":
        nu = nu or 5.0
        sims = mu + sigma * np.sqrt((nu - 2) / nu) * rng.standard_t(nu, n_sims)
    else:
        raise ValueError("dist must be 'normal' or 't'")
    return historical_var_es(sims, alpha)


def ewma_volatility(returns, lam: float = 0.94, init_window: int = 30) -> np.ndarray:
    """RiskMetrics EWMA volatility forecast for each date.

    ``out[t]`` is the forecast for day t made with returns up to t-1:
        sigma2[t] = lam * sigma2[t-1] + (1 - lam) * r[t-1]**2
    """
    r = np.asarray(returns, dtype=float)
    s2 = np.empty_like(r)
    s2[0] = np.var(r[:init_window])
    for t in range(1, len(r)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * r[t - 1] ** 2
    return np.sqrt(s2)


def rolling_var(returns, window: int = 500, alpha: float = 0.01,
                method: str = "historical", lam: float = 0.94) -> np.ndarray:
    """One-day-ahead VaR forecast for each day after the first ``window`` days.

    Returns an array the same length as ``returns``; the first ``window`` entries
    are NaN. ``method`` is one of 'historical', 'normal', 'ewma'.
    """
    r = np.asarray(returns, dtype=float)
    out = np.full(len(r), np.nan)
    if method == "ewma":
        vol = ewma_volatility(r, lam)
        z = stats.norm.ppf(alpha)
        out[window:] = -z * vol[window:]
        return out
    for t in range(window, len(r)):
        hist = r[t - window:t]
        if method == "historical":
            out[t] = historical_var_es(hist, alpha)[0]
        elif method == "normal":
            out[t] = normal_var_es(hist, alpha)[0]
        else:
            raise ValueError("method must be 'historical', 'normal' or 'ewma'")
    return out

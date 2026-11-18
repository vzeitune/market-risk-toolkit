"""Operational risk capital with the Loss Distribution Approach (LDA).

Annual loss L = sum of N individual losses, where
    N ~ Poisson(lambda)              (frequency)
    X_i ~ Lognormal(mu, sigma)       (severity)
Capital is read off the simulated distribution of L at 99.9%.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LDAResult:
    annual_losses: np.ndarray
    expected_loss: float
    var: float
    es: float
    confidence: float

    @property
    def unexpected_loss(self) -> float:
        return self.var - self.expected_loss


def simulate_annual_losses(lam: float, mu: float, sigma: float, n_sims: int = 100_000,
                           seed: int | None = 42, chunk: int = 20_000) -> np.ndarray:
    """Simulate ``n_sims`` annual aggregate losses (vectorised, in chunks to cap memory)."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_sims)
    for start in range(0, n_sims, chunk):
        m = min(chunk, n_sims - start)
        counts = rng.poisson(lam, m)
        sev = rng.lognormal(mu, sigma, counts.sum())
        # sum the severities belonging to each simulated year
        idx = np.repeat(np.arange(m), counts)
        out[start:start + m] = np.bincount(idx, weights=sev, minlength=m)
    return out


def lda_capital(lam: float, mu: float, sigma: float, confidence: float = 0.999,
                n_sims: int = 100_000, seed: int | None = 42) -> LDAResult:
    losses = simulate_annual_losses(lam, mu, sigma, n_sims, seed)
    var = np.quantile(losses, confidence)
    es = losses[losses >= var].mean()
    return LDAResult(losses, losses.mean(), var, es, confidence)


def scale_severity_mu(mu_sample: float, firm_size: float, sample_size: float) -> float:
    """Shift log-severity for firm size: mu + ln(firm/sample).

    Equivalent to multiplying every loss by firm/sample (a simple size scaling).
    """
    return mu_sample + np.log(firm_size / sample_size)

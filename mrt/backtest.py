"""VaR backtesting: Kupiec, Christoffersen and the Basel traffic light."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.special import xlogy


def exceptions(returns, var) -> np.ndarray:
    """Boolean array: True where the realised loss exceeded the VaR forecast.

    NaN forecasts (e.g. the estimation window) are dropped.
    """
    r = np.asarray(returns, dtype=float)
    v = np.asarray(var, dtype=float)
    mask = ~np.isnan(v)
    return (-r[mask]) > v[mask]


def _loglik_bernoulli(n0: int, n1: int, p: float) -> float:
    return xlogy(n0, 1 - p) + xlogy(n1, p)


def kupiec_pof(exc, alpha: float = 0.01) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures test. Returns (LR statistic, p-value).

    H0: the exception rate equals ``alpha``.  LR ~ chi2(1).
    """
    exc = np.asarray(exc, dtype=bool)
    T, x = len(exc), int(exc.sum())
    p_hat = x / T
    lr = -2 * (_loglik_bernoulli(T - x, x, alpha) - _loglik_bernoulli(T - x, x, p_hat))
    return lr, 1 - stats.chi2.cdf(lr, 1)


def christoffersen_independence(exc) -> tuple[float, float]:
    """Christoffersen (1998) independence test. Returns (LR statistic, p-value).

    H0: an exception today is no more likely after an exception yesterday
    (no clustering). LR ~ chi2(1).
    """
    e = np.asarray(exc, dtype=int)
    prev, curr = e[:-1], e[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    pi0 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    l_null = _loglik_bernoulli(n00 + n10, n01 + n11, pi)
    l_alt = _loglik_bernoulli(n00, n01, pi0) + _loglik_bernoulli(n10, n11, pi1)
    lr = -2 * (l_null - l_alt)
    return lr, 1 - stats.chi2.cdf(lr, 1)


def basel_traffic_light(n_exceptions: int, n_obs: int = 250, alpha: float = 0.01) -> str:
    """Basel traffic-light zone from the binomial distribution of exceptions.

    Green while the cumulative probability is below 95%, yellow below 99.99%,
    red above. For 250 days at 99% this gives green 0-4, yellow 5-9, red 10+.
    """
    cum = stats.binom.cdf(n_exceptions, n_obs, alpha)
    if cum < 0.95:
        return "green"
    if cum < 0.9999:
        return "yellow"
    return "red"


@dataclass
class BacktestResult:
    method: str
    n_obs: int
    n_exceptions: int
    expected: float
    kupiec_p: float
    christoffersen_p: float
    traffic_light_last_250: str

    def as_row(self) -> dict:
        return {
            "Method": self.method,
            "Days": self.n_obs,
            "Exceptions": self.n_exceptions,
            "Expected": round(self.expected, 1),
            "Kupiec p": round(self.kupiec_p, 3),
            "Christoffersen p": round(self.christoffersen_p, 3),
            "Traffic light (last 250d)": self.traffic_light_last_250,
        }


def backtest(returns, var, alpha: float = 0.01, method: str = "") -> BacktestResult:
    exc = exceptions(returns, var)
    last = exc[-250:]
    return BacktestResult(
        method=method,
        n_obs=len(exc),
        n_exceptions=int(exc.sum()),
        expected=alpha * len(exc),
        kupiec_p=kupiec_pof(exc, alpha)[1],
        christoffersen_p=christoffersen_independence(exc)[1],
        traffic_light_last_250=basel_traffic_light(int(last.sum()), len(last), alpha),
    )

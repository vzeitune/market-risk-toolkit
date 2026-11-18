import numpy as np
import pytest

from mrt import backtest as bt
from mrt import oprisk_lda as lda
from mrt import options as opt
from mrt import var_es as ve


def test_normal_var_es_closed_form():
    var, es = ve.normal_var_es(alpha=0.01, mu=0.0, sigma=1.0)
    assert var == pytest.approx(2.32635, abs=1e-4)
    assert es == pytest.approx(2.66521, abs=1e-4)


def test_student_t_matches_monte_carlo():
    var, es = ve.student_t_var_es(alpha=0.01, mu=0.0, sigma=0.01, nu=5)
    mc_var, mc_es = ve.monte_carlo_var_es(alpha=0.01, mu=0.0, sigma=0.01, dist="t",
                                          n_sims=2_000_000, seed=1)
    assert mc_var == pytest.approx(var, rel=0.02)
    assert mc_es == pytest.approx(es, rel=0.02)


def test_t_es_exceeds_normal_es_for_same_volatility():
    _, es_n = ve.normal_var_es(alpha=0.01, mu=0.0, sigma=0.01)
    _, es_t = ve.student_t_var_es(alpha=0.01, mu=0.0, sigma=0.01, nu=4)
    assert es_t > es_n


def test_historical_es_at_least_var():
    r = np.random.default_rng(0).standard_t(4, 5_000) * 0.01
    var, es = ve.historical_var_es(r, 0.01)
    assert es >= var > 0


def test_ewma_uses_only_past_returns():
    r = np.zeros(100)
    r[50] = 0.1
    vol = ve.ewma_volatility(r, lam=0.94, init_window=10)
    assert vol[50] == pytest.approx(vol[49] * np.sqrt(0.94))   # shock not yet seen
    assert vol[51] > vol[50]                                    # shock seen next day


def test_kupiec_and_christoffersen():
    exc = np.zeros(1000, dtype=bool)
    exc[::100] = True                                   # exactly 1% and evenly spaced
    assert bt.kupiec_pof(exc, 0.01)[1] == pytest.approx(1.0)
    assert bt.christoffersen_independence(exc)[1] > 0.5
    clustered = np.zeros(1000, dtype=bool)
    clustered[500:510] = True                           # 1% but all in a row
    assert bt.christoffersen_independence(clustered)[1] < 0.001
    too_many = np.zeros(250, dtype=bool)
    too_many[:10] = True
    assert bt.kupiec_pof(too_many, 0.01)[1] < 0.01


@pytest.mark.parametrize("n, zone", [(0, "green"), (4, "green"), (5, "yellow"),
                                     (9, "yellow"), (10, "red")])
def test_basel_traffic_light_zones(n, zone):
    assert bt.basel_traffic_light(n, 250, 0.01) == zone


def test_lda_mean_matches_theory():
    lam, mu, sigma = 50, 10.0, 1.0
    res = lda.lda_capital(lam, mu, sigma, n_sims=200_000, seed=3)
    theory = lam * np.exp(mu + sigma ** 2 / 2)
    assert res.expected_loss == pytest.approx(theory, rel=0.01)
    assert res.es >= res.var > res.expected_loss


def test_black_scholes_reference_value():
    price, delta = opt.bs_call(100, 100, 1.0, 0.05, 0.2)
    assert float(price) == pytest.approx(10.4506, abs=1e-4)
    assert float(delta) == pytest.approx(0.6368, abs=1e-4)


def test_delta_hedge_error_shrinks_with_rebalancing():
    paths = opt.simulate_gbm(100, 0.05, 0.2, 1.0, 252, 4_000, seed=7)
    daily = opt.delta_hedge_pnl(paths, 100, 1.0, 0.02, 0.2, rebalance_every=1)
    weekly = opt.delta_hedge_pnl(paths, 100, 1.0, 0.02, 0.2, rebalance_every=5)
    assert abs(daily.mean()) < 0.02
    assert daily.std() < weekly.std()

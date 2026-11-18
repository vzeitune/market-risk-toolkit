"""Compare three VaR models and backtest them.

By default the returns are simulated from a GARCH(1,1) process with Student-t
shocks, which has the two features that break simple VaR models: fat tails and
volatility clustering. To use real data instead, pass a CSV with a 'close' column:

    python examples/01_var_backtest.py prices.csv
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mrt import backtest as bt
from mrt import var_es as ve

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)
ALPHA, WINDOW = 0.01, 500


def simulate_garch_t(n=3000, omega=2e-6, a=0.08, b=0.90, nu=5, seed=11):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)   # unit-variance t shocks
    r, s2 = np.empty(n), omega / (1 - a - b)
    for t in range(n):
        r[t] = np.sqrt(s2) * z[t]
        s2 = omega + a * r[t] ** 2 + b * s2
    return r


if len(sys.argv) > 1:
    prices = pd.read_csv(sys.argv[1])["close"].to_numpy()
    returns, source = np.diff(np.log(prices)), Path(sys.argv[1]).name
else:
    returns, source = simulate_garch_t(), "simulated GARCH(1,1)-t returns"

forecasts = {m: ve.rolling_var(returns, WINDOW, ALPHA, m) for m in ("historical", "normal", "ewma")}
labels = {"historical": "Historical simulation (500d)", "normal": "Normal (500d)",
          "ewma": "EWMA normal (lambda=0.94)"}

table = pd.DataFrame([bt.backtest(returns, v, ALPHA, labels[m]).as_row() for m, v in forecasts.items()])
print(f"99% 1-day VaR backtest on {source}\n")
print(table.to_string(index=False))
table.to_csv(OUT / "01_var_backtest_results.csv", index=False)

var_hist, es_hist = ve.historical_var_es(returns, ALPHA)
var_n, es_n = ve.normal_var_es(returns, ALPHA)
var_t, es_t = ve.student_t_var_es(returns, ALPHA)
print("\nFull-sample 99% VaR / ES (% of portfolio):")
for name, v, e in [("Historical", var_hist, es_hist), ("Normal", var_n, es_n), ("Student-t", var_t, es_t)]:
    print(f"  {name:<11} VaR {v:6.2%}   ES {e:6.2%}")

fig, ax = plt.subplots(figsize=(12, 5))
days = np.arange(len(returns))
ax.plot(days, -returns, lw=0.5, color="grey", label="Daily loss")
for m, colour in zip(forecasts, ("tab:blue", "tab:orange", "tab:green")):
    ax.plot(days, forecasts[m], lw=1, color=colour, label=labels[m])
exc = (-returns > forecasts["ewma"])
ax.scatter(days[exc], -returns[exc], s=10, color="red", zorder=3, label="EWMA exceptions")
ax.set(title="99% VaR forecasts vs realised losses", xlabel="Day", ylabel="Loss (fraction of value)")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "01_var_backtest.png", dpi=150)
print(f"\nSaved figures/01_var_backtest.png and figures/01_var_backtest_results.csv")

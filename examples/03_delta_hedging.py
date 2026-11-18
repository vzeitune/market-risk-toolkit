"""How well does discrete delta hedging replicate a short call?

Sell a 1-year at-the-money call at its Black-Scholes price, hedge with the
underlying, and measure the hedging error across 10,000 simulated paths for
different rebalancing frequencies, and when the volatility used to hedge is wrong.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mrt import options as opt

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.03, 0.20
paths = opt.simulate_gbm(S0, 0.06, SIGMA, T, 252, 10_000, seed=5)

rows, pnls = [], {}
for label, every in [("Daily", 1), ("Weekly", 5), ("Monthly", 21)]:
    p = opt.delta_hedge_pnl(paths, K, T, R, SIGMA, every)
    pnls[label] = p
    rows.append({"Rebalancing": label, "Hedge vol": f"{SIGMA:.0%}",
                 "Mean P&L (% premium)": f"{p.mean():+.1%}", "Std (% premium)": f"{p.std():.1%}"})
for wrong in (0.15, 0.25):
    p = opt.delta_hedge_pnl(paths, K, T, R, wrong, 1)
    rows.append({"Rebalancing": "Daily", "Hedge vol": f"{wrong:.0%}",
                 "Mean P&L (% premium)": f"{p.mean():+.1%}", "Std (% premium)": f"{p.std():.1%}"})

table = pd.DataFrame(rows)
print(f"Short 1y ATM call, true vol {SIGMA:.0%}, 10,000 paths\n")
print(table.to_string(index=False))
table.to_csv(OUT / "03_delta_hedging.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 5))
for label, p in pnls.items():
    ax.hist(p, bins=120, alpha=0.5, density=True, label=label)
ax.set(title="Hedging error vs rebalancing frequency", xlabel="P&L (fraction of premium)", ylabel="Density")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "03_delta_hedging.png", dpi=150)
print("\nSaved figures/03_delta_hedging.png and figures/03_delta_hedging.csv")

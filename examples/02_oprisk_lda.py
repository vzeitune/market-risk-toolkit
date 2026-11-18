"""Operational risk capital for external fraud at a large bank (LDA), with a
reasonableness check against the bank's published capital.

Calibration comes from an MSc group project (my part was the Monte Carlo engine;
the fitted parameters below were a group effort): frequency and severity were fitted
to public industry loss events (ORX News) and scaled to HSBC's size by total
assets. The point of this script is the second half: the naive calibration
fails a basic benchmark test, and the sensitivity table shows why.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mrt import oprisk_lda as lda

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)

LAMBDA_SAMPLE = 30.4            # events per year in the sample
SIZE_RATIO = 6.4                # bank size / average sample bank size
MU_SAMPLE, SIGMA = 16.8239, 1.8924
BANK_ASSETS, SAMPLE_ASSETS = 3_233_034, 500_000   # USD m

BENCHMARK_OWN_FUNDS = 9.7e9     # HSBC op-risk own funds requirement, all event types (Pillar 3, Dec 2025)

lam = LAMBDA_SAMPLE * SIZE_RATIO
mu = lda.scale_severity_mu(MU_SAMPLE, BANK_ASSETS, SAMPLE_ASSETS)
N = 100_000
base = lda.lda_capital(lam, mu, SIGMA, 0.999, n_sims=N)

bn = 1e9
print(f"External fraud LDA, {N:,} simulated years (naive calibration)")
print(f"  lambda = {lam:.1f} events/yr, mu = {mu:.3f}, sigma = {SIGMA}")
print(f"  Expected annual loss : USD {base.expected_loss / bn:8.2f}bn")
print(f"  99.9% VaR (capital)  : USD {base.var / bn:8.2f}bn")
print(f"  99.9% ES             : USD {base.es / bn:8.2f}bn")
print(f"\nReasonableness check: published op-risk own funds (ALL event types) = USD {BENCHMARK_OWN_FUNDS / bn:.1f}bn")
print(f"  Model external-fraud capital is {base.var / BENCHMARK_OWN_FUNDS:.0f}x the bank's total. The calibration fails.")
print("  Likely causes: (1) size is applied to both frequency AND severity (double counting);")
print("  (2) ORX News only records large public events, so severity is biased upwards;")
print("  (3) the lognormal tail is extremely sensitive to sigma (see table).")

# Sensitivity: which assumption moves capital most?
rows = []
for name, l, m, s in [
    ("Naive base case", lam, mu, SIGMA),
    ("Frequency scale 3x", LAMBDA_SAMPLE * 3, mu, SIGMA),
    ("Frequency scale 10x", LAMBDA_SAMPLE * 10, mu, SIGMA),
    ("Severity sigma -10%", lam, mu, SIGMA * 0.9),
    ("Severity sigma +10%", lam, mu, SIGMA * 1.1),
    ("Scale frequency only", lam, MU_SAMPLE, SIGMA),
    ("Scale frequency only, sigma -20%", lam, MU_SAMPLE, SIGMA * 0.8),
]:
    r = lda.lda_capital(l, m, s, 0.999, n_sims=N, seed=42)
    rows.append({"Scenario": name, "Lambda": round(l, 1), "Sigma": round(s, 3),
                 "99.9% VaR (USD bn)": round(r.var / bn, 2),
                 "vs base": f"{r.var / base.var - 1:+.0%}",
                 "x benchmark": round(r.var / BENCHMARK_OWN_FUNDS, 1)})
table = pd.DataFrame(rows)
print("\nSensitivity of 99.9% capital:\n" + table.to_string(index=False))
table.to_csv(OUT / "02_oprisk_sensitivity.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x = base.annual_losses / bn
axes[0].hist(x, bins=200, density=True, color="steelblue", alpha=0.75)
for val, lab, col, ls in [(base.expected_loss, "Expected loss", "green", "-"),
                          (base.var, "99.9% VaR", "red", "--"), (base.es, "99.9% ES", "darkred", ":")]:
    axes[0].axvline(val / bn, color=col, ls=ls, lw=2, label=f"{lab} = {val / bn:.2f}bn")
axes[0].set(title="Annual external fraud loss", xlabel="USD bn", ylabel="Density")
axes[0].legend(fontsize=8)
axes[1].barh(table["Scenario"], table["99.9% VaR (USD bn)"], color="salmon")
axes[1].axvline(BENCHMARK_OWN_FUNDS / bn, color="black", ls="--",
                label=f"Bank's total op-risk own funds ({BENCHMARK_OWN_FUNDS / bn:.1f}bn)")
axes[1].set_xscale("log")
axes[1].set(title="99.9% capital under each assumption (log scale)", xlabel="USD bn")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "02_oprisk_lda.png", dpi=150)
print("\nSaved figures/02_oprisk_lda.png and figures/02_oprisk_sensitivity.csv")

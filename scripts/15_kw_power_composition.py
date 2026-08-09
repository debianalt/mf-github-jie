"""
================================================================================
Kruskal-Wallis Power Simulation and Cluster Income Composition (Revision 2)
================================================================================
Part A: Monte Carlo power of the Kruskal-Wallis test with the observed cluster
sizes (8, 7, 3, 13), under scenarios where one cluster's mean GDP per capita
differs from the others by 0.5-2.0 SD. Quantifies how informative the observed
H = 1.84, p = 0.61 actually is.

Part B: descriptive income composition of the four trajectory clusters (mean,
min, max GDP per capita and member list), which provides the direct evidence
that clusters mix income levels.

Outputs: table_s2_kw_power.csv, table_s3_cluster_income.csv
"""

import pandas as pd
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'panel_analysis_1996_2021.csv')
OUT = os.path.join(ROOT, 'output')
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Part A: power simulation
# ---------------------------------------------------------------------------
SIZES = [8, 7, 3, 13]          # Clusters 1-4 (Section 4.1)
REPS = 5000
rng = np.random.default_rng(42)

print("=" * 72)
print("PART A: KRUSKAL-WALLIS POWER, cluster sizes 8/7/3/13, alpha = 0.05")
print("=" * 72)

rows = []
for shift in [0.5, 1.0, 1.5, 2.0]:
    for target, target_label in [(2, 'n=3 cluster differs'), (3, 'n=13 cluster differs')]:
        rej = 0
        for _ in range(REPS):
            groups = [rng.normal(0, 1, n) for n in SIZES]
            groups[target] = groups[target] + shift
            _, p = stats.kruskal(*groups)
            rej += (p < 0.05)
        power = rej / REPS
        rows.append({'Shift_SD': shift, 'Scenario': target_label,
                     'Power': round(power, 3)})
        print(f"  shift = {shift:.1f} SD, {target_label:<22}: power = {power:.0%}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, 'table_s2_kw_power.csv'), index=False)
print("\nSaved: output/table_s2_kw_power.csv")

# ---------------------------------------------------------------------------
# Part B: cluster income composition
# ---------------------------------------------------------------------------
CLUSTERS = {
    'Cluster 1 (Progressive)': ['BR', 'CY', 'DK', 'IE', 'NL', 'PY', 'PL', 'SE'],
    'Cluster 2 (Stable)':      ['HR', 'FR', 'GR', 'MT', 'PT', 'ES', 'UY'],
    'Cluster 3 (Atypical)':    ['AR', 'EE', 'FI'],
    'Cluster 4 (Convergence)': ['AT', 'BE', 'BG', 'CZ', 'DE', 'HU', 'IT', 'LV',
                                'LT', 'LU', 'RO', 'SK', 'SI'],
}

panel = pd.read_csv(DATA)
d = panel[(panel['year'] >= 2000) & (panel['year'] <= 2021)].copy()
gdp = d.groupby('iso2_code')['GDP_pc'].mean()  # arithmetic mean GDP pc, 2000-2021

print("\n" + "=" * 72)
print("PART B: GDP PER CAPITA BY TRAJECTORY CLUSTER (2000-2021 mean, USD)")
print("=" * 72)

missing = [c for members in CLUSTERS.values() for c in members if c not in gdp.index]
if missing:
    print(f"WARNING: no GDP data for {missing}")

rows = []
for label, members in CLUSTERS.items():
    vals = gdp.reindex(members).dropna()
    lo, hi = vals.idxmin(), vals.idxmax()
    print(f"\n{label} (n={len(members)}): mean = {vals.mean():,.0f}, "
          f"min = {vals.min():,.0f} ({lo}), max = {vals.max():,.0f} ({hi})")
    print(f"  ratio max/min = {vals.max()/vals.min():.1f}")
    print("  " + ", ".join(f"{c}: {v:,.0f}" for c, v in vals.sort_values(ascending=False).items()))
    rows.append({'Cluster': label, 'n': len(members),
                 'GDPpc_mean': round(vals.mean()), 'GDPpc_min': round(vals.min()),
                 'Min_country': lo, 'GDPpc_max': round(vals.max()),
                 'Max_country': hi,
                 'Ratio_max_min': round(vals.max()/vals.min(), 1)})

pd.DataFrame(rows).to_csv(os.path.join(OUT, 'table_s3_cluster_income.csv'), index=False)
print("\nSaved: output/table_s3_cluster_income.csv")

# Observed KW test on mean GDP per capita across clusters.
# This is the statistic reported in Section 4.1 of the manuscript.
groups = [gdp.reindex(m).dropna().values for m in CLUSTERS.values()]
H, p = stats.kruskal(*groups)
print(f"\nObserved Kruskal-Wallis, mean GDP per capita (2000-2021) across clusters:"
      f" H = {H:.2f}, p = {p:.2f}  [reported in Section 4.1]")

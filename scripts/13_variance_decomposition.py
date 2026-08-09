"""
================================================================================
Variance Decomposition: between- vs within-country variation (Revision 2)
================================================================================
Quantifies the share of total variation in each technology indicator that is
within-country over time, versus between-country. Motivates the identification
argument in Section 3.5: pooled OLS answers a between-country question because
within-country variation is a minority share of the total.

Output: table_s1_variance_decomposition.csv
"""

import pandas as pd
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'panel_analysis_1996_2021.csv')
OUT = os.path.join(ROOT, 'output')
import numpy as np

panel = pd.read_csv(DATA)
d = panel[(panel['year'] >= 2000) & (panel['year'] <= 2021)].copy()

VARS = [
    ('RD_pct_GDP', 'R&D expenditure (% GDP)'),
    ('Patents_per_million', 'Patents per million'),
    ('Hightech_exports_pct', 'High-tech exports (%)'),
    ('Researchers_per_million', 'Researchers per million'),
    ('log_GDP_pc', 'log GDP per capita'),
    ('MF_pc', 'Material Footprint per capita'),
]

rows = []
print("=" * 78)
print("VARIANCE DECOMPOSITION (2000-2021)")
print("=" * 78)
print(f"{'Variable':<32}{'SD_total':>9}{'SD_betw':>9}{'SD_with':>9}{'SD_2way':>9}{'with%':>8}")

for var, label in VARS:
    s = d[['iso2_code', 'year', var]].dropna()
    sd_total = s[var].std()
    sd_between = s.groupby('iso2_code')[var].mean().std()
    demeaned = s[var] - s.groupby('iso2_code')[var].transform('mean')
    sd_within = demeaned.std()
    twoway = demeaned - demeaned.groupby(s['year']).transform('mean')
    sd_twoway = twoway.std()
    share = sd_within / sd_total
    print(f"{label:<32}{sd_total:9.3f}{sd_between:9.3f}{sd_within:9.3f}{sd_twoway:9.3f}{share:8.1%}")
    rows.append({
        'Variable': label,
        'SD_total': round(sd_total, 3),
        'SD_between': round(sd_between, 3),
        'SD_within': round(sd_within, 3),
        'SD_twoway_demeaned': round(sd_twoway, 3),
        'Within_share_pct': round(100 * share, 1),
        'N': len(s),
        'Countries': s['iso2_code'].nunique(),
    })

out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, 'table_s1_variance_decomposition.csv'), index=False)
print(f"\nSaved: output/table_s1_variance_decomposition.csv")
print("\nNotes: SD_between = SD of country means; SD_within = SD after removing")
print("country means; SD_2way = SD after additionally removing year means")
print("(the identifying variation available to the two-way FE estimator).")

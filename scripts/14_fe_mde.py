"""
================================================================================
Fixed Effects with Standard Errors and Minimum Detectable Effects (Revision 2)
================================================================================
Re-estimates the four specifications of Table 3 (pooled OLS, time FE, country
FE, two-way FE), now reporting the cluster-robust standard error of the
technology coefficient and, for the FE specifications, the minimum detectable
effect (MDE) at 80% power (2.8 x SE). The MDE adjudicates between the two
candidate explanations for the two-way FE null results: if |B_countryFE| > MDE
under two-way FE, the two-way specification had the power to detect an effect
of that size, so its null cannot be attributed to power loss alone.

Output: table3_fe_robustness_v2.csv
"""

import pandas as pd
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'panel_analysis_1996_2021.csv')
OUT = os.path.join(ROOT, 'output')
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

panel = pd.read_csv(DATA)
analysis = panel[(panel['year'] >= 2000) & (panel['year'] <= 2021)].copy()
print(f"Panel: {len(analysis)} obs, {analysis['iso2_code'].nunique()} countries, "
      f"{analysis['year'].min()}-{analysis['year'].max()}")

# Standardise variables (same convention as 12_fe_robustness.py)
std_vars = ['RD_pct_GDP', 'Patents_per_million', 'Hightech_exports_pct',
            'Researchers_per_million', 'log_GDP_pc']
for var in std_vars:
    m, s = analysis[var].mean(), analysis[var].std()
    if s > 0:
        analysis[f'{var}_std'] = (analysis[var] - m) / s

year_dummies = pd.get_dummies(analysis['year'].astype(str), prefix='yr',
                              drop_first=True, dtype=float)
country_dummies = pd.get_dummies(analysis['iso2_code'], prefix='ctry',
                                 drop_first=True, dtype=float)
analysis = pd.concat([analysis, year_dummies, country_dummies], axis=1)
year_cols = list(year_dummies.columns)
country_cols = list(country_dummies.columns)


def run_fe_model(data, y_var, x_vars, fe_cols):
    reg_data = data.dropna(subset=[y_var] + x_vars).copy()
    available_fe = [c for c in fe_cols if c in reg_data.columns]
    Y = reg_data[y_var]
    X = sm.add_constant(reg_data[x_vars + available_fe])
    model = sm.OLS(Y, X).fit(cov_type='cluster',
                             cov_kwds={'groups': reg_data['iso2_code']})
    return model, reg_data


tech_indicators = [
    ('RD_pct_GDP_std', 'R&D (% GDP)'),
    ('Patents_per_million_std', 'Patents/million'),
    ('Hightech_exports_pct_std', 'High-tech exports'),
    ('Researchers_per_million_std', 'Researchers/million'),
]

SPECS = [
    ('Pooled OLS', []),
    ('Time FE', year_cols),
    ('Country FE', country_cols),
    ('Two-way FE', country_cols + year_cols),
]

results = []
print("\n" + "=" * 96)
print("TABLE 3 (v2): FE SPECIFICATIONS WITH SE AND MDE (DV = MF per capita)")
print("=" * 96)
print(f"{'Indicator':<20}{'Model':<12}{'B_tech':>8}{'SE':>7}{'p':>8}{'MDE80':>7}{'B_GDP':>8}{'p_GDP':>8}{'N':>6}")

for tech_var, tech_label in tech_indicators:
    b_cfe = None
    for spec_name, fe in SPECS:
        m, d = run_fe_model(analysis, 'MF_pc', [tech_var, 'log_GDP_pc_std'], fe)
        b, se, p = m.params[tech_var], m.bse[tech_var], m.pvalues[tech_var]
        bg, pg = m.params['log_GDP_pc_std'], m.pvalues['log_GDP_pc_std']
        mde = 2.8 * se if 'FE' in spec_name else np.nan
        if spec_name == 'Country FE':
            b_cfe = b
        print(f"{tech_label:<20}{spec_name:<12}{b:8.2f}{se:7.2f}{p:8.4f}"
              f"{(f'{mde:7.2f}' if not np.isnan(mde) else '      -')}{bg:8.2f}{pg:8.4f}{int(m.nobs):6d}")
        results.append({
            'Indicator': tech_label, 'Model': spec_name,
            'B_tech': round(b, 3), 'SE_tech': round(se, 3),
            'p_tech': round(p, 4),
            'MDE_80pct': round(mde, 3) if not np.isnan(mde) else '',
            'B_GDP': round(bg, 3), 'p_GDP': round(pg, 4),
            'N': int(m.nobs),
        })
    # Adjudication line: could two-way FE detect an effect of country-FE size?
    tw = results[-1]
    if b_cfe is not None and tw['MDE_80pct'] != '':
        detectable = abs(b_cfe) > tw['MDE_80pct']
        print(f"{'':<20}-> |B_countryFE| = {abs(b_cfe):.2f} vs two-way MDE = "
              f"{tw['MDE_80pct']:.2f}: two-way FE "
              f"{'COULD detect an effect of that size (null not attributable to power)' if detectable else 'could NOT reliably detect an effect of that size (power insufficient)'}")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUT, 'table_fe_robustness_mde.csv'), index=False)
print(f"\nSaved: output/table_fe_robustness_mde.csv")

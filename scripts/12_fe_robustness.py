"""
================================================================================
Fixed Effects Robustness Checks (Revision — Reviewer 1.5)
================================================================================
Runs country FE, time FE, and two-way FE models as robustness checks
for the extended panel (2000-2021).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("FIXED EFFECTS ROBUSTNESS CHECKS")
print("=" * 70)

# Load data
panel = pd.read_csv('../../data/processed/panel_long_tech_mfa.csv')
analysis = panel[(panel['year'] >= 2000) & (panel['year'] <= 2021)].copy()
print(f"Panel: {len(analysis)} obs, {analysis['iso2_code'].nunique()} countries, "
      f"{analysis['year'].min()}-{analysis['year'].max()}")

# Standardise variables
std_vars = ['RD_pct_GDP', 'Patents_per_million', 'Hightech_exports_pct',
            'Researchers_per_million', 'log_GDP_pc']
for var in std_vars:
    if var in analysis.columns:
        m, s = analysis[var].mean(), analysis[var].std()
        if s > 0:
            analysis[f'{var}_std'] = (analysis[var] - m) / s

# Year dummies
analysis['year_cat'] = analysis['year'].astype(str)
year_dummies = pd.get_dummies(analysis['year_cat'], prefix='yr', drop_first=True, dtype=float)
analysis = pd.concat([analysis, year_dummies], axis=1)
year_cols = [c for c in year_dummies.columns]

# Country dummies
country_dummies = pd.get_dummies(analysis['iso2_code'], prefix='ctry', drop_first=True, dtype=float)
analysis = pd.concat([analysis, country_dummies], axis=1)
country_cols = [c for c in country_dummies.columns]


def within_r2(Y, residuals, groups, time_groups=None):
    """Compute within-R² by demeaning Y by group (and optionally time)."""
    Y_arr = np.array(Y, dtype=float)
    # Demean by entity (country)
    group_arr = np.array(groups)
    Y_demeaned = Y_arr.copy()
    for g in np.unique(group_arr):
        mask = group_arr == g
        Y_demeaned[mask] -= Y_arr[mask].mean()
    # Additionally demean by time if provided
    if time_groups is not None:
        time_arr = np.array(time_groups)
        for t in np.unique(time_arr):
            mask = time_arr == t
            Y_demeaned[mask] -= Y_demeaned[mask].mean()
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum(Y_demeaned ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def run_fe_model(data, y_var, x_vars, fe_cols, title):
    """Run OLS with fixed effects (as dummies) and cluster-robust SE."""
    all_vars = [y_var, 'iso2_code'] + x_vars
    # Only keep rows where all x_vars and y_var are non-null
    reg_data = data.dropna(subset=[y_var] + x_vars).copy()

    # Make sure FE columns exist in reg_data
    available_fe = [c for c in fe_cols if c in reg_data.columns]

    Y = reg_data[y_var]
    X = sm.add_constant(reg_data[x_vars + available_fe])

    model = sm.OLS(Y, X).fit(cov_type='cluster',
                              cov_kwds={'groups': reg_data['iso2_code']})
    return model, reg_data


# ============================================================================
# Define model specifications
# ============================================================================
tech_indicators = [
    ('RD_pct_GDP_std', 'R&D (% GDP)'),
    ('Patents_per_million_std', 'Patents/million'),
    ('Hightech_exports_pct_std', 'High-tech exports'),
    ('Researchers_per_million_std', 'Researchers/million'),
]

results = []

print("\n" + "=" * 70)
print("TABLE 3: FIXED EFFECTS ROBUSTNESS (DV = Material Footprint per capita)")
print("=" * 70)

for tech_var, tech_label in tech_indicators:
    print(f"\n{'-' * 70}")
    print(f"  INDICATOR: {tech_label}")
    print(f"{'-' * 70}")

    # ---- A: Pooled OLS (baseline, replicating Table 2) ----
    try:
        m_pooled, d = run_fe_model(analysis, 'MF_pc',
                                    [tech_var, 'log_GDP_pc_std'],
                                    [], f'Pooled: {tech_label}+GDP')
        b_tech = m_pooled.params[tech_var]
        p_tech = m_pooled.pvalues[tech_var]
        b_gdp = m_pooled.params['log_GDP_pc_std']
        p_gdp = m_pooled.pvalues['log_GDP_pc_std']
        r2 = m_pooled.rsquared
        n = int(m_pooled.nobs)
        print(f"\n  A. Pooled OLS (n={n})")
        print(f"     {tech_label}: B={b_tech:.3f}, p={p_tech:.4f}")
        print(f"     log(GDP/cap):  B={b_gdp:.3f}, p={p_gdp:.4f}")
        print(f"     R2={r2:.3f}")
        results.append({
            'Indicator': tech_label, 'Model': 'Pooled OLS',
            'B_tech': f'{b_tech:.3f}', 'p_tech': f'{p_tech:.4f}',
            'B_GDP': f'{b_gdp:.3f}', 'p_GDP': f'{p_gdp:.4f}',
            'R2': f'{r2:.3f}', 'N': n
        })
    except Exception as e:
        print(f"  A. Pooled OLS: FAILED ({e})")

    # ---- B: Time FE (year dummies) ----
    try:
        m_tfe, d = run_fe_model(analysis, 'MF_pc',
                                 [tech_var, 'log_GDP_pc_std'],
                                 year_cols, f'Time FE: {tech_label}+GDP')
        b_tech = m_tfe.params[tech_var]
        p_tech = m_tfe.pvalues[tech_var]
        b_gdp = m_tfe.params['log_GDP_pc_std']
        p_gdp = m_tfe.pvalues['log_GDP_pc_std']
        r2 = m_tfe.rsquared
        n = int(m_tfe.nobs)
        print(f"\n  B. Time FE (n={n})")
        print(f"     {tech_label}: B={b_tech:.3f}, p={p_tech:.4f}")
        print(f"     log(GDP/cap):  B={b_gdp:.3f}, p={p_gdp:.4f}")
        print(f"     R2={r2:.3f}")
        results.append({
            'Indicator': tech_label, 'Model': 'Time FE',
            'B_tech': f'{b_tech:.3f}', 'p_tech': f'{p_tech:.4f}',
            'B_GDP': f'{b_gdp:.3f}', 'p_GDP': f'{p_gdp:.4f}',
            'R2': f'{r2:.3f}', 'N': n
        })
    except Exception as e:
        print(f"  B. Time FE: FAILED ({e})")

    # ---- C: Country FE ----
    try:
        m_cfe, d = run_fe_model(analysis, 'MF_pc',
                                 [tech_var, 'log_GDP_pc_std'],
                                 country_cols, f'Country FE: {tech_label}+GDP')
        b_tech = m_cfe.params[tech_var]
        p_tech = m_cfe.pvalues[tech_var]
        b_gdp = m_cfe.params['log_GDP_pc_std']
        p_gdp = m_cfe.pvalues['log_GDP_pc_std']
        r2 = within_r2(d['MF_pc'], m_cfe.resid, d['iso2_code'])
        n = int(m_cfe.nobs)
        print(f"\n  C. Country FE (n={n})")
        print(f"     {tech_label}: B={b_tech:.3f}, p={p_tech:.4f}")
        print(f"     log(GDP/cap):  B={b_gdp:.3f}, p={p_gdp:.4f}")
        print(f"     Within-R2={r2:.3f}")
        results.append({
            'Indicator': tech_label, 'Model': 'Country FE',
            'B_tech': f'{b_tech:.3f}', 'p_tech': f'{p_tech:.4f}',
            'B_GDP': f'{b_gdp:.3f}', 'p_GDP': f'{p_gdp:.4f}',
            'R2': f'{r2:.3f}', 'N': n
        })
    except Exception as e:
        print(f"  C. Country FE: FAILED ({e})")

    # ---- D: Two-way FE (country + year) ----
    try:
        m_twfe, d = run_fe_model(analysis, 'MF_pc',
                                  [tech_var, 'log_GDP_pc_std'],
                                  country_cols + year_cols,
                                  f'Two-way FE: {tech_label}+GDP')
        b_tech = m_twfe.params[tech_var]
        p_tech = m_twfe.pvalues[tech_var]
        b_gdp = m_twfe.params['log_GDP_pc_std']
        p_gdp = m_twfe.pvalues['log_GDP_pc_std']
        r2 = within_r2(d['MF_pc'], m_twfe.resid, d['iso2_code'], d['year'])
        n = int(m_twfe.nobs)
        print(f"\n  D. Two-way FE (n={n})")
        print(f"     {tech_label}: B={b_tech:.3f}, p={p_tech:.4f}")
        print(f"     log(GDP/cap):  B={b_gdp:.3f}, p={p_gdp:.4f}")
        print(f"     Within-R2={r2:.3f}")
        results.append({
            'Indicator': tech_label, 'Model': 'Two-way FE',
            'B_tech': f'{b_tech:.3f}', 'p_tech': f'{p_tech:.4f}',
            'B_GDP': f'{b_gdp:.3f}', 'p_GDP': f'{p_gdp:.4f}',
            'R2': f'{r2:.3f}', 'N': n
        })
    except Exception as e:
        print(f"  D. Two-way FE: FAILED ({e})")

# ============================================================================
# Save results
# ============================================================================
results_df = pd.DataFrame(results)
results_df.to_csv('table3_fe_robustness.csv', index=False)
print(f"\n{'=' * 70}")
print("Results saved to: table3_fe_robustness.csv")
print(f"{'=' * 70}")

# Summary table
print("\n\nSUMMARY TABLE:")
print(f"{'Indicator':<22} {'Model':<15} {'B_tech':>8} {'p_tech':>8} {'B_GDP':>8} {'p_GDP':>8} {'R2':>6} {'N':>5}")
print("-" * 85)
for _, row in results_df.iterrows():
    sig = '***' if float(row['p_tech']) < 0.001 else '**' if float(row['p_tech']) < 0.01 else '*' if float(row['p_tech']) < 0.05 else ''
    print(f"{row['Indicator']:<22} {row['Model']:<15} {row['B_tech']:>8}{sig:<3} {row['p_tech']:>8} "
          f"{row['B_GDP']:>8} {row['p_GDP']:>8} {row['R2']:>6} {row['N']:>5}")

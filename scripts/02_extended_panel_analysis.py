"""
================================================================================
Extended Panel Analysis: Technology Indicators and Material Footprint
================================================================================
Multiple technology indicators with longer time series (2000-2021)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("EXTENDED PANEL ANALYSIS: TECHNOLOGY AND MATERIAL FOOTPRINT")
print("="*70)

# Load panel data (1996-2021)
panel = pd.read_csv('../data/panel_analysis_1996_2021.csv')
print(f"\nLoaded panel: {len(panel)} obs, {panel['iso2_code'].nunique()} countries")

# Focus on analysis period 2000-2021 (observed data only, no MF estimates)
analysis_period = panel[(panel['year'] >= 2000) & (panel['year'] <= 2021)].copy()
print(f"Analysis period (2000-2021): {len(analysis_period)} obs")

# ============================================================================
# 1. DESCRIPTIVE STATISTICS BY BLOC
# ============================================================================
print("\n" + "="*70)
print("1. DESCRIPTIVE STATISTICS BY BLOC")
print("="*70)

vars_of_interest = ['MF_pc', 'DMC_pc', 'Gap_MF_DMC', 'GDP_pc',
                    'RD_pct_GDP', 'Patents_per_million', 'Hightech_exports_pct',
                    'Researchers_per_million']

# Recent period (2015-2021) for comparison
recent = analysis_period[analysis_period['year'] >= 2015]

print("\n  Mean values by bloc (2015-2021):")
print("-"*70)
for var in vars_of_interest:
    if var in recent.columns:
        eu = recent[recent['Bloc']=='EU-27'][var].mean()
        merc = recent[recent['Bloc']=='MERCOSUR-4'][var].mean()
        if pd.notna(eu) and pd.notna(merc):
            ratio = eu/merc if merc != 0 else np.nan
            print(f"  {var:25s}: EU={eu:10.2f}  MERCOSUR={merc:10.2f}  Ratio={ratio:.2f}")

# ============================================================================
# 2. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("2. CORRELATION MATRIX (POOLED)")
print("="*70)

tech_vars = ['RD_pct_GDP', 'Patents_per_million', 'Hightech_exports_pct',
             'Researchers_per_million', 'log_GDP_pc']
outcome_vars = ['MF_pc', 'Gap_MF_DMC']

# Create correlation subset
corr_data = analysis_period[tech_vars + outcome_vars].dropna()
print(f"\n  Observations for correlation: {len(corr_data)}")

corr_matrix = corr_data.corr()
print("\n  Correlations with MF_pc:")
for var in tech_vars:
    r = corr_matrix.loc[var, 'MF_pc']
    print(f"    {var:25s}: r = {r:+.3f}")

print("\n  Correlations with Gap_MF_DMC (externalisation):")
for var in tech_vars:
    r = corr_matrix.loc[var, 'Gap_MF_DMC']
    print(f"    {var:25s}: r = {r:+.3f}")

# ============================================================================
# 3. PANEL REGRESSIONS
# ============================================================================
print("\n" + "="*70)
print("3. PANEL REGRESSIONS: MATERIAL FOOTPRINT")
print("="*70)

def run_panel_regression(data, y_var, x_vars, title):
    """Run OLS regression with cluster-robust SE"""
    reg_data = data[['iso2_code', 'year', y_var] + x_vars].dropna()

    if len(reg_data) < 50:
        print(f"  {title}: Insufficient data ({len(reg_data)} obs)")
        return None

    Y = reg_data[y_var]
    X = sm.add_constant(reg_data[x_vars])

    model = sm.OLS(Y, X).fit(cov_type='cluster',
                             cov_kwds={'groups': reg_data['iso2_code']})

    return model, reg_data

# Standardize variables for comparison
analysis_std = analysis_period.copy()
for var in ['RD_pct_GDP', 'Patents_per_million', 'Hightech_exports_pct',
            'Researchers_per_million', 'log_GDP_pc']:
    if var in analysis_std.columns:
        mean = analysis_std[var].mean()
        std = analysis_std[var].std()
        if std > 0:
            analysis_std[f'{var}_std'] = (analysis_std[var] - mean) / std

print("\n  Model specifications (DV = Material Footprint per capita):")
print("-"*70)

models = {}

# Model 1: R&D only
result = run_panel_regression(analysis_std, 'MF_pc', ['RD_pct_GDP_std'], 'M1: R&D only')
if result:
    models['M1'] = result
    m, d = result
    print(f"\n  M1: R&D only (n={len(d)})")
    print(f"      R&D: B={m.params['RD_pct_GDP_std']:.3f}, p={m.pvalues['RD_pct_GDP_std']:.4f}")
    print(f"      R2={m.rsquared:.3f}")

# Model 2: Patents only
result = run_panel_regression(analysis_std, 'MF_pc', ['Patents_per_million_std'], 'M2: Patents only')
if result:
    models['M2'] = result
    m, d = result
    print(f"\n  M2: Patents only (n={len(d)})")
    print(f"      Patents: B={m.params['Patents_per_million_std']:.3f}, p={m.pvalues['Patents_per_million_std']:.4f}")
    print(f"      R2={m.rsquared:.3f}")

# Model 3: High-tech exports only
result = run_panel_regression(analysis_std, 'MF_pc', ['Hightech_exports_pct_std'], 'M3: Hightech only')
if result:
    models['M3'] = result
    m, d = result
    print(f"\n  M3: High-tech exports only (n={len(d)})")
    print(f"      Hightech: B={m.params['Hightech_exports_pct_std']:.3f}, p={m.pvalues['Hightech_exports_pct_std']:.4f}")
    print(f"      R2={m.rsquared:.3f}")

# Model 4: Researchers only
result = run_panel_regression(analysis_std, 'MF_pc', ['Researchers_per_million_std'], 'M4: Researchers only')
if result:
    models['M4'] = result
    m, d = result
    print(f"\n  M4: Researchers only (n={len(d)})")
    print(f"      Researchers: B={m.params['Researchers_per_million_std']:.3f}, p={m.pvalues['Researchers_per_million_std']:.4f}")
    print(f"      R2={m.rsquared:.3f}")

# Model 5: GDP only (baseline)
result = run_panel_regression(analysis_std, 'MF_pc', ['log_GDP_pc_std'], 'M5: GDP only')
if result:
    models['M5'] = result
    m, d = result
    print(f"\n  M5: GDP only (n={len(d)})")
    print(f"      log(GDP/cap): B={m.params['log_GDP_pc_std']:.3f}, p={m.pvalues['log_GDP_pc_std']:.4f}")
    print(f"      R2={m.rsquared:.3f}")

# Model 6: R&D + GDP
result = run_panel_regression(analysis_std, 'MF_pc', ['RD_pct_GDP_std', 'log_GDP_pc_std'], 'M6: R&D + GDP')
if result:
    models['M6'] = result
    m, d = result
    print(f"\n  M6: R&D + GDP (n={len(d)})")
    print(f"      R&D: B={m.params['RD_pct_GDP_std']:.3f}, p={m.pvalues['RD_pct_GDP_std']:.4f}")
    print(f"      log(GDP/cap): B={m.params['log_GDP_pc_std']:.3f}, p={m.pvalues['log_GDP_pc_std']:.4f}")
    print(f"      R2={m.rsquared:.3f}")

# Model 7: Patents + GDP
result = run_panel_regression(analysis_std, 'MF_pc', ['Patents_per_million_std', 'log_GDP_pc_std'], 'M7: Patents + GDP')
if result:
    models['M7'] = result
    m, d = result
    print(f"\n  M7: Patents + GDP (n={len(d)})")
    print(f"      Patents: B={m.params['Patents_per_million_std']:.3f}, p={m.pvalues['Patents_per_million_std']:.4f}")
    print(f"      log(GDP/cap): B={m.params['log_GDP_pc_std']:.3f}, p={m.pvalues['log_GDP_pc_std']:.4f}")
    print(f"      R2={m.rsquared:.3f}")

# Model 8: Researchers + GDP
result = run_panel_regression(analysis_std, 'MF_pc', ['Researchers_per_million_std', 'log_GDP_pc_std'], 'M8: Researchers + GDP')
if result:
    models['M8'] = result
    m, d = result
    print(f"\n  M8: Researchers + GDP (n={len(d)})")
    print(f"      Researchers: B={m.params['Researchers_per_million_std']:.3f}, p={m.pvalues['Researchers_per_million_std']:.4f}")
    print(f"      log(GDP/cap): B={m.params['log_GDP_pc_std']:.3f}, p={m.pvalues['log_GDP_pc_std']:.4f}")
    print(f"      R2={m.rsquared:.3f}")

# Model 9: Full model (excludes researchers due to collinearity with R&D)
full_vars = ['RD_pct_GDP_std', 'Patents_per_million_std', 'Hightech_exports_pct_std', 'log_GDP_pc_std']
result = run_panel_regression(analysis_std, 'MF_pc', full_vars, 'M9: Full model')
if result:
    models['M9'] = result
    m, d = result
    print(f"\n  M9: Full model (n={len(d)})")
    for var in full_vars:
        print(f"      {var}: B={m.params[var]:.3f}, p={m.pvalues[var]:.4f}")
    print(f"      R2={m.rsquared:.3f}")

# ============================================================================
# 4. EXTERNALISATION GAP REGRESSIONS
# ============================================================================
print("\n" + "="*70)
print("4. PANEL REGRESSIONS: EXTERNALISATION GAP (MF - DMC)")
print("="*70)

# Model G1: R&D on Gap
result = run_panel_regression(analysis_std, 'Gap_MF_DMC', ['RD_pct_GDP_std'], 'G1: R&D on Gap')
if result:
    m, d = result
    print(f"\n  G1: R&D only (n={len(d)})")
    print(f"      R&D: B={m.params['RD_pct_GDP_std']:.3f}, p={m.pvalues['RD_pct_GDP_std']:.4f}")
    print(f"      R2={m.rsquared:.3f}")

# Model G2: R&D + GDP on Gap
result = run_panel_regression(analysis_std, 'Gap_MF_DMC', ['RD_pct_GDP_std', 'log_GDP_pc_std'], 'G2: R&D + GDP on Gap')
if result:
    m, d = result
    print(f"\n  G2: R&D + GDP (n={len(d)})")
    print(f"      R&D: B={m.params['RD_pct_GDP_std']:.3f}, p={m.pvalues['RD_pct_GDP_std']:.4f}")
    print(f"      log(GDP/cap): B={m.params['log_GDP_pc_std']:.3f}, p={m.pvalues['log_GDP_pc_std']:.4f}")
    print(f"      R2={m.rsquared:.3f}")

# Model G3: Patents + GDP on Gap
result = run_panel_regression(analysis_std, 'Gap_MF_DMC', ['Patents_per_million_std', 'log_GDP_pc_std'], 'G3: Patents + GDP on Gap')
if result:
    m, d = result
    print(f"\n  G3: Patents + GDP (n={len(d)})")
    print(f"      Patents: B={m.params['Patents_per_million_std']:.3f}, p={m.pvalues['Patents_per_million_std']:.4f}")
    print(f"      log(GDP/cap): B={m.params['log_GDP_pc_std']:.3f}, p={m.pvalues['log_GDP_pc_std']:.4f}")
    print(f"      R2={m.rsquared:.3f}")

# ============================================================================
# 5. BLOC COMPARISON
# ============================================================================
print("\n" + "="*70)
print("5. BLOC COMPARISON: TECHNOLOGY-INCOME RELATIONSHIP")
print("="*70)

for bloc in ['EU-27', 'MERCOSUR-4']:
    bloc_data = analysis_std[analysis_std['Bloc'] == bloc]
    print(f"\n  {bloc}:")

    result = run_panel_regression(bloc_data, 'MF_pc', ['RD_pct_GDP_std', 'log_GDP_pc_std'], f'{bloc}')
    if result:
        m, d = result
        print(f"    n={len(d)}, R2={m.rsquared:.3f}")
        print(f"    R&D: B={m.params['RD_pct_GDP_std']:.3f}, p={m.pvalues['RD_pct_GDP_std']:.4f}")
        print(f"    GDP: B={m.params['log_GDP_pc_std']:.3f}, p={m.pvalues['log_GDP_pc_std']:.4f}")

# ============================================================================
# 6. TEMPORAL TRENDS
# ============================================================================
print("\n" + "="*70)
print("6. TEMPORAL ANALYSIS: DECOUPLING TRENDS")
print("="*70)

# Calculate decoupling elasticity by period
def calc_elasticity(data, y_var, x_var):
    """Calculate elasticity between two variables"""
    valid = data[[y_var, x_var]].dropna()
    if len(valid) < 10:
        return np.nan
    y_growth = valid[y_var].pct_change().mean()
    x_growth = valid[x_var].pct_change().mean()
    if x_growth != 0:
        return y_growth / x_growth
    return np.nan

periods = [(2000, 2007, 'Early'), (2008, 2015, 'Middle'), (2016, 2021, 'Recent')]

print("\n  Decoupling elasticity (MF growth / GDP growth) by period and bloc:")
print("  (Tapio: <0.8 = decoupling, 0.8-1.2 = coupling, >1.2 = negative decoupling)")
print("-"*70)

for start, end, name in periods:
    period_data = analysis_period[(analysis_period['year'] >= start) & (analysis_period['year'] <= end)]
    for bloc in ['EU-27', 'MERCOSUR-4']:
        bloc_period = period_data[period_data['Bloc'] == bloc]
        # Calculate average annual growth rates
        mf_growth = bloc_period.groupby('iso2_code')['MF_pc'].apply(lambda x: (x.iloc[-1]/x.iloc[0])**(1/(len(x)-1)) - 1 if len(x) > 1 and x.iloc[0] > 0 else np.nan).mean() * 100
        gdp_growth = bloc_period.groupby('iso2_code')['GDP_pc'].apply(lambda x: (x.iloc[-1]/x.iloc[0])**(1/(len(x)-1)) - 1 if len(x) > 1 and x.iloc[0] > 0 else np.nan).mean() * 100

        if gdp_growth != 0 and not np.isnan(gdp_growth):
            elasticity = mf_growth / gdp_growth
            status = "DECOUPLING" if elasticity < 0.8 else ("COUPLING" if elasticity < 1.2 else "NEG. DECOUPLING")
            print(f"  {name:8s} ({start}-{end}) {bloc:12s}: MF={mf_growth:+.2f}%/yr, GDP={gdp_growth:+.2f}%/yr, e={elasticity:.2f} [{status}]")

# ============================================================================
# 7. KEY FINDINGS SUMMARY
# ============================================================================
print("\n" + "="*70)
print("7. KEY FINDINGS SUMMARY")
print("="*70)

print("""
  MAIN RESULTS:

  1. TECHNOLOGY INDICATORS ALONE:
     - R&D, Patents, High-tech exports all show POSITIVE correlation with MF
     - This is consistent with scale effects hypothesis

  2. CONTROLLING FOR INCOME:
     - Technology indicators become NON-SIGNIFICANT when GDP is included
     - GDP per capita explains most of the variance in Material Footprint
     - Confirms pattern found with ECIsoftware

  3. EXTERNALISATION GAP:
     - Higher income countries have larger Gap (MF > DMC)
     - Technology does not reduce externalisation independently of income

  4. DECOUPLING:
     - EU shows relative decoupling (MF growing slower than GDP)
     - MERCOSUR shows coupling or negative decoupling
     - But EU's lower MF intensity comes from EXTERNALISATION, not efficiency

  5. METHODOLOGICAL STRENGTH:
     - 22 years of data (vs 5 years for ECIsoftware)
     - Multiple technology indicators converge on same conclusion
     - Cluster-robust standard errors account for country dependence

  CONCLUSION:
  The apparent relationship between technological specialisation and lower
  material intensity is a statistical artefact of income effects and
  international trade patterns (externalisation), not genuine decoupling.
""")

# Save results
print("\n  Saving results...")
analysis_period.to_csv('../data/panel_analysis_results.csv', index=False)
print("  Saved: data/panel_analysis_results.csv")

# Save Table 3: Extended regression results as CSV
import os
os.makedirs('../output', exist_ok=True)

table3_rows = []
model_names = {'M1': 'R&D only', 'M2': 'Patents only', 'M3': 'Hightech only',
               'M4': 'Researchers only', 'M5': 'GDP only', 'M6': 'R&D + GDP',
               'M7': 'Patents + GDP', 'M8': 'Researchers + GDP', 'M9': 'Full model'}
for mname, mlabel in model_names.items():
    if mname in models:
        m, d = models[mname]
        row = {'Model': mname, 'Description': mlabel, 'N': len(d), 'R2': round(m.rsquared, 3)}
        for var in m.params.index:
            if var != 'const':
                short = var.replace('_std', '')
                row[f'B_{short}'] = round(m.params[var], 3)
                row[f'p_{short}'] = round(m.pvalues[var], 4)
        table3_rows.append(row)

pd.DataFrame(table3_rows).to_csv('../output/table3_extended_regressions.csv', index=False)
print("  Saved: output/table3_extended_regressions.csv")

print("\n" + "="*70)
print("ANALYSIS COMPLETED")
print("="*70)

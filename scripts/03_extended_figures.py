"""
================================================================================
Figures for Extended Panel Analysis Results
Figures 4 (coefficients), 5 (scatters), S1 (temporal trends)
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Unified publication-quality style (matches 01_sequence_analysis.py)
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.grid': False,
})

# Shared constants
FONTSIZE_PANEL = 11
FONTSIZE_ANNOT = 8
SIGNIFICANT_COLOR = '#2ca02c'
NONSIG_COLOR = '#7f7f7f'
EU_COLOR = '#2C3E50'       # Navy (matches script 01)
MERCOSUR_COLOR = '#27AE60'  # Emerald (matches script 01)

# GDP tercile sequential purple palette
TERCILE_COLORS = {'Low': '#DADAEB', 'Medium': '#9E9AC8', 'High': '#54278F'}

print("Loading data...")
panel = pd.read_csv('../data/panel_analysis_1996_2021.csv')
analysis = panel[(panel['year'] >= 2000) & (panel['year'] <= 2021)].copy()

print(f"Panel: {len(analysis)} observations, {analysis['year'].min()}-{analysis['year'].max()}")

# ============================================================================
# FIGURE 4: Coefficient Plot - Technology Effects With and Without GDP Control
# ============================================================================
print("\nCreating Figure 4: Coefficient comparison...")

fig, ax = plt.subplots(figsize=(7, 4.2))

indicators = ['R&D\n(% GDP)', 'Patents\n(per million)', 'High-tech\nexports (%)', 'Researchers\n(per million)']

# Without GDP control
coef_alone = [3.75, 2.93, 1.63, 4.44]
pval_alone = [0.0001, 0.0104, 0.1873, 0.0000]

# With GDP control
coef_gdp = [-1.44, -1.19, -0.11, -1.27]
pval_gdp = [0.2422, 0.2026, 0.9085, 0.3729]

x = np.arange(len(indicators))
width = 0.35

colors_alone = [SIGNIFICANT_COLOR if p < 0.05 else NONSIG_COLOR for p in pval_alone]
colors_gdp = [SIGNIFICANT_COLOR if p < 0.05 else NONSIG_COLOR for p in pval_gdp]

bars1 = ax.bar(x - width/2, coef_alone, width, label='Without GDP control',
               color=colors_alone, edgecolor='white', linewidth=0.3)
bars2 = ax.bar(x + width/2, coef_gdp, width,
               label='With GDP control', color=colors_gdp, edgecolor='white', linewidth=0.3,
               alpha=0.7, hatch='///')

# R-squared values in LaTeX format
r2_alone = [0.182, 0.114, 0.037, 0.247]
r2_gdp = [0.547, 0.573, 0.517, 0.557]

for i, (c, r2) in enumerate(zip(coef_alone, r2_alone)):
    ax.annotate(f'$R^2$={r2:.2f}', (x[i] - width/2, c + 0.15), ha='center', fontsize=7)

for i, (c, r2) in enumerate(zip(coef_gdp, r2_gdp)):
    y_pos = c - 0.45 if c < 0 else c + 0.15
    ax.annotate(f'$R^2$={r2:.2f}', (x[i] + width/2, y_pos), ha='center', fontsize=7,
                style='italic')

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel('Standardised coefficient (effect on MF per capita)')
ax.set_xticks(x)
ax.set_xticklabels(indicators)

# Subtle Y-axis grid only
ax.yaxis.grid(True, alpha=0.15, linewidth=0.5)
ax.set_axisbelow(True)

# Custom legend in upper right
legend_elements = [
    Patch(facecolor=SIGNIFICANT_COLOR, edgecolor='0.5', label='Significant ($p$ < 0.05)'),
    Patch(facecolor=NONSIG_COLOR, edgecolor='0.5', label='Not significant'),
    Patch(facecolor='white', edgecolor='0.5', hatch='///', label='With GDP control')
]
ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=7)

ax.set_ylim(-3, 6)
ax.text(0.02, 0.98, '$N$ = 430\u2013682', transform=ax.transAxes, fontsize=7.5,
        verticalalignment='top', style='italic')

plt.tight_layout()
plt.savefig('../figures/fig_extended_coefficients.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('../figures/fig_extended_coefficients.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figures/fig_extended_coefficients.png")

# ============================================================================
# FIGURE 5: Scatter plots - Technology vs MF with GDP as color
# ============================================================================
print("\nCreating Figure 5: Scatter plots...")

fig, axes = plt.subplots(2, 2, figsize=(8, 7))

vars_to_plot = [
    ('RD_pct_GDP', 'R&D expenditure (% of GDP)'),
    ('Patents_per_million', 'Patents per million population'),
    ('Hightech_exports_pct', 'High-tech exports (% of manufactures)'),
    ('log_GDP_pc', 'Log GDP per capita')
]

panel_letters = ['A', 'B', 'C', 'D']

for idx, (var, label) in enumerate(vars_to_plot):
    ax = axes[idx // 2, idx % 2]

    mask = analysis[['MF_pc', var, 'log_GDP_pc', 'Bloc']].notna().all(axis=1)
    x_data = analysis.loc[mask, var].values
    y_data = analysis.loc[mask, 'MF_pc'].values
    gdp_data = analysis.loc[mask, 'log_GDP_pc'].values

    # GDP terciles with sequential purple
    q33 = np.percentile(gdp_data, 33)
    q67 = np.percentile(gdp_data, 67)
    terciles = np.where(gdp_data <= q33, 'Low',
                        np.where(gdp_data <= q67, 'Medium', 'High'))

    for tercile in ['Low', 'Medium', 'High']:
        mask_t = terciles == tercile
        ax.scatter(x_data[mask_t], y_data[mask_t],
                   c=TERCILE_COLORS[tercile], alpha=0.45, s=18, label=f'GDP {tercile}',
                   edgecolor='white', linewidth=0.3)

    # Regression line
    iso_data = analysis.loc[mask, 'iso2_code'].values
    X = sm.add_constant(x_data)
    model_line = sm.OLS(y_data, X).fit()
    model = sm.OLS(y_data, X).fit(cov_type='cluster',
                                   cov_kwds={'groups': iso_data})
    x_line = np.linspace(x_data.min(), x_data.max(), 100)
    y_line = model_line.params[0] + model_line.params[1] * x_line
    ax.plot(x_line, y_line, color='0.2', linewidth=1.0, linestyle='--')

    # R-squared and p-value annotation
    r2 = model.rsquared
    pval = model.pvalues[1]
    sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else ''))
    ax.text(0.05, 0.95, f'$R^2$ = {r2:.3f}{sig}\n$n$ = {len(x_data)}',
            transform=ax.transAxes, fontsize=7.5, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                      edgecolor='0.8', linewidth=0.5))

    ax.set_xlabel(label)
    ax.set_ylabel('Material Footprint (t/cap)')
    ax.text(-0.02, 1.08, panel_letters[idx], transform=ax.transAxes,
            fontsize=FONTSIZE_PANEL, fontweight='bold', va='top', ha='right')
    ax.set_title(f'  {label.split("(")[0].strip()}', fontsize=9, loc='left')

    if idx == 0:
        ax.legend(title='GDP tercile', loc='lower right', fontsize=7, title_fontsize=7)

plt.tight_layout()
plt.savefig('../figures/fig_extended_scatters.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('../figures/fig_extended_scatters.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figures/fig_extended_scatters.png")

# ============================================================================
# FIGURE S1: Bloc comparison over time
# ============================================================================
print("\nCreating Figure S1: Bloc comparison over time...")

fig, axes = plt.subplots(2, 2, figsize=(8, 6.5))

bloc_style = {
    'EU-27': dict(color=EU_COLOR, marker='o', markersize=3, linewidth=1.5),
    'MERCOSUR-4': dict(color=MERCOSUR_COLOR, marker='^', markersize=3, linewidth=1.5),
}
panel_letters_s1 = ['A', 'B', 'C', 'D']

# Panel A: MF per capita over time
ax = axes[0, 0]
for bloc, style in bloc_style.items():
    bloc_data = analysis[analysis['Bloc'] == bloc].groupby('year')['MF_pc'].mean()
    ax.plot(bloc_data.index, bloc_data.values, label=bloc, **style)
ax.set_xlabel('Year')
ax.set_ylabel('Material Footprint (t/cap)')
ax.text(-0.02, 1.08, 'A', transform=ax.transAxes, fontsize=FONTSIZE_PANEL,
        fontweight='bold', va='top', ha='right')
ax.set_title('  Material Footprint per capita', fontsize=9, loc='left')
ax.legend(fontsize=7)

# Panel B: R&D over time
ax = axes[0, 1]
for bloc, style in bloc_style.items():
    bloc_data = analysis[analysis['Bloc'] == bloc].groupby('year')['RD_pct_GDP'].mean()
    ax.plot(bloc_data.index, bloc_data.values, label=bloc, **style)
ax.set_xlabel('Year')
ax.set_ylabel('R&D (% of GDP)')
ax.text(-0.02, 1.08, 'B', transform=ax.transAxes, fontsize=FONTSIZE_PANEL,
        fontweight='bold', va='top', ha='right')
ax.set_title('  R&D expenditure', fontsize=9, loc='left')
ax.legend(fontsize=7)

# Panel C: Gap MF-DMC over time — repositioned annotations
ax = axes[1, 0]
for bloc, style in bloc_style.items():
    bloc_data = analysis[analysis['Bloc'] == bloc].groupby('year')['Gap_MF_DMC'].mean()
    ax.plot(bloc_data.index, bloc_data.values, label=bloc, **style)
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Externalisation gap (t/cap)')
ax.text(-0.02, 1.08, 'C', transform=ax.transAxes, fontsize=FONTSIZE_PANEL,
        fontweight='bold', va='top', ha='right')
ax.set_title('  Material externalisation (MF \u2013 DMC)', fontsize=9, loc='left')
ax.legend(loc='upper right', fontsize=7)
# Position annotations away from data lines to prevent overlap
ax.text(0.95, 0.85, 'Net importer', fontsize=7, ha='right', color=EU_COLOR,
        transform=ax.transAxes)
ax.text(0.95, 0.10, 'Net exporter', fontsize=7, ha='right', color=MERCOSUR_COLOR,
        transform=ax.transAxes)

# Panel D: GDP per capita over time
ax = axes[1, 1]
for bloc, style in bloc_style.items():
    bloc_data = analysis[analysis['Bloc'] == bloc].groupby('year')['GDP_pc'].mean()
    ax.plot(bloc_data.index, bloc_data.values/1000, label=bloc, **style)
ax.set_xlabel('Year')
ax.set_ylabel('GDP per capita (thousand USD)')
ax.text(-0.02, 1.08, 'D', transform=ax.transAxes, fontsize=FONTSIZE_PANEL,
        fontweight='bold', va='top', ha='right')
ax.set_title('  GDP per capita (constant 2015 USD)', fontsize=9, loc='left')
ax.legend(loc='upper left', fontsize=7)

plt.tight_layout()
plt.savefig('../figures/fig_extended_temporal.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('../figures/fig_extended_temporal.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figures/fig_extended_temporal.png")

print("\n" + "="*70)
print("ALL FIGURES CREATED")
print("="*70)
print("\nFigures saved in figures/ directory:")
print("  1. fig_extended_coefficients.png - Coefficient comparison (Figure 4)")
print("  2. fig_extended_scatters.png - Scatter plots by GDP tercile (Figure 5)")
print("  3. fig_extended_temporal.png - Temporal trends by bloc (Figure S1)")

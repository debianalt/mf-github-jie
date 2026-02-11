"""
================================================================================
Does Technology Drive Dematerialisation? Software Complexity, Income, and
Material Footprint across EU-27 and MERCOSUR-4
Methodologies: Cross-Sectional Correlations + Sequence Analysis (Optimal Matching)
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.lines import Line2D
import scipy.stats as stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, set_link_color_palette
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')

# Unified publication-quality style (Nature/JIE standard)
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

# Standard font sizes for manual use
FONTSIZE_TITLE = 10
FONTSIZE_LABEL = 9
FONTSIZE_TICK = 8
FONTSIZE_LEGEND = 7.5
FONTSIZE_ANNOT = 8  # Annotations, values on bars/cells
FONTSIZE_PANEL = 11  # Bold panel labels (A, B, C...)

print("="*70)
print("ANALYSIS: Cross-Sectional Correlations + Sequence Analysis")
print("="*70)

#==============================================================================
# 1. DEFINITIONS AND DATA LOADING
#==============================================================================
print("\n[1/8] Loading data...")

COUNTRIES = {
    'AT': ('Austria', 'EU-27'), 'BE': ('Belgium', 'EU-27'), 'BG': ('Bulgaria', 'EU-27'),
    'HR': ('Croatia', 'EU-27'), 'CY': ('Cyprus', 'EU-27'), 'CZ': ('Czech Republic', 'EU-27'),
    'DK': ('Denmark', 'EU-27'), 'EE': ('Estonia', 'EU-27'), 'FI': ('Finland', 'EU-27'),
    'FR': ('France', 'EU-27'), 'DE': ('Germany', 'EU-27'), 'GR': ('Greece', 'EU-27'),
    'HU': ('Hungary', 'EU-27'), 'IE': ('Ireland', 'EU-27'), 'IT': ('Italy', 'EU-27'),
    'LV': ('Latvia', 'EU-27'), 'LT': ('Lithuania', 'EU-27'), 'LU': ('Luxembourg', 'EU-27'),
    'MT': ('Malta', 'EU-27'), 'NL': ('Netherlands', 'EU-27'), 'PL': ('Poland', 'EU-27'),
    'PT': ('Portugal', 'EU-27'), 'RO': ('Romania', 'EU-27'), 'SK': ('Slovakia', 'EU-27'),
    'SI': ('Slovenia', 'EU-27'), 'ES': ('Spain', 'EU-27'), 'SE': ('Sweden', 'EU-27'),
    'AR': ('Argentina', 'MERCOSUR-4'), 'BR': ('Brazil', 'MERCOSUR-4'),
    'PY': ('Paraguay', 'MERCOSUR-4'), 'UY': ('Uruguay', 'MERCOSUR-4')
}

mfa_raw = pd.read_csv('../data/mfa_filled.csv')
github_raw = pd.read_csv('../data/github_summary_by_country.csv')

# Load official R&D data from verified source (World Bank / WIPO 2021)
rd_official = pd.read_csv('../data/rd_patents_official.csv')
rd_data = {}
for _, row in rd_official.iterrows():
    iso = row['iso2_code']
    rd_data[iso] = (row['RD_pct_GDP'], row['Patents_residents'],
                    row['Researchers_per_million'], row['High_tech_exports_pct'])

#==============================================================================
# 2. BUILD CROSS-SECTIONAL DATASET (2024)
#==============================================================================
print("[2/8] Building cross-sectional dataset (2024)...")

mfa_vars_2024 = ['GDP (constant 2015 USD)', 'Material Footprint (RMC) per capita',
                 'Domestic Material Consumption per capita', 'Domestic Extraction per capita',
                 'Population']

rows = []
for iso, (country_name, bloc) in COUNTRIES.items():
    row = {'iso2_code': iso, 'Country': country_name, 'Bloc': bloc}

    country_mfa = mfa_raw[mfa_raw['Country'] == country_name]
    if len(country_mfa) == 0 and country_name == 'Czech Republic':
        country_mfa = mfa_raw[mfa_raw['Country'] == 'Czechia']

    for var in mfa_vars_2024:
        var_data = country_mfa[country_mfa['Flow name'] == var]
        if len(var_data) > 0 and '2024' in var_data.columns:
            val = var_data['2024'].values[0]
            if val > 0:
                row[var] = val

    gh = github_raw[github_raw['iso2_code'] == iso]
    if len(gh) > 0:
        for col in ['developers_2024', 'num_languages', 'pct_web', 'pct_data_ml',
                    'pct_systems', 'pct_mobile', 'pct_devops', 'pct_other']:
            row[col] = gh[col].values[0]

    if iso in rd_data:
        row['RD_pct_GDP'], row['Patents_residents'], row['Researchers_per_million'], row['HighTech_exports'] = rd_data[iso]

    rows.append(row)

df = pd.DataFrame(rows)

df = df.rename(columns={
    'GDP (constant 2015 USD)': 'GDP',
    'Material Footprint (RMC) per capita': 'MF_pc',
    'Domestic Material Consumption per capita': 'DMC_pc',
    'Domestic Extraction per capita': 'DE_pc',
    'developers_2024': 'Developers'
})

df['GDP_pc'] = df['GDP'] / df['Population']
df['DMC_total'] = df['DMC_pc'] * df['Population']
df['Mat_Intensity'] = df['DMC_total'] / df['GDP']
df['Gap_MF_DMC'] = df['MF_pc'] - df['DMC_pc']
df['Devs_per_million'] = df['Developers'] / (df['Population'] / 1e6)
df['Mat_Productivity'] = df['GDP_pc'] / (df['MF_pc'] + 0.001)
df['pct_advanced'] = df['pct_data_ml'] + df['pct_devops'] + df['pct_systems']

print(f"   Dataset: {len(df)} countries (2024 data)")

#==============================================================================
# 3. BUILD TEMPORAL PANEL AND SEQUENCES
#==============================================================================
print("[3/8] Building temporal panel and sequences...")

years = list(range(1999, 2025))  # 1999-2024: 1999 as base year for Tapio elasticity from 2000
panel_rows = []

for iso, (country_name, bloc) in COUNTRIES.items():
    country_mfa = mfa_raw[mfa_raw['Country'] == country_name]
    if len(country_mfa) == 0 and country_name == 'Czech Republic':
        country_mfa = mfa_raw[mfa_raw['Country'] == 'Czechia']
    if len(country_mfa) == 0:
        continue

    for year in years:
        row = {'iso2_code': iso, 'Country': country_name, 'Bloc': bloc, 'Year': year}
        year_str = str(year)

        for var, new_name in [('GDP (constant 2015 USD)', 'GDP'), ('Material Footprint (RMC) per capita', 'MF_pc')]:
            var_data = country_mfa[country_mfa['Flow name'] == var]
            if len(var_data) > 0 and year_str in var_data.columns:
                val = var_data[year_str].values[0]
                if val > 0:
                    row[new_name] = val

        panel_rows.append(row)

panel = pd.DataFrame(panel_rows)
print(f"   Panel: {len(panel)} observations")

# Tapio decoupling states with English labels
STATE_ORDER = ['SD', 'WD', 'EC', 'END', 'RD', 'RC', 'WND', 'SND', 'NA']
STATE_NAMES = {
    'SD': 'Strong Decoupling', 'WD': 'Weak Decoupling',
    'EC': 'Expansive Coupling', 'END': 'Expansive Negative Decoupling',
    'RD': 'Recessive Decoupling', 'RC': 'Recessive Coupling',
    'WND': 'Weak Negative Decoupling', 'SND': 'Strong Negative Decoupling', 'NA': 'No data'
}
STATE_CODES = {s: i for i, s in enumerate(STATE_ORDER)}

# Refined colour palette — visible against white background
STATE_COLORS = [
    '#1A9850',  # SD - Strong decoupling - green
    '#91CF60',  # WD - Weak decoupling - light green
    '#D9D9D9',  # EC - Expansive coupling - visible grey (not white)
    '#FEB24C',  # END - Expansive negative decoupling - orange
    '#66BD63',  # RD - Recessive decoupling - pale green
    '#BDBDBD',  # RC - Recessive coupling - medium grey
    '#FC8D59',  # WND - Weak negative decoupling - orange
    '#D73027',  # SND - Strong negative decoupling - red
    '#FFFFFF'   # NA - white
]

def classify_decoupling(gdp_growth, mf_growth):
    if pd.isna(gdp_growth) or pd.isna(mf_growth) or abs(gdp_growth) < 0.001:
        return 'NA'
    e = mf_growth / gdp_growth
    if gdp_growth > 0:
        if e < 0: return 'SD'
        elif e < 0.8: return 'WD'
        elif e < 1.2: return 'EC'
        else: return 'END'
    else:
        if e < 0: return 'RD'
        elif e < 0.8: return 'WND'
        elif e < 1.2: return 'RC'
        else: return 'SND'

sequences = {}
for iso in panel['iso2_code'].unique():
    cdata = panel[panel['iso2_code'] == iso].sort_values('Year').copy()
    cdata['GDP_growth'] = cdata['GDP'].pct_change()
    cdata['MF_growth'] = cdata['MF_pc'].pct_change()
    cdata['State'] = cdata.apply(lambda r: classify_decoupling(r['GDP_growth'], r['MF_growth']), axis=1)
    sequences[iso] = cdata['State'].tolist()[1:]

seq_matrix = []
seq_labels = []
for iso, seq in sequences.items():
    if len(seq) == len(years) - 1:
        seq_matrix.append([STATE_CODES.get(s, 8) for s in seq])
        seq_labels.append(iso)

seq_matrix = np.array(seq_matrix)
print(f"   Sequences: {len(seq_labels)} countries, {seq_matrix.shape[1]} years")

#==============================================================================
# 4. OPTIMAL MATCHING AND CLUSTERING
#==============================================================================
print("[4/8] Optimal Matching and clustering...")

def optimal_matching_distance(seq1, seq2, subst_cost=2, indel_cost=1):
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n + 1, m + 1))
    for i in range(n + 1): dp[i, 0] = i * indel_cost
    for j in range(m + 1): dp[0, j] = j * indel_cost
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i-1] == seq2[j-1] else subst_cost
            dp[i, j] = min(dp[i-1, j] + indel_cost, dp[i, j-1] + indel_cost, dp[i-1, j-1] + cost)
    return dp[n, m]

n_seq = len(seq_matrix)
dist_matrix = np.zeros((n_seq, n_seq))
for i in range(n_seq):
    for j in range(i+1, n_seq):
        d = optimal_matching_distance(seq_matrix[i], seq_matrix[j])
        dist_matrix[i, j] = dist_matrix[j, i] = d

linkage_matrix = linkage(squareform(dist_matrix), method='ward')
n_clusters = 4
cluster_labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')

cluster_df = pd.DataFrame({'iso2_code': seq_labels, 'Cluster': cluster_labels})
df = df.merge(cluster_df, on='iso2_code', how='left')

# Save cross-sectional dataset with cluster assignments
df.to_csv('../data/data_final_en.csv', index=False)

print(f"   Clusters: {n_clusters}")
for cl in range(1, n_clusters + 1):
    members = [seq_labels[i] for i in range(len(seq_labels)) if cluster_labels[i] == cl]
    blocs = list(set([COUNTRIES[m][1] for m in members]))
    print(f"   Cluster {cl} ({len(members)}): {', '.join(members[:6])}{'...' if len(members)>6 else ''} [{', '.join(blocs)}]")

#==============================================================================
# 5. COMPUTE STATE TRANSITIONS
#==============================================================================
print("[5/8] Computing state transitions...")

# Build transition matrix
n_states = len(STATE_ORDER) - 1  # Exclude NA for transition analysis
transition_counts = np.zeros((n_states, n_states))

for seq in seq_matrix:
    for t in range(len(seq) - 1):
        from_state = seq[t]
        to_state = seq[t + 1]
        if from_state < n_states and to_state < n_states:  # Exclude NA
            transition_counts[from_state, to_state] += 1

# Normalize to probabilities
transition_probs = transition_counts / (transition_counts.sum(axis=1, keepdims=True) + 0.001)

print("   Transition matrix computed")

#==============================================================================
# 6. FIGURES
#==============================================================================
print("[6/8] Generating figures...")

# Refined colour palette for blocs
colors_bloc = {'EU-27': '#2C3E50', 'MERCOSUR-4': '#27AE60'}
markers_bloc = {'EU-27': 'o', 'MERCOSUR-4': '^'}

# UNIFIED cluster colors — consistent across ALL figures
CLUSTER_COLORS = {
    1: '#3C78B5',  # Steel blue - Progressive Trajectories
    2: '#2D8659',  # Forest green - Stable Performers
    3: '#D4820C',  # Amber - Atypical Trajectories (EE, FI, AR)
    4: '#C03030'   # Crimson - Broad Convergence
}
CLUSTER_NAMES = {
    1: 'Progressive Trajectories',
    2: 'Stable Performers',
    3: 'Atypical Trajectories',
    4: 'Broad Convergence'
}
colors_cluster = [CLUSTER_COLORS[i] for i in range(1, 5)]

# FIGURE 1: Cross-Sectional Analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# 1A: Correlation Heatmap
ax = axes[0, 0]
tech_vars = ['Devs_per_million', 'pct_advanced', 'pct_web', 'RD_pct_GDP', 'Patents_residents']
mat_vars = ['GDP_pc', 'MF_pc', 'DMC_pc', 'Gap_MF_DMC', 'Mat_Intensity']
corr_mat = np.zeros((len(tech_vars), len(mat_vars)))
for i, tv in enumerate(tech_vars):
    for j, mv in enumerate(mat_vars):
        valid = df[[tv, mv]].dropna()
        if len(valid) > 5:
            corr_mat[i, j], _ = stats.pearsonr(valid[tv], valid[mv])

# Use diverging colormap
cmap_corr = plt.cm.RdBu_r
im = ax.imshow(corr_mat, cmap=cmap_corr, vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(mat_vars)))
ax.set_yticks(range(len(tech_vars)))
ax.set_xticklabels(['GDP/cap', 'MF/cap', 'DMC/cap', 'Gap', 'Intensity'], fontsize=11)
ax.set_yticklabels(['Devs/M', '%Advanced', '%Web', 'R&D%', 'Patents'], fontsize=11)
for i in range(len(tech_vars)):
    for j in range(len(mat_vars)):
        color = 'white' if abs(corr_mat[i, j]) > 0.5 else 'black'
        ax.text(j, i, f'{corr_mat[i, j]:.2f}', ha='center', va='center', fontsize=11, color=color, fontweight='bold')
ax.set_title('A. Correlation Matrix: Technology vs Material Variables (2024)', fontweight='bold', fontsize=13)
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Pearson r', fontsize=11)

# 1B: Scatter GDP vs Developers by bloc
ax = axes[0, 1]
for bloc in ['EU-27', 'MERCOSUR-4']:
    subset = df[df['Bloc'] == bloc]
    ax.scatter(subset['GDP_pc']/1000, subset['Devs_per_million']/1000, c=colors_bloc[bloc],
               marker=markers_bloc[bloc], s=80, label=bloc, alpha=0.85,
               edgecolors='white', linewidths=1.5)
valid = df[['GDP_pc', 'Devs_per_million']].dropna()
r, p = stats.pearsonr(valid['GDP_pc'], valid['Devs_per_million'])
slope, intercept, _, _, _ = stats.linregress(valid['GDP_pc']/1000, valid['Devs_per_million']/1000)
x_line = np.linspace(valid['GDP_pc'].min()/1000, valid['GDP_pc'].max()/1000, 100)
ax.plot(x_line, slope*x_line + intercept, 'k--', linewidth=2, alpha=0.7)
ax.set_xlabel('GDP per capita (thousand USD)', fontsize=11)
ax.set_ylabel('Developers per million (thousands)', fontsize=11)
ax.set_title(f'B. GDP vs Technological Density (r = {r:.2f}***, 2024)', fontweight='bold', fontsize=13)
ax.legend(loc='upper left', framealpha=0.9)

# 1C: Bloc comparison - Material
ax = axes[1, 0]
blocs = ['EU-27', 'MERCOSUR-4']
x = np.arange(len(blocs))
width = 0.25
mf_vals = [df[df['Bloc']==b]['MF_pc'].mean() for b in blocs]
dmc_vals = [df[df['Bloc']==b]['DMC_pc'].mean() for b in blocs]
de_vals = [df[df['Bloc']==b]['DE_pc'].mean() for b in blocs]

bars1 = ax.bar(x - width, mf_vals, width, label='MF/cap', color='#8B0000', alpha=0.85)
bars2 = ax.bar(x, dmc_vals, width, label='DMC/cap', color='#00008B', alpha=0.85)
bars3 = ax.bar(x + width, de_vals, width, label='DE/cap', color='#006400', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(blocs, fontsize=11)
ax.set_ylabel('Tonnes per capita', fontsize=11)
ax.set_title('C. Material Metabolism Indicators by Bloc (2024)', fontweight='bold', fontsize=13)
ax.legend(loc='upper right', framealpha=0.9)
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', fontsize=11)

# 1D: Bloc comparison - Tech
ax = axes[1, 1]
adv_vals = [df[df['Bloc']==b]['pct_advanced'].mean() for b in blocs]
web_vals = [df[df['Bloc']==b]['pct_web'].mean() for b in blocs]
rd_vals = [df[df['Bloc']==b]['RD_pct_GDP'].mean() for b in blocs]

width = 0.25
bars1 = ax.bar(x - width, adv_vals, width, label='% Advanced Tech', color='#2E8B57', alpha=0.85)
bars2 = ax.bar(x, web_vals, width, label='% Web', color='#FF8C00', alpha=0.85)
bars3 = ax.bar(x + width, [v*10 for v in rd_vals], width, label='R&D % GDP (×10)', color='#4169E1', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(blocs, fontsize=11)
ax.set_ylabel('Percentage', fontsize=11)
ax.set_title('D. Technological Specialisation by Bloc (2024)', fontweight='bold', fontsize=13)
ax.legend(loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.savefig('../figures/fig1_transversal_en.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# FIGURE 2: Sequence Analysis - REORGANIZED for better readability
# Panel C (heatmap) needs to be much taller to read country names

# Figure 2: Temporal distribution + Dendrogram
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Temporal distribution of states
ax = axes[0]
state_counts = np.zeros((len(years)-1, len(STATE_ORDER)))
for seq in seq_matrix:
    for t, state in enumerate(seq):
        state_counts[t, state] += 1
state_props = state_counts / state_counts.sum(axis=1, keepdims=True)

bottom = np.zeros(len(years)-1)
for i, state in enumerate(STATE_ORDER[:-1]):
    ax.bar(years[1:], state_props[:, i], bottom=bottom, label=state, color=STATE_COLORS[i],
           width=0.8, edgecolor='white', linewidth=0.2)
    bottom += state_props[:, i]
ax.set_xlabel('Year')
ax.set_ylabel('Proportion of countries')
ax.text(-0.02, 1.05, 'A', transform=ax.transAxes, fontsize=FONTSIZE_PANEL,
        fontweight='bold', va='top', ha='right')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=7.5, ncol=4, framealpha=0.9)
ax.set_xlim(1999, 2025)
ax.set_ylim(0, 1)

# Panel B: Dendrogram
ax = axes[1]
set_link_color_palette([CLUSTER_COLORS[i] for i in range(1, n_clusters+1)])
dend = dendrogram(linkage_matrix, labels=seq_labels, leaf_rotation=90, ax=ax,
           color_threshold=linkage_matrix[-n_clusters+1, 2], above_threshold_color='0.7')
for coll in ax.collections:
    coll.set_linewidth(1.0)
ax.set_ylabel('OM Distance')
ax.set_xlabel('Country')
ax.text(-0.02, 1.05, 'B', transform=ax.transAxes, fontsize=FONTSIZE_PANEL,
        fontweight='bold', va='top', ha='right')

# Cluster legend below dendrogram
legend_elements = [mpatches.Patch(facecolor=CLUSTER_COLORS[i+1], edgecolor='0.5',
                                   label=f'Cl. {i+1}: {CLUSTER_NAMES[i+1]}') for i in range(n_clusters)]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12),
          fontsize=7.5, ncol=2, framealpha=0.9)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.savefig('../figures/fig2_temporal_dendrogram_en.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Figure 1: Heatmap of sequences — publication quality
fig, ax = plt.subplots(figsize=(14, 16))

sorted_idx = np.argsort(cluster_labels)
sorted_matrix = seq_matrix[sorted_idx]
sorted_labels = [seq_labels[i] for i in sorted_idx]
sorted_clusters = cluster_labels[sorted_idx]

# Get full country names for better labels
sorted_country_names = [COUNTRIES[iso][0] for iso in sorted_labels]

# Create expanded matrix with spacing between countries
n_countries = len(sorted_labels)
n_years = len(years) - 1
row_height = 1.0
spacing = 0.25

# Calculate y-positions for each country (with spacing)
y_positions = []
current_y = 0
for i in range(n_countries):
    y_positions.append(current_y)
    current_y += row_height + spacing
    # Extra spacing at cluster boundaries
    if i < n_countries - 1 and sorted_clusters[i] != sorted_clusters[i + 1]:
        current_y += spacing * 2

# Plot each country as a horizontal bar
for i, (y_pos, seq) in enumerate(zip(y_positions, sorted_matrix)):
    for j, state in enumerate(seq):
        rect = plt.Rectangle((j - 0.5, y_pos), 1, row_height,
                             facecolor=STATE_COLORS[state],
                             edgecolor='white', linewidth=0.2)
        ax.add_patch(rect)

# Set axis limits
ax.set_xlim(-0.5, n_years - 0.5)
ax.set_ylim(-0.5, y_positions[-1] + row_height + 0.5)

# Y-axis: country labels
ax.set_yticks([y + row_height/2 for y in y_positions])
ax.set_yticklabels([f"{sorted_labels[i]}  {sorted_country_names[i][:18]}"
                   for i in range(len(sorted_labels))], fontsize=9, fontweight='medium')

# X-axis: years
ax.set_xticks(range(n_years))
ax.set_xticklabels([str(y) for y in years[1:]], fontsize=8, rotation=45, ha='right')
ax.set_xlabel('Year')
ax.set_ylabel('Country (ordered by cluster)')

# Key year markers (2008 financial crisis, 2020 COVID)
for key_year in [2008, 2020]:
    x_pos = key_year - years[1]
    ax.axvline(x=x_pos, color='0.4', linewidth=0.6, linestyle=':', zorder=0)

# Cluster separators — 1.5pt black lines
prev_cluster = sorted_clusters[0]
for i in range(1, n_countries):
    if sorted_clusters[i] != prev_cluster:
        sep_y = (y_positions[i-1] + row_height + y_positions[i]) / 2
        ax.axhline(y=sep_y, color='black', linewidth=1.5, linestyle='-')
        prev_cluster = sorted_clusters[i]

# Cluster labels on the right side
cluster_start_indices = [0]
for i in range(1, n_countries):
    if sorted_clusters[i] != sorted_clusters[i-1]:
        cluster_start_indices.append(i)
cluster_start_indices.append(n_countries)

for k in range(len(cluster_start_indices) - 1):
    start_i = cluster_start_indices[k]
    end_i = cluster_start_indices[k + 1] - 1
    mid_y = (y_positions[start_i] + y_positions[end_i] + row_height) / 2
    cl_id = sorted_clusters[start_i]
    ax.text(n_years + 0.3, mid_y, f'Cluster {cl_id}', fontsize=9, fontweight='bold',
            va='center', ha='left', color=colors_cluster[cl_id-1])

# Compact legend — no bold, 4 columns, 7.5pt
legend_patches = [mpatches.Patch(color=STATE_COLORS[i], label=f'{STATE_ORDER[i]}: {STATE_NAMES[STATE_ORDER[i]]}')
                  for i in range(len(STATE_ORDER)-1)]
ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fontsize=7.5, ncol=4, framealpha=0.95, edgecolor='0.7')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('../figures/fig1_sequences_heatmap_en.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Figure 2D: Cluster evolution profiles
fig, ax = plt.subplots(figsize=(12, 5))

cluster_profiles = {}
for cl in range(1, n_clusters + 1):
    cl_idx = [i for i, c in enumerate(cluster_labels) if c == cl]
    cl_seqs = seq_matrix[cl_idx]
    good = np.isin(cl_seqs, [0, 1]).mean(axis=0)
    cluster_profiles[cl] = good

for cl, profile in cluster_profiles.items():
    members = [seq_labels[i] for i in range(len(seq_labels)) if cluster_labels[i] == cl]
    blocs_in = [COUNTRIES[m][1] for m in members]
    dominant_bloc = max(set(blocs_in), key=blocs_in.count)
    ax.plot(years[1:], profile, label=f'Cluster {cl} (n={len(members)}, {dominant_bloc})',
            linewidth=3, marker='o', markersize=6, color=colors_cluster[cl-1])

ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Proportion in decoupling (SD+WD)', fontsize=11)
ax.set_title('D. Cluster Evolution: Proportion of Countries in Decoupling States', fontweight='bold', fontsize=13)
ax.legend(loc='best', framealpha=0.9, fontsize=11)
ax.set_ylim(0, 1)
ax.set_xlim(1999, 2025)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../figures/fig2d_cluster_evolution_en.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# FIGURE 3: Clusters and Technology
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 3A: Gap by bloc
ax = axes[0]
gap_vals = [df[df['Bloc']==b]['Gap_MF_DMC'].mean() for b in blocs]
colors_gap = [colors_bloc[b] for b in blocs]
bars = ax.bar(blocs, gap_vals, color=colors_gap, alpha=0.85, edgecolor='black', linewidth=1.5)
ax.axhline(y=0, color='black', linewidth=1.5)
ax.set_ylabel('Gap MF - DMC (t/capita)', fontsize=11)
ax.set_title('A', fontweight='bold', fontsize=14, loc='left')
for bar, val in zip(bars, gap_vals):
    y_pos = val + 0.5 if val > 0 else val - 1.2
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylim(min(gap_vals)-3, max(gap_vals)+3)

# 3B: Cluster composition by bloc
ax = axes[1]
cluster_bloc = df.groupby(['Cluster', 'Bloc']).size().unstack(fill_value=0)
cluster_bloc = cluster_bloc.reindex(columns=['EU-27', 'MERCOSUR-4'], fill_value=0)
cluster_bloc.plot(kind='bar', ax=ax, color=[colors_bloc['EU-27'], colors_bloc['MERCOSUR-4']], alpha=0.85, edgecolor='black')
ax.set_xlabel('Trajectory Cluster', fontsize=11)
ax.set_ylabel('Number of countries', fontsize=11)
ax.set_title('B', fontweight='bold', fontsize=14, loc='left')
ax.legend(title='Bloc', framealpha=0.9)
ax.set_xticklabels([f'Cl.{int(i)}' for i in cluster_bloc.index], rotation=0)

# 3C: Tech profile by cluster
ax = axes[2]
cluster_tech = df.groupby('Cluster')[['pct_advanced', 'pct_web']].mean()
x = np.arange(len(cluster_tech))
width = 0.35
ax.bar(x - width/2, cluster_tech['pct_advanced'], width, label='% Advanced Tech', color='#2E8B57', alpha=0.85, edgecolor='black')
ax.bar(x + width/2, cluster_tech['pct_web'], width, label='% Web', color='#FF8C00', alpha=0.85, edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels([f'Cluster {int(i)}' for i in cluster_tech.index])
ax.set_ylabel('Percentage', fontsize=11)
ax.set_title('C', fontweight='bold', fontsize=14, loc='left')
ax.legend(loc='center', framealpha=0.9, bbox_to_anchor=(0.5, 0.5))

plt.tight_layout()
plt.savefig('../figures/fig3_clusters_tech_en.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# FIGURE 4: TRANSITION ANALYSIS BY BLOC
print("   Computing transitions by bloc...")

def compute_bloc_transitions(bloc_name, bloc_countries):
    """Compute transition matrix for a specific bloc"""
    bloc_seqs = []
    for i, iso in enumerate(seq_labels):
        if COUNTRIES[iso][1] == bloc_name:
            bloc_seqs.append(seq_matrix[i])

    if len(bloc_seqs) == 0:
        return None, None

    bloc_seqs = np.array(bloc_seqs)
    n_states = len(STATE_ORDER) - 1
    trans_counts = np.zeros((n_states, n_states))

    for seq in bloc_seqs:
        for t in range(len(seq) - 1):
            from_s, to_s = seq[t], seq[t + 1]
            if from_s < n_states and to_s < n_states:
                trans_counts[from_s, to_s] += 1

    trans_probs = trans_counts / (trans_counts.sum(axis=1, keepdims=True) + 0.001)
    return trans_counts, trans_probs

# Compute for each bloc
trans_eu_c, trans_eu_p = compute_bloc_transitions('EU-27', [k for k, v in COUNTRIES.items() if v[1] == 'EU-27'])
trans_merc_c, trans_merc_p = compute_bloc_transitions('MERCOSUR-4', [k for k, v in COUNTRIES.items() if v[1] == 'MERCOSUR-4'])

# Figure S2: Comparative transition analysis by bloc
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

state_labels_short = STATE_ORDER[:-1]

# A: Global transition matrix
ax = axes[0, 0]
im = ax.imshow(transition_probs[:len(state_labels_short), :len(state_labels_short)],
               cmap='YlOrRd', vmin=0, vmax=0.5, aspect='auto')
ax.set_xticks(range(len(state_labels_short)))
ax.set_yticks(range(len(state_labels_short)))
ax.set_xticklabels(state_labels_short, fontsize=8, rotation=45)
ax.set_yticklabels(state_labels_short, fontsize=8)
ax.set_xlabel('State at t+1')
ax.set_ylabel('State at t')
ax.text(-0.02, 1.08, 'A', transform=ax.transAxes, fontsize=FONTSIZE_PANEL,
        fontweight='bold', va='top', ha='right')
ax.set_title('  All Countries (n=31)', fontsize=9, loc='left')

for i in range(len(state_labels_short)):
    for j in range(len(state_labels_short)):
        val = transition_probs[i, j]
        if val > 0.05:
            color = 'white' if val > 0.3 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8.5, color=color)
plt.colorbar(im, ax=ax, shrink=0.75, label='Probability')

# B: EU-27 transition matrix
ax = axes[0, 1]
if trans_eu_p is not None:
    im = ax.imshow(trans_eu_p[:len(state_labels_short), :len(state_labels_short)],
                   cmap='Blues', vmin=0, vmax=0.5, aspect='auto')
    ax.set_xticks(range(len(state_labels_short)))
    ax.set_yticks(range(len(state_labels_short)))
    ax.set_xticklabels(state_labels_short, fontsize=8, rotation=45)
    ax.set_yticklabels(state_labels_short, fontsize=8)
    ax.set_xlabel('State at t+1')
    ax.set_ylabel('State at t')
    ax.text(-0.02, 1.08, 'B', transform=ax.transAxes, fontsize=FONTSIZE_PANEL,
            fontweight='bold', va='top', ha='right')
    ax.set_title('  European Union (n=27)', fontsize=9, loc='left')

    for i in range(len(state_labels_short)):
        for j in range(len(state_labels_short)):
            val = trans_eu_p[i, j]
            if val > 0.05:
                color = 'white' if val > 0.3 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8.5, color=color)
    plt.colorbar(im, ax=ax, shrink=0.75, label='Probability')

# C: MERCOSUR-4 transition matrix
ax = axes[1, 0]
if trans_merc_p is not None:
    im = ax.imshow(trans_merc_p[:len(state_labels_short), :len(state_labels_short)],
                   cmap='Greens', vmin=0, vmax=0.5, aspect='auto')
    ax.set_xticks(range(len(state_labels_short)))
    ax.set_yticks(range(len(state_labels_short)))
    ax.set_xticklabels(state_labels_short, fontsize=8, rotation=45)
    ax.set_yticklabels(state_labels_short, fontsize=8)
    ax.set_xlabel('State at t+1')
    ax.set_ylabel('State at t')
    ax.text(-0.02, 1.08, 'C', transform=ax.transAxes, fontsize=FONTSIZE_PANEL,
            fontweight='bold', va='top', ha='right')
    ax.set_title('  MERCOSUR (n=4)', fontsize=9, loc='left')

    for i in range(len(state_labels_short)):
        for j in range(len(state_labels_short)):
            val = trans_merc_p[i, j]
            if val > 0.05:
                color = 'white' if val > 0.3 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8.5, color=color)
    plt.colorbar(im, ax=ax, shrink=0.75, label='Probability')

# D: Key transitions comparison by bloc
ax = axes[1, 1]

def get_key_transitions(trans_p):
    if trans_p is None:
        return [0, 0, 0, 0]
    return [
        trans_p[0, 0],  # SD persistence
        trans_p[1, 0],  # WD -> SD (upgrading)
        trans_p[0, 2] + trans_p[1, 2],  # -> EC (recoupling)
        np.mean([trans_p[2, 2], trans_p[3, 3]]) if trans_p.shape[0] > 3 else trans_p[2, 2]
    ]

metrics_labels = ['SD Persistence', 'Upgrading\n(WD\u2192SD)', 'Recoupling\n(\u2192EC)', 'Coupling\nPersistence']
x = np.arange(len(metrics_labels))
width = 0.25

vals_eu = get_key_transitions(trans_eu_p)
vals_merc = get_key_transitions(trans_merc_p)
vals_all = get_key_transitions(transition_probs)

bars1 = ax.bar(x - width, vals_all, width, label='All (n=31)', color='#666666', alpha=0.85, edgecolor='white', linewidth=0.3)
bars2 = ax.bar(x, vals_eu, width, label='EU-27', color=colors_bloc['EU-27'], alpha=0.85, edgecolor='white', linewidth=0.3)
bars3 = ax.bar(x + width, vals_merc, width, label='MERCOSUR-4', color=colors_bloc['MERCOSUR-4'], alpha=0.85, edgecolor='white', linewidth=0.3)

ax.set_xticks(x)
ax.set_xticklabels(metrics_labels, fontsize=8)
ax.set_ylabel('Probability')
ax.text(-0.02, 1.08, 'D', transform=ax.transAxes, fontsize=FONTSIZE_PANEL,
        fontweight='bold', va='top', ha='right')
ax.set_title('  Key Transitions by Bloc', fontsize=9, loc='left')
ax.legend(loc='upper right', framealpha=0.9, fontsize=7.5)
ax.set_ylim(0, 0.6)

# Value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0.02:
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.2f}',
                    ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig('../figures/fig4_transitions_by_bloc_en.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()


# FIGURE 5: TRANSITION ANALYSIS BY CLUSTER
print("   Computing transitions by cluster...")

def compute_cluster_transitions(cluster_id):
    """Compute transition matrix for a specific trajectory cluster"""
    cl_seqs = []
    for i, iso in enumerate(seq_labels):
        if cluster_labels[i] == cluster_id:
            cl_seqs.append(seq_matrix[i])

    if len(cl_seqs) == 0:
        return None, None, 0

    cl_seqs = np.array(cl_seqs)
    n_st = len(STATE_ORDER) - 1  # Exclude NA
    trans_counts = np.zeros((n_st, n_st))

    for seq in cl_seqs:
        for t in range(len(seq) - 1):
            from_s, to_s = seq[t], seq[t + 1]
            if from_s < n_st and to_s < n_st:
                trans_counts[from_s, to_s] += 1

    row_sums = trans_counts.sum(axis=1, keepdims=True)
    trans_probs = trans_counts / (row_sums + 0.001)
    return trans_counts, trans_probs, len(cl_seqs)

# Compute transitions for each cluster
cluster_trans = {}
for cl in range(1, n_clusters + 1):
    counts, probs, n_members = compute_cluster_transitions(cl)
    cluster_trans[cl] = {'counts': counts, 'probs': probs, 'n': n_members}

# Figure 3: Cluster-based transition analysis (3 rows x 2 cols)
fig = plt.figure(figsize=(12, 15))
gs = fig.add_gridspec(3, 2, hspace=0.30, wspace=0.25)

# Panels A-D: Transition heatmaps for each cluster
panel_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
panel_letters = ['A', 'B', 'C', 'D']
cluster_cmaps = {
    1: LinearSegmentedColormap.from_list('blue_cmap', ['#FFFFFF', '#3C78B5'], N=256),
    2: LinearSegmentedColormap.from_list('green_cmap', ['#FFFFFF', '#2D8659'], N=256),
    3: LinearSegmentedColormap.from_list('orange_cmap', ['#FFFFFF', '#D4820C'], N=256),
    4: LinearSegmentedColormap.from_list('red_cmap', ['#FFFFFF', '#C03030'], N=256),
}

for cl in range(1, n_clusters + 1):
    row, col = panel_positions[cl - 1]
    ax = fig.add_subplot(gs[row, col])
    probs = cluster_trans[cl]['probs']
    n_mem = cluster_trans[cl]['n']

    if probs is not None:
        im = ax.imshow(probs[:len(state_labels_short), :len(state_labels_short)],
                       cmap=cluster_cmaps[cl], vmin=0, vmax=0.6, aspect='auto')
        ax.set_xticks(range(len(state_labels_short)))
        ax.set_yticks(range(len(state_labels_short)))
        ax.set_xticklabels(state_labels_short, fontsize=8, rotation=45)
        ax.set_yticklabels(state_labels_short, fontsize=8)
        ax.set_xlabel('State at t+1')
        ax.set_ylabel('State at t')
        ax.text(-0.02, 1.08, panel_letters[cl-1], transform=ax.transAxes,
                fontsize=FONTSIZE_PANEL, fontweight='bold', va='top', ha='right')
        ax.set_title(f'  Cluster {cl}: {CLUSTER_NAMES[cl]} (n={n_mem})',
                      fontsize=9, color=CLUSTER_COLORS[cl], loc='left')

        for i in range(len(state_labels_short)):
            for j in range(len(state_labels_short)):
                val = probs[i, j]
                if val > 0.05:
                    color = 'white' if val > 0.35 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=FONTSIZE_ANNOT, color=color)
        plt.colorbar(im, ax=ax, shrink=0.75, label='Probability')

# Panel E: Key transitions comparison across clusters
ax_e = fig.add_subplot(gs[2, 0])

metrics_labels_cl = ['SD Persist.', 'Upgrading\n(WD\u2192SD)', 'Recoupling\nRisk (\u2192EC)', 'Coupling\nPersist.']
x_cl = np.arange(len(metrics_labels_cl))
width_cl = 0.18

for cl in range(1, n_clusters + 1):
    probs = cluster_trans[cl]['probs']
    vals = get_key_transitions(probs)
    offset = (cl - 2.5) * width_cl
    bars = ax_e.bar(x_cl + offset, vals, width_cl, label=f'Cl.{cl} ({CLUSTER_NAMES[cl][:8]})',
                    color=CLUSTER_COLORS[cl], alpha=0.85, edgecolor='white', linewidth=0.3)
    for bar in bars:
        h = bar.get_height()
        if h > 0.03:
            ax_e.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.2f}',
                      ha='center', va='bottom', fontsize=7, rotation=0)

ax_e.set_xticks(x_cl)
ax_e.set_xticklabels(metrics_labels_cl, fontsize=8)
ax_e.set_ylabel('Probability')
ax_e.text(-0.02, 1.08, 'E', transform=ax_e.transAxes, fontsize=FONTSIZE_PANEL,
          fontweight='bold', va='top', ha='right')
ax_e.set_title('  Key Transitions by Cluster', fontsize=9, loc='left')
ax_e.legend(loc='upper right', fontsize=7, framealpha=0.9)
ax_e.set_ylim(0, 0.85)

# Panel F: Transition entropy/volatility by cluster
ax_f = fig.add_subplot(gs[2, 1])

# Compute Shannon entropy of transition distributions for each cluster
cluster_entropy = []
cluster_n_transitions = []
for cl in range(1, n_clusters + 1):
    probs = cluster_trans[cl]['probs']
    if probs is not None:
        flat_probs = probs[:len(state_labels_short), :len(state_labels_short)].flatten()
        flat_probs = flat_probs[flat_probs > 0]
        entropy = -np.sum(flat_probs * np.log2(flat_probs + 1e-10))
        cluster_entropy.append(entropy)
        n_distinct = np.sum(probs[:len(state_labels_short), :len(state_labels_short)] > 0.05)
        cluster_n_transitions.append(n_distinct)
    else:
        cluster_entropy.append(0)
        cluster_n_transitions.append(0)

x_ent = np.arange(1, n_clusters + 1)
bar_width = 0.35

bars_ent = ax_f.bar(x_ent - bar_width/2, cluster_entropy, bar_width,
                     color=[CLUSTER_COLORS[cl] for cl in range(1, n_clusters + 1)],
                     alpha=0.85, edgecolor='white', linewidth=0.3, label='Shannon Entropy')

ax_f2 = ax_f.twinx()
bars_trans = ax_f2.bar(x_ent + bar_width/2, cluster_n_transitions, bar_width,
                        color=[CLUSTER_COLORS[cl] for cl in range(1, n_clusters + 1)],
                        alpha=0.40, edgecolor='white', linewidth=0.3, hatch='//',
                        label='Distinct Transitions')

for bar, val in zip(bars_ent, cluster_entropy):
    ax_f.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
              f'{val:.1f}', ha='center', va='bottom', fontsize=7.5)
for bar, val in zip(bars_trans, cluster_n_transitions):
    ax_f2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
               f'{int(val)}', ha='center', va='bottom', fontsize=7.5)

ax_f.set_xticks(x_ent)
ax_f.set_xticklabels([f'Cl.{cl}\n{CLUSTER_NAMES[cl][:10]}' for cl in range(1, n_clusters + 1)], fontsize=8)
ax_f.set_ylabel('Shannon Entropy (bits)')
ax_f2.set_ylabel('N. Distinct Transitions (P > 0.05)')
ax_f.text(-0.02, 1.08, 'F', transform=ax_f.transAxes, fontsize=FONTSIZE_PANEL,
          fontweight='bold', va='top', ha='right')
ax_f.set_title('  Transition Volatility by Cluster', fontsize=9, loc='left')

lines1, labels1 = ax_f.get_legend_handles_labels()
lines2, labels2 = ax_f2.get_legend_handles_labels()
ax_f.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=7, framealpha=0.9,
            bbox_to_anchor=(0.5, -0.15), ncol=2)

plt.savefig('../figures/fig3_transitions_by_cluster_en.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Print cluster transition summary
print("\n   CLUSTER TRANSITION SUMMARY:")
for cl in range(1, n_clusters + 1):
    probs = cluster_trans[cl]['probs']
    vals = get_key_transitions(probs)
    print(f"   Cluster {cl} ({CLUSTER_NAMES[cl]:25s}, n={cluster_trans[cl]['n']:2d}): "
          f"SD_persist={vals[0]:.2f}, Upgrading={vals[1]:.2f}, "
          f"Recoupling={vals[2]:.2f}, Coupling_persist={vals[3]:.2f}, "
          f"Entropy={cluster_entropy[cl-1]:.1f} bits, Distinct_trans={cluster_n_transitions[cl-1]}")

# FIGURE 6: TECHNOLOGY INDICATORS BY CLUSTER (Kruskal-Wallis)
print("   Generating technology-by-cluster boxplots...")

fig, axes = plt.subplots(2, 2, figsize=(9, 7.5))

tech_indicators = [
    ('RD_pct_GDP', 'R&D expenditure (% GDP)'),
    ('Patents_residents', 'Patent applications'),
    ('Researchers_per_million', 'Researchers per million'),
    ('GDP_pc', 'GDP per capita (USD)')
]

panel_letters = ['A', 'B', 'C', 'D']
for idx, (var, label) in enumerate(tech_indicators):
    ax = axes[idx // 2, idx % 2]
    plot_data = []
    plot_labels = []
    for cl in range(1, n_clusters + 1):
        cl_data = df[df['Cluster'] == cl][var].dropna()
        if len(cl_data) > 0:
            plot_data.append(cl_data.values)
            plot_labels.append(f'Cl.{cl}\n(n={len(cl_data)})')

    bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True, widths=0.6,
                    boxprops=dict(linewidth=0.8),
                    whiskerprops=dict(linewidth=0.8),
                    medianprops=dict(linewidth=1.2, color='0.15'),
                    capprops=dict(linewidth=0.8),
                    flierprops=dict(markersize=3))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(CLUSTER_COLORS[i + 1])
        patch.set_alpha(0.6)

    # Jittered strip plot overlay (all individual observations visible, n=31)
    for i, data in enumerate(plot_data):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(data))
        ax.scatter(np.full_like(data, i + 1, dtype=float) + jitter, data,
                   color=CLUSTER_COLORS[i + 1], s=12, alpha=0.7, edgecolor='white',
                   linewidth=0.3, zorder=3)

    # Kruskal-Wallis test
    if len(plot_data) >= 2:
        valid_groups = [g for g in plot_data if len(g) >= 2]
        if len(valid_groups) >= 2:
            h_stat, p_val = stats.kruskal(*valid_groups)
            p_str = f'p < 0.001' if p_val < 0.001 else f'p = {p_val:.3f}'
            sig_label = f'K-W: H = {h_stat:.2f}, {p_str}'
            ax.text(0.05, 0.95, sig_label, transform=ax.transAxes, fontsize=7,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                              edgecolor='0.7', linewidth=0.5))

    ax.set_ylabel(label)
    ax.text(-0.02, 1.08, panel_letters[idx], transform=ax.transAxes,
            fontsize=FONTSIZE_PANEL, fontweight='bold', va='top', ha='right')
    ax.set_title(f'  {label}', fontsize=9, loc='left')

plt.tight_layout()
plt.savefig('../figures/fig_technology_by_cluster.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

#==============================================================================
# 7. SAVE TABLE 1: DESCRIPTIVE STATISTICS BY BLOC
#==============================================================================
print("[7/8] Computing statistics and saving Table 1...")

import os
os.makedirs('../output', exist_ok=True)

table1_rows = []
table1_vars = [('MF_pc', 'MF per capita (t)'), ('DMC_pc', 'DMC per capita (t)'),
               ('Gap_MF_DMC', 'Gap MF-DMC (t/cap)'), ('GDP_pc', 'GDP per capita (USD)'),
               ('RD_pct_GDP', 'R&D (% GDP)'), ('Devs_per_million', 'Developers per million')]
for var, label in table1_vars:
    eu_mean = df[df['Bloc'] == 'EU-27'][var].mean()
    merc_mean = df[df['Bloc'] == 'MERCOSUR-4'][var].mean()
    table1_rows.append({'Metric': label, 'EU-27': round(eu_mean, 2),
                        'MERCOSUR-4': round(merc_mean, 2),
                        'Difference': round(eu_mean - merc_mean, 2)})
pd.DataFrame(table1_rows).to_csv('../output/table1_descriptive_stats.csv', index=False)
print("  Saved: output/table1_descriptive_stats.csv")

print("\n   SIGNIFICANT CORRELATIONS (p < 0.05):")
sig_corrs = []
for i, v1 in enumerate(tech_vars):
    for j, v2 in enumerate(mat_vars):
        valid = df[[v1, v2]].dropna()
        if len(valid) > 5:
            r, p = stats.pearsonr(valid[v1], valid[v2])
            if p < 0.05:
                sig_corrs.append((v1, v2, r, p))
                print(f"   {v1:20} - {v2:15}: r={r:+.3f}, p={p:.4f}")

print("\n   T-TESTS (EU-27 vs MERCOSUR-4):")
test_vars = ['GDP_pc', 'MF_pc', 'Gap_MF_DMC', 'pct_advanced', 'pct_web', 'RD_pct_GDP']
for var in test_vars:
    eu_vals = df[df['Bloc'] == 'EU-27'][var].dropna()
    merc_vals = df[df['Bloc'] == 'MERCOSUR-4'][var].dropna()
    if len(eu_vals) > 1 and len(merc_vals) > 1:
        t, p = stats.ttest_ind(eu_vals, merc_vals)
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        print(f"   {var:15}: EU={eu_vals.mean():.1f}, MERC={merc_vals.mean():.1f}, p={p:.4f} {sig}")

#==============================================================================
# 8. TRANSITION PROBABILITIES SUMMARY
#==============================================================================
print("\n" + "="*70)
print("TRANSITION PROBABILITIES")
print("="*70)

print("\n   Global (n=31):")
print(f"   P(SD->SD) = {transition_probs[0,0]:.2f}")
print(f"   P(WD->WD) = {transition_probs[1,1]:.2f}")
print(f"   P(WD->SD) = {transition_probs[1,0]:.2f}")
print(f"   P(WD->EC) = {transition_probs[1,2]:.2f}")

print("\n   EU-27:")
print(f"   P(SD->SD) = {trans_eu_p[0,0]:.2f}")
print(f"   P(WD->WD) = {trans_eu_p[1,1]:.2f}")
print(f"   P(WD->SD) = {trans_eu_p[1,0]:.2f}")
print(f"   P(WD->EC) = {trans_eu_p[1,2]:.2f}")

print("\n   MERCOSUR-4:")
print(f"   P(SD->SD) = {trans_merc_p[0,0]:.2f}")
print(f"   P(WD->WD) = {trans_merc_p[1,1]:.2f}")
print(f"   P(WD->SD) = {trans_merc_p[1,0]:.2f}")
print(f"   P(WD->EC) = {trans_merc_p[1,2]:.2f}")

#==============================================================================
# COMPLETION
#==============================================================================
print("\n" + "="*70)
print("ANALYSIS COMPLETED")
print("="*70)
print("Files generated:")
print("  - data_final_en.csv (cross-sectional data with cluster assignments)")
print("  - fig1_sequences_heatmap_en.png (Figure 1)")
print("  - fig2_temporal_dendrogram_en.png (Figure 2)")
print("  - fig3_transitions_by_cluster_en.png (Figure 3)")
print("  - fig4_transitions_by_bloc_en.png (Figure S2)")
print("  - fig_technology_by_cluster.png (Figure 6)")
print("  - table1_descriptive_stats.csv (Table 1)")
print("="*70)


# Replication Materials: Does Technology Drive Dematerialisation?

## Overview

This repository contains replication materials and supplementary data for the article "Does Technology Drive Dematerialisation? Software Complexity, Income, and Material Footprint across EU-27 and MERCOSUR-4" submitted to the Journal of Industrial Ecology.

**Author:** Raimundo Elias Gomez
**Affiliations:** CONICET / National University of Misiones (Argentina); Institute of Sociology, University of Porto (Portugal)
**Contact:** elias.gomez@conicet.gov.ar
**ORCID:** 0000-0002-4468-9618

## Repository Structure

```
subirGH/
├── data/                    # Input datasets
│   ├── data_final_en.csv           # Cross-sectional analysis (2024) with cluster assignments
│   ├── mfa_filled.csv              # Material Flow Accounts (UNEP-IRP, 1970-2024)
│   ├── github_summary_by_country.csv   # GitHub developer data by country
│   ├── panel_analysis_1996_2021.csv    # Panel dataset (1996-2021)
│   ├── panel_analysis_results.csv      # Extended panel filtered (2000-2021)
│   └── rd_patents_official.csv     # R&D and patent data (World Bank WDI)
├── scripts/                 # Analysis scripts (Python)
│   ├── 01_sequence_analysis.py     # Sequence Analysis + cross-sectional figures
│   ├── 02_extended_panel_analysis.py   # Extended panel regressions (2000-2021)
│   └── 03_extended_figures.py      # Extended analysis figures
├── output/                  # Regression tables (CSV)
│   ├── table1_descriptive_stats.csv    # Table 1: Descriptive statistics by bloc (2024)
│   ├── table2_eci_regressions.csv      # Table 2 Panel A: ECIsoftware regressions (2020-2024, pre-computed)
│   └── table3_extended_regressions.csv # Table 2 Panel B: Extended panel regressions (2000-2021)
├── figures/                 # All figures (article + supplementary)
├── CODEBOOK.md              # Variable definitions
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
└── README.md                # This file
```

## Data Sources

| Dataset | Source | Period | Description |
|---------|--------|--------|-------------|
| Material Flow Accounts | UNEP-IRP Global Material Flows Database (GLORIA MRIO) | 1970-2024 | MF, DMC, DE per capita (2022-2024 based on economic proxies) |
| GitHub Data | GitHub Innovation Graph | 2020-2024 | Developers by country and programming language (quarterly) |
| R&D, Patents, Researchers | World Bank WDI | 2000-2021 | R&D expenditure, patent applications, researchers per million, high-tech exports |
| GDP | World Bank WDI | 1970-2024 | GDP constant 2015 USD |

## Methodology

### 1. Sequence Analysis (Abbott, 1995)

25-year trajectories (2000-2024) for 31 countries. Each country-year is classified into decoupling states following Tapio's (2005) taxonomy based on the elasticity of material footprint change to GDP change (*e* = %Delta MF / %Delta GDP):

**Expansive states (GDP growth > 0):**
- **Strong Decoupling (SD):** e < 0
- **Weak Decoupling (WD):** 0 <= e < 0.8
- **Expansive Coupling (EC):** 0.8 <= e < 1.2
- **Expansive Negative Decoupling (END):** e >= 1.2

**Recessive states (GDP growth < 0):**
- **Recessive Decoupling (RD), Recessive Coupling (RC), Weak Negative Decoupling (WND), Strong Negative Decoupling (SND)** with symmetric thresholds

Sequence distance computed via Optimal Matching with substitution cost = 2 and indel cost = 1. Hierarchical clustering (Ward's method, k = 4) identifies trajectory typologies based on similarity in temporal decoupling patterns.

### 2. Panel Regressions

Two analytical samples:
- **Short panel (2020-2024, N = 155):** Tests ECIsoftware derived from GitHub Innovation Graph
- **Extended panel (2000-2021, N = 430-682):** Tests four World Bank technology indicators (R&D, patents, high-tech exports, researchers)

All models use pooled OLS with cluster-robust standard errors (clustered by country).

### 3. State Transition Analysis

Markov transition probabilities computed from year-to-year state changes, disaggregated by economic bloc and by trajectory cluster.

## Reproduction Instructions

### Requirements

```bash
pip install -r requirements.txt
```

### Execution Order

```bash
# 1. Main sequence analysis, cross-sectional results, and transition figures
python scripts/01_sequence_analysis.py

# 2. Extended panel regressions (2000-2021)
python scripts/02_extended_panel_analysis.py

# 3. Extended analysis figures
python scripts/03_extended_figures.py
```

### Expected Outputs

- `data/data_final_en.csv`: Cross-sectional dataset with cluster assignments
- `data/panel_analysis_results.csv`: Filtered panel dataset (2000-2021)
- `output/table1_descriptive_stats.csv`: Table 1 (descriptive statistics by bloc)
- `output/table3_extended_regressions.csv`: Table 2 Panel B (extended panel regressions)
- `figures/fig*.png`: All figures (article and supplementary)
- Console output: Regression coefficients, transition probabilities, and test statistics

**Note:** `output/table2_eci_regressions.csv` (Table 2 Panel A: ECI*software* regressions) is included as pre-computed output from the primary analysis pipeline, which requires the ECI*software* computation not included in this repository (see Juhász et al., 2026 for methodology).

## Sample (N = 31)

**EU-27:** Austria, Belgium, Bulgaria, Croatia, Cyprus, Czech Republic, Denmark, Estonia, Finland, France, Germany, Greece, Hungary, Ireland, Italy, Latvia, Lithuania, Luxembourg, Malta, Netherlands, Poland, Portugal, Romania, Slovakia, Slovenia, Spain, Sweden

**MERCOSUR-4:** Argentina, Brazil, Paraguay, Uruguay

## Trajectory Clusters

Clusters group countries by similarity in *temporal sequences of decoupling states* (2000-2024), **not** by absolute income, GDP, or development levels. Countries with vastly different economic conditions can exhibit similar temporal patterns.

| Cluster | Label | n | Countries |
|---------|-------|---|-----------|
| 1 | Progressive Trajectories | 8 | BR, CY, DK, IE, NL, PY, PL, SE |
| 2 | Stable Performers | 7 | HR, FR, GR, MT, PT, ES, UY |
| 3 | Atypical Trajectories | 3 | AR, EE, FI |
| 4 | Broad Convergence | 13 | AT, BE, BG, CZ, DE, HU, IT, LV, LT, LU, RO, SK, SI |

MERCOSUR members are distributed across three of four clusters: Brazil and Paraguay in Cluster 1 alongside Nordic and small open EU economies, Uruguay in Cluster 2 alongside Southern European economies, and Argentina in Cluster 3 with Estonia and Finland.

## Figure Mapping

| Article Figure | File | Description |
|----------------|------|-------------|
| Figure 1 | `fig1_sequences_heatmap_en.png` | Decoupling state sequences by country (2000-2024) |
| Figure 2 | `fig2_temporal_dendrogram_en.png` | Hierarchical clustering dendrogram |
| Figure 3 | `fig3_transitions_by_cluster_en.png` | Transition dynamics by trajectory cluster |
| Figure 4 | `fig_extended_coefficients.png` | Technology indicator effects: confounding by income |
| Figure 5 | `fig_extended_scatters.png` | Technology indicators vs Material Footprint by GDP tercile |
| Figure 6 | `fig_technology_by_cluster.png` | Technology indicators by trajectory cluster |
| **Figure S1** | `fig_extended_temporal.png` | **Supplementary: EU-27 vs MERCOSUR-4 temporal trends (2000-2021)** |
| **Figure S2** | `fig4_transitions_by_bloc_en.png` | **Supplementary: State transition probabilities by bloc** |

## Key Results

1. **Confounding by income:** Three of four World Bank technology indicators (R&D, patents, researchers) show significant positive associations with Material Footprint when entered alone but become non-significant when GDP per capita is controlled (p = 0.20–0.37); ECI*software* (p = 0.55) and high-technology exports (p = 0.19) never reach significance even alone.

2. **GDP per capita explains 46–48%** of variance in Material Footprint in the short panel (2020–2024) and **54–57%** in the extended panel (2000–2021).

3. **Trajectory heterogeneity:** MERCOSUR members are distributed across three of four trajectory clusters, challenging bloc-level generalisations about decoupling dynamics.

## References

Abbott, A. (1995). Sequence analysis: New methods for old ideas. *Annual Review of Sociology*, 21, 93-113. https://doi.org/10.1146/annurev.so.21.080195.000521

Tapio, P. (2005). Towards a theory of decoupling: Degrees of decoupling in the EU and the case of road traffic in Finland between 1970 and 2001. *Transport Policy*, 12(2), 137-151. https://doi.org/10.1016/j.tranpol.2005.01.001

Hidalgo, C. A., & Hausmann, R. (2009). The building blocks of economic complexity. *Proceedings of the National Academy of Sciences*, 106(26), 10570-10575. https://doi.org/10.1073/pnas.0900943106

Juhász, S., Wachs, J., Kaminski, J., & Hidalgo, C. A. (2026). The software complexity of nations. *Research Policy*, 55, 105422. https://doi.org/10.1016/j.respol.2026.105422

## License

MIT License - See LICENSE file

## Citation

If you use these materials, please cite:

```
Gomez, R. E. (2026). Does Technology Drive Dematerialisation? Software Complexity,
Income, and Material Footprint across EU-27 and MERCOSUR-4. Journal of Industrial Ecology.
```

**Zenodo DOI:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18612284.svg)](https://doi.org/10.5281/zenodo.18612284)

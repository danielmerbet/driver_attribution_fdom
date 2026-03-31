# Driver Attribution of Fluorescent Dissolved Organic Matter (fDOM)

![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue)
![Language](https://img.shields.io/badge/language-R%20%7C%20Python-orange)
![Status](https://img.shields.io/badge/status-research-green)

Repository containing **data and machine learning workflows** used to identify environmental drivers controlling **fluorescent dissolved organic matter (fDOM)** dynamics in freshwater systems:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19354955.svg)](https://doi.org/10.5281/zenodo.19354955)

The repository accompanies the preprint:

**A machine learning approach to driver attribution of dissolved organic matter dynamics in two contrasting freshwater systems**

[https://doi.org/10.5194/egusphere-2025-4049](https://doi.org/10.5194/egusphere-2025-4049)

---

# Overview

Dissolved organic matter (DOM) dynamics are influenced by multiple environmental drivers including hydrology, meteorology, and seasonal cycles. This repository provides:

* Data used in the study
* Machine learning workflows
* Feature importance analysis
* SHAP interpretation of models
* Scripts to reproduce all results

The workflow combines multiple machine learning algorithms:

* Random Forest
* XGBoost
* LightGBM
* CatBoost
* Kernel methods
* k-nearest neighbors

These models are used to identify the most influential drivers controlling **fDOM variability** across two contrasting freshwater systems.

---

# Study Sites

Two study sites were analyzed:

| Site          | Country | Description                                           |
| ------------- | ------- | ----------------------------------------------------- |
| Feeagh        | Ireland | humic oligotrophic lake with a peatland-dominated catchment and temperate oceanic climate |
| Sau Reservoir | Spain   | eutrophic reservoir with a human-influenced catchment and Mediterranean climate  |

---

# Repository Structure

```
driver_attribution_fdom/
│
├── README.md
│
├── 1_hyperparameter_tuning.R
│     Hyperparameter optimization for all ML models
│
├── 2_MLrun_most-influential-features.R
│     ML simulations using selected drivers
│
├── 3_MLrun_reanalysis-julianday.R
│     ML simulations using reanalysis meteorology + seasonal predictors
│
├── 4_extract_importance.R
│     Extraction of feature importance across ML models
│
├── 5_shap_analysis.py
│     SHAP analysis for model interpretation
│
├── feeagh/
│     ├── data/
│     └── output/
│
├── sau/
│     ├── data/
│     └── output/
│
├── figures/
│     Figures used in the manuscript
│
├── codes_supplementary/
│     Additional scripts used in supplementary analyses
│
└── old_codes/
      Archived development scripts
```

---

# Workflow

The full analysis pipeline consists of five main steps.

```
Data preprocessing
        │
        ▼
Hyperparameter tuning
        │
        ▼
Machine learning simulations
        │
        ▼
Feature importance extraction
        │
        ▼
SHAP model interpretation
```

Scripts should be executed sequentially:

```
1 → 2 → 3 → 4 → 5
```

---

# Installation

## Clone repository

```bash
git clone https://github.com/danielmerbet/driver_attribution_fdom.git
cd driver_attribution_fdom
```

---

# Software Requirements

The analysis was developed using **R and Python**.

## R packages

```
lubridate
dplyr
caret
randomForest
xgboost
lightgbm
catboost
kknn
kernlab
DEoptim
pdp
ggplot2
```

Install them with:

```r
install.packages(c(
"lubridate","dplyr","caret","randomForest","xgboost",
"lightgbm","catboost","kknn","kernlab","DEoptim",
"pdp","ggplot2"))
```

---

## Python packages

```
shap
xgboost
pandas
matplotlib
lightgbm
catboost
```

Install with:

```bash
pip install shap xgboost pandas matplotlib lightgbm catboost
```

---

# Running the Analysis

To reproduce the full workflow:

### Step 1 — Hyperparameter tuning

```
1_hyperparameter_tuning.R
```

### Step 2 — ML simulation (most influential drivers)

```
2_MLrun_most-influential-features.R
```

### Step 3 — ML simulation (reanalysis + Julian day)

```
3_MLrun_reanalysis-julianday.R
```

### Step 4 — Feature importance extraction

```
4_extract_importance.R
```

### Step 5 — Model interpretation with SHAP

```
5_shap_analysis.py
```

---

# Reproducibility

This repository contains:

* raw and processed datasets
* scripts used for analysis
* scripts for generating figures

---

# Citation

If you use this repository, please cite:

Mercado-Bettín, D., Paíz, R., McCarthy, V., Jennings, E., de Eyto, E., Gallegos, A. M., ... & Marcé, R. (2025). A machine learning approach to driver attribution of dissolved organic matter dynamics in two contrasting freshwater systems. EGUsphere, 2025, 1-26.

[https://doi.org/10.5194/egusphere-2025-4049](https://doi.org/10.5194/egusphere-2025-4049)

---

# Metadata (Dublin Core)

| Element     | Value                                                                                   |
| ----------- | --------------------------------------------------------------------------------------- |
| Title       | Driver Attribution of FDOM Dataset and Analysis                                         |
| Creator     | Daniel Mercado Bettin; Ricardo Marroquín                                                |
| Subject     | Fluorescent Dissolved Organic Matter, water quality, limnology                          |
| Description | Code and data used to identify drivers controlling FDOM dynamics using machine learning |
| Publisher   | CSIC; Dublin City University                                                            |
| Type        | Dataset / Research Code                                                                 |
| Format      | `.R`, `.py`, `.csv`                                                                     |
| Language    | English                                                                                 |
| Coverage    | Case studies in Spain and Ireland                                                       |
| Rights      | CC BY-NC 4.0                                                                            |

---

# License

This repository is licensed under the **Creative Commons Attribution–NonCommercial 4.0 International License**.

[https://creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/)

---


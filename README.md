# driver_attribution_fdom

This repository contains code and data for analyzing the drivers of FDOM (Fluorescent Dissolved Organic Matter) attribution. It contails all codes and data used in the preprint: [**A machine learning approach to driver attribution of dissolved organic matter dynamics in two contrasting freshwater systems**](https://doi.org/10.5194/egusphere-2025-4049) 

---

## Metadata (Dublin Core)

Below is a metadata description for this repository using the **Dublin Core** metadata standard. The [Dublin Core Metadata Element Set](https://www.dublincore.org/specifications/dublin-core/dces/) is a widely adopted, generic schema for describing digital resources.

| Dublin Core Element | Value |
|---|---|
| **Title** | Driver Attribution of FDOM Dataset and Analysis |
| **Creator** | Daniel Mercado Bettin | Ricardo Marroquín
| **Subject** | Fluorescent Dissolved Organic Matter, water quality, environmental drivers, limnology |
| **Description** | Code and data to identify and quantify the drivers behind FDOM variability using statistical and machine-learning methods. This includes scripts, analysis notebooks, and raw/processed data. |
| **Publisher** | CSIC and DCU |
| **Date** | 2025-08-01 (or the date when version was published) |
| **Type** | Dataset / Research Code |
| **Format** | Directory of `.R`, `.py` scripts; CSV data files
| **Identifier** | `https://github.com/danielmerbet/driver_attribution_fdom` |
| **Language** | English |
| **Relation** | A machine learning approach to driver attribution of dissolved organic matter dynamics in two contrasting freshwater systems: https://doi.org/10.5194/egusphere-2025-4049|
| **Coverage** | Daily resolution, applied to two case studies in Spain and Ireland, but could be applied globally |
| **Rights** | CC BY-NC 4.0 license |

---

## Getting Started

1. **Clone the repository**  
   ```bash
   git clone https://github.com/danielmerbet/driver_attribution_fdom.git  
   cd driver_attribution_fdom  
````

2. **Environment setup: libraries needed to run**

   * The codes are designed to be run in the terminal or CMD, there is a commented option in the codes to also run from RStudio
   * R: lubridate, dplyr, caret, randomForest, xgboost, lightgbm, catboost, kknn, kernlab, DEoptim, pdp, ggplot2
   * Python: shap, xgboost, pandas, matplotlib.pyplot, os, lightgbm, catboost


3. **Run analyses**

   * Use the scripts in `/code/` to process data
   * Notebooks in `/notebooks/` show step-by-step analysis (THIS WILL BE PROVIDED ONCE THE PUBLICATION IS ACCEPTED)

---

## Directory Structure

```
driver_attribution_fdom/
├── README.md                # This file  
├── feeagh/data/             # Raw and processed data for the first study site
├── sau/data/                # Raw and processed data for the second study site
├── codes                    # Scripts (R, Python) 
├── feaagh/output/           # Output figures and tables for the first study site   
├── sau/output/              # Output figures and tables for the secod study site  
├── figures/                 # Figures used in the publication
├── notebooks/               # Analysis notebooks (THIS WILL BE PROVIDED AFTER THE ACCEPTANCE OF PUBLICATION)
├── catboost_info            # Auxiliary folder to run catboost method  
└── old_codes                # Back-up folder with previuos codes used  
```

---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

---

## Contact

If you have any questions or want to contribute, please contact the repository owners.

```

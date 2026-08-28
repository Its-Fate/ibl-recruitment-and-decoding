# Temporal Dynamics of Population Decoding

This repository investigates how neural populations across the mouse brain encode sensory inputs and spatial decisions. Utilizing the International Brain Laboratory (IBL) dataset, this project maps the spatiotemporal dynamics of stimulus contrast and spatial choice through predictive modeling and statistical analysis.

## Quick Start (Pre-computed Results)
The full population decoding pipeline takes ~30 minutes to execute. To bypass the model training and instantly generate the visualizations:
1. Download the pre-computed results from [here](https://drive.google.com/drive/folders/1gmy2McFCOGCkHYE4SXjiJZiHeHyDGb2Q?usp=sharing).
2. Place the `.csv` files directly into the `results/` directory.
3. Run `notebooks/population_decoding.ipynb`.

## Core Machine Learning Architecture (`src/`)
This repository features a highly modular, production-ready machine learning pipeline designed to decode stimulus side and contrast directly from population spike trains:

* **Sliding Window Logistic Regression:** Implements an $L_2$-regularized Logistic Regression model over discrete time bins relative to stimulus onset. 
* **Rigorous Validation:** Evaluates decoding accuracy using a 5-fold Stratified K-Fold cross-validation strategy, balancing class weights to account for behavioral asymmetries.
* **Statistical Correction:** Uses a 1-sample t-test against theoretical chance, controlled for multiple comparisons using the False Discovery Rate (FDR/Benjamini-Hochberg) across all region-time pairs.

## Group Analyses To Be Added
Alongside predictive decoding pipeline, this repository will house complementary analyses conducted by the research team:

* **`brain_wide_recruitment/`**: Investigates chronological recruitment by calculating first-response sensory latencies to track signal propagation.
* **`regional_feature_selectivity/`**: Evaluates how neural populations represent distinct variables utilizing Principal Component Analysis (PCA).

## Repository Structure
```text
├── src/                                
│   ├── data_utils.py          # ONE API data querying and PSTH formatting
│   ├── preprocessing.py       # Trial splitting and window matrix generation
│   ├── decoders.py            # Logistic Regression and cross-validation logic
│   ├── stats.py               # Statistical validation and FDR correction
│   └── plotting.py            # Swanson flatmaps and temporal visualization
├── notebooks/                          
│   └── population_decoding.ipynb    # Core visualization and results
├── results/                            # Pre-computed decoding outputs (.csv)
├── brain_wide_recruitment/          # Chronological recruitment logic
├── regional_feature_selectivity/    # PCA feature extraction logic
├── .gitignore                          # Excludes large data files and __pycache__
└── requirements.txt                    # Project dependencies

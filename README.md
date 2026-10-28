# Emotional Neural Network (ENN) for predicting sustainable e-commerce delivery preferences

## Contents

`ENN_nested_cv.ipynb` – a single notebook (Google Colab) with:

1. setup and random seeds,
2. survey columns, input sets MENN1–MENN3 and the target,
3. the ENN for classification (adapted from Nourani et al., 2020, Eq. 8-15; full formulation in Appendix B of the paper), the ablated ENN without hormones and the five-seed ensemble,
4. repeated nested cross-validation (5 × 5 outer folds, 3-fold inner tuning) for the ENN, the ablated ENN, logistic regression, a decision tree and a random forest,
5. results (mean ± SD and 95% confidence intervals based on the corrected resampled t-statistic) and paired tests (corrected resampled t-test and Wilcoxon test with Holm correction).

## Usage

Open the notebook in Google Colab, upload the survey file and set these variables in Section 2.

- `DATA_FILE` – name of the survey file
- `INPUTS` – `MENN1`, `MENN2`, `MENN3` or `MENN3_without3`
- `TARGET_TYPE` – `binary`, `binary_alt` or `multiclass`

The per-fold results are saved to `results_<INPUTS>_<TARGET_TYPE>.csv`.

## Data

The survey data are not publicly available. They can be obtained from the corresponding author upon reasonable request.

## Settings used in the paper

- Outer CV: `RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)`; inner CV: `StratifiedKFold(3, shuffle=True, random_state=42)`, selection by macro F1.
- Preprocessing (fitted on training data only): median imputation; min–max scaling for the ENN, standardisation for the reference models.
- Balanced class weights and a decision threshold of 0.5 for all models.
- ENN ensemble seeds: 42-46; hyperparameters are tuned on a single network.

## Requirements

Python 3 with numpy, pandas, scipy, scikit-learn, numba and openpyxl. In Google Colab only numba and openpyxl need to be installed, which is done in the first cell of the notebook.

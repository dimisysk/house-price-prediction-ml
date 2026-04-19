# House Price Prediction with Ensemble Learning

## Overview

This project predicts residential house prices using supervised machine learning models on the Ames Housing dataset.

The main objective is to build a reproducible regression pipeline, compare baseline and ensemble models, evaluate them with cross-validation, and generate final predictions for unseen test data.

The workflow includes exploratory data analysis, preprocessing, model comparison, feature importance analysis, and submission generation.

---

## Project Objective

The goal is to predict the `SalePrice` of residential homes based on property-related features.

Due to strong positive skewness in the target variable, a `log1p` transformation is applied, and models are evaluated using RMSE on the transformed target, which is aligned with RMSLE.

---

## Dataset

- `train.csv` (1460 rows, 81 columns)
- `test.csv` (1459 rows, 80 columns)

### Target Variable
- `SalePrice`

### Feature Summary
- Numeric features: 36
- Categorical features: 43

### Missing Values

High missingness appears in:

- `PoolQC`
- `MiscFeature`
- `Alley`
- `Fence`
- `MasVnrType`
- `FireplaceQu`
- `LotFrontage`

Handling strategy:
- Numeric features: median imputation
- Categorical features: filled with `"None"`

---

## Exploratory Data Analysis

The target variable is strongly right-skewed.

### Target Summary

- Mean: `$180,921.20`
- Median: `$163,000.00`
- Standard deviation: `$79,442.50`
- Skewness: `1.8829`
- Minimum: `$34,900.00`
- Maximum: `$755,000.00`

This justifies applying a log transformation before training.

---

## Preprocessing

A `ColumnTransformer` is used to handle different feature types within a unified pipeline.

### Numeric Features
- Median imputation

### Categorical Features
- Missing value imputation with `"None"`
- One-hot encoding with `handle_unknown="ignore"`

This preprocessing setup is applied consistently across all evaluated models.

---

## Models Evaluated

The following regression models were compared:

- Dummy Regressor
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

All models were evaluated using 5-fold cross-validation.

---

## Evaluation Metric

The project uses:

- `log1p(SalePrice)` target transformation
- RMSE on the transformed target

This is effectively aligned with RMSLE and is suitable for skewed house price data.

---

## Model Performance

| Model | RMSE | Std | Runtime (sec) |
|------|------:|------:|------:|
| Dummy | 0.3992 | 0.0159 | 0.46 |
| Linear Regression | 0.1460 | 0.0224 | 0.93 |
| Decision Tree | 0.1940 | 0.0105 | 0.80 |
| Random Forest | 0.1433 | 0.0097 | 20.81 |
| Gradient Boosting | 0.1230 | 0.0083 | 27.49 |
| XGBoost | 0.1289 | 0.0109 | 11.13 |

### Best Model

The best-performing model was the **Gradient Boosting Regressor**, with an RMSE of **0.1230**.

This model was selected for the final prediction pipeline.

---

## Key Findings

Permutation importance analysis identified the following features as the most influential:

1. `GrLivArea`
2. `OverallQual`
3. `TotalBsmtSF`
4. `YearBuilt`
5. `OverallCond`
6. `YearRemodAdd`
7. `LotArea`
8. `BsmtFinSF1`
9. `GarageCars`
10. `MSZoning`

These results make practical sense, since living area, overall quality, basement size, renovation history, and garage capacity are all strong drivers of housing prices.

---

## Final Output

The final model was trained on the full training dataset and used to generate predictions for the unseen test dataset.

Example submission format:

```csv
Id,SalePrice
1461,120150.88
1462,155264.30
1463,185957.26
```

## Submission Summary

- Total predictions: 1459
- Mean predicted price: `$177,365.52`
- Median predicted price: `$157,064.59`
- Minimum predicted price: `$41,495.62`
- Maximum predicted price: `$584,231.44`

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- XGBoost

---

## Project Structure

```bash
house-price-prediction-ml/
├── data/
├── docs/
├── outputs/
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run

### Clone the repository

```bash
git clone https://github.com/dimisysk/house-price-prediction-ml.git
cd house-price-prediction-ml
```

### Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the project

```bash
python main.py
```

---

## Future Improvements

Possible next steps include:

- hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV`
- feature engineering based on domain knowledge
- outlier handling
- model stacking or blending
- SHAP-based interpretability
- modular project restructuring with `src/`, `notebooks/`, and `models/`

---

## Author

**Dimitrios Syskakis**

GitHub: [dimisysk](https://github.com/dimisysk)
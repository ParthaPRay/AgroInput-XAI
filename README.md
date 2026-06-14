# AgroInput-XAI

**AgroInput-XAI: Explainable Modelling of Crop Yield, Fertilizer-Use Efficiency, and Pesticide-Use Efficiency under Soil--Weather Variability}**

AgroInput-XAI is an explainable machine learning framework for modelling crop yield and sustainable input-efficiency classes using integrated crop-yield, soil, and weather data. The framework combines crop production records, state-level soil properties, annual weather descriptors, feature engineering, statistical testing, regression modelling, classification modelling, and explainable artificial intelligence (XAI) methods.

The project is designed to support interpretable analysis of crop yield, fertilizer-use efficiency, pesticide-use efficiency, and input-efficiency behaviour under heterogeneous crop, soil, weather, seasonal, temporal, and spatial conditions.

Repository:  
<https://github.com/ParthaPRay/AgroInput-XAI>

---

## Overview

Sustainable agriculture requires crop production systems that maintain yield while improving input-use efficiency. Fertilizer and pesticide inputs can increase productivity, but excessive or poorly matched input use may reduce efficiency and increase environmental pressure. AgroInput-XAI addresses this problem by integrating crop-yield, soil, and weather data into a reproducible machine learning and explainability pipeline.

The framework performs two main predictive tasks:

1. **Regression task:** prediction of log-transformed crop yield.
2. **Classification task:** prediction of sustainable input-efficiency class.

The four input-efficiency classes are:

- **Efficient:** high yield with low fertilizer intensity
- **Intensive:** high yield with high fertilizer intensity
- **Inefficient:** low yield with high fertilizer intensity
- **Low-input low-output:** low yield with low fertilizer intensity

---

## Framework

The overall AgroInput-XAI workflow includes:

1. Input data collection  
2. Data cleaning and integration  
3. Feature engineering  
4. Statistical and exploratory analysis  
5. Regression and classification modelling  
6. Explainable AI analysis  
7. Final decision-support outputs  

If the framework figure is included in the repository, it can be displayed as:

```markdown
![AgroInput-XAI Framework](assets/agroinput_xai_framework_flowchart.png)
````

---

## Dataset

The project uses three datasets:

| Dataset         |      Shape | Main variables                                                            | Integration key |
| --------------- | ---------: | ------------------------------------------------------------------------- | --------------- |
| Crop-yield data | 19,689 × 9 | crop, year, season, state, area, production, fertilizer, pesticide, yield | state, year     |
| Weather data    |    720 × 5 | state, year, average temperature, total rainfall, average humidity        | state, year     |
| Soil data       |     30 × 5 | state, nitrogen, phosphorus, potassium, pH                                | state           |

After integration, the merged dataset contains:

* **19,689 records**
* **30 states**
* **55 crops**
* **6 seasons**
* **24 years**
* **34 total columns after feature engineering**

The dataset source used in the notebook is:

[https://www.kaggle.com/datasets/anshumish/crop-yield-data-with-soil-and-weather-dataset/](https://www.kaggle.com/datasets/anshumish/crop-yield-data-with-soil-and-weather-dataset/)

---

## Important Note on Target Leakage

The variable `production` is deliberately excluded from the predictive feature set because crop yield is generally derived from production and cultivated area. Including `production` as an input feature may introduce target leakage and artificially inflate model performance.

---

## Feature Engineering

AgroInput-XAI generates the following feature groups:

### Input-intensity features

* `fertilizer_per_ha`
* `pesticide_per_ha`

### Log-transformed features

* `log_area`
* `log_fertilizer`
* `log_pesticide`
* `log_fertilizer_per_ha`
* `log_pesticide_per_ha`
* `log_yield`

### Soil nutrient indicators

* `npk_sum`
* `n_p_ratio`
* `n_k_ratio`
* `p_k_ratio`

### Weather-ratio indicators

* `rainfall_temp_ratio`
* `humidity_temp_ratio`

### Input-efficiency indicators

* `fertilizer_use_efficiency`
* `pesticide_use_efficiency`
* `log_fue`
* `log_pue`

### Classification target

* `efficiency_class`

---

## Machine Learning Tasks

### 1. Crop Yield Regression

The regression target is:

```text
log_yield
```

The regression models evaluated are:

* Ridge
* ElasticNet
* Random Forest
* Extra Trees
* Gradient Boosting
* HistGradientBoosting
* XGBoost
* LightGBM
* CatBoost
* MLP Regressor

### Best Regression Model

Under the random 80:20 train-test split, the best regression model is:

```text
ExtraTrees
```

Main performance values:

| Metric             |   Value |
| ------------------ | ------: |
| MAE                |   0.088 |
| RMSE               |   0.164 |
| Median AE          |   0.044 |
| R²                 |   0.979 |
| Explained variance |   0.979 |
| MAPE               | 16.372% |

### Temporal Validation

Temporal validation uses:

```text
Train years: 1997–2016
Test years: 2017–2020
```

Best temporal model:

```text
ExtraTrees
```

Performance:

| Metric | Value |
| ------ | ----: |
| MAE    | 0.125 |
| RMSE   | 0.236 |
| R²     | 0.955 |

### Leave-One-State-Out Validation

Leave-one-state-out validation evaluates spatial transferability by excluding one state at a time from training and testing on the excluded state.

The results show that spatial generalization is not uniform across states. Some states show strong transferability, while states such as Chhattisgarh and Jammu and Kashmir show weaker transferability.

---

## 2. Efficiency-Class Classification

The classification target is:

```text
efficiency_class
```

Class encoding:

| Class ID | Class label          |
| -------: | -------------------- |
|        0 | Efficient            |
|        1 | Inefficient          |
|        2 | Intensive            |
|        3 | Low-input low-output |

The classification models evaluated are:

* Logistic Regression
* Random Forest
* Extra Trees
* Gradient Boosting
* HistGradientBoosting
* XGBoost
* LightGBM
* CatBoost
* SVM
* KNN
* MLP Classifier

### Best Classification Model

The best classification model is:

```text
LightGBM
```

Performance:

| Metric          | Value |
| --------------- | ----: |
| Accuracy        | 0.925 |
| Macro precision | 0.924 |
| Macro recall    | 0.924 |
| Macro F1        | 0.924 |
| Weighted F1     | 0.925 |
| MCC             | 0.899 |
| Cohen’s kappa   | 0.899 |

---

## Statistical Analysis

The workflow includes the following statistical analyses:

* Descriptive statistics
* Distributional analysis
* Shapiro–Wilk normality testing
* Spearman correlation analysis
* Kruskal–Wallis group comparison
* Dunn post-hoc test
* Variance inflation factor analysis
* Crop-wise yield and input-efficiency summaries
* State-wise yield and input-efficiency summaries
* Season-wise yield and input-intensity summaries
* Year-wise trend analysis

---

## Explainable AI Components

AgroInput-XAI includes global, local, and sensitivity-based explainability.

### Regression explainability

* SHAP beeswarm plot
* SHAP feature importance
* Permutation importance
* Partial dependence plots
* LIME local explanations
* Residual analysis
* Crop-wise prediction error analysis
* State-wise prediction error analysis
* Counterfactual-style fertilizer sensitivity

### Classification explainability

* Permutation importance for best classifier
* Confusion matrices
* LIME local explanation for a representative classification sample

---

## Key Findings

The main findings are:

* ExtraTrees achieved the best performance for crop-yield regression.
* LightGBM achieved the best performance for input-efficiency classification.
* Crop identity was the dominant driver of yield prediction.
* Fertilizer intensity was the most important driver of efficiency-class prediction.
* Nonlinear ensemble models performed better than linear baselines.
* Temporal validation showed strong generalization to later years.
* Leave-one-state-out validation showed non-uniform spatial transferability.
* Counterfactual-style fertilizer sensitivity showed a mild saturation pattern, suggesting that predicted yield did not increase indefinitely with increasing fertilizer intensity.

---

## Repository Structure

A recommended repository structure is:

```text
AgroInput-XAI/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── data/
│   ├── crop_yield.csv
│   ├── state_weather_data_1997_2020.csv
│   └── state_soil_data.csv
│
├── notebooks/
│   └── AgroInput_XAI.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── modelling.py
│   ├── explainability.py
│   └── utils.py
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── models/
│
├── assets/
│   └── agroinput_xai_framework_flowchart.png
│
└── paper/
    ├── AgroInput_XAI.tex
    └── figures/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ParthaPRay/AgroInput-XAI.git
cd AgroInput-XAI
```

Create a virtual environment:

```bash
python -m venv agroinput_xai_env
```

Activate the environment:

For Linux/macOS:

```bash
source agroinput_xai_env/bin/activate
```

For Windows:

```bash
agroinput_xai_env\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Required Python Packages

The main packages used are:

```text
numpy
pandas
matplotlib
seaborn
scipy
statsmodels
scikit-learn
xgboost
lightgbm
catboost
shap
lime
scikit-posthocs
openpyxl
```

A sample `requirements.txt` may include:

```text
numpy
pandas
matplotlib
seaborn
scipy
statsmodels
scikit-learn
xgboost
lightgbm
catboost
shap
lime
scikit-posthocs
openpyxl
```

---

## Running the Notebook

Open the notebook:

```bash
jupyter notebook notebooks/AgroInput_XAI.ipynb
```

or use JupyterLab:

```bash
jupyter lab
```

Run the cells sequentially to generate:

* cleaned and merged dataset
* engineered features
* statistical tables
* regression model results
* classification model results
* SHAP explanations
* permutation-importance outputs
* LIME explanations
* residual plots
* validation tables
* final output tables and figures

---

## Output Directory

The notebook generates outputs in:

```text
AgroInput_XAI_outputs/
```

with subdirectories:

```text
AgroInput_XAI_outputs/
├── figures/
├── tables/
└── models/
```

Figures are saved as high-resolution PNG and PDF files. Tables are saved as CSV and Excel files.

---

## Main Outputs

The workflow produces:

* dataset audit tables
* descriptive statistics
* efficiency-class distribution
* crop-level yield and input-efficiency tables
* state-level yield and input-efficiency tables
* regression benchmark table
* temporal validation table
* leave-one-state-out validation table
* classification benchmark table
* SHAP plots
* permutation-importance plots
* partial dependence plots
* LIME explanation plots
* confusion matrices
* residual plots
* crop-wise and state-wise error plots
* counterfactual-style fertilizer sensitivity outputs

---

## Reproducibility

The analysis uses:

```python
RANDOM_STATE = 42
```

This ensures reproducible train-test splits, sampling, and model results wherever supported by the corresponding algorithm.

---

## Citation

If you use this repository, please cite the associated article or repository as:

```bibtex
@misc{ray2026agroinputxai,
  author       = {Ray, Partha Pratim},
  title        = {AgroInput-XAI: Explainable Modelling of Crop Yield, Fertilizer-Use Efficiency, and Pesticide-Use Efficiency under Soil--Weather Variability},
  year         = {2026},
  howpublished = {\url{https://github.com/ParthaPRay/AgroInput-XAI}}
}
```

---

## Author

**Partha Pratim Ray**
Department of Computer Applications
Sikkim University, India

Email:
[parthapratimray1986@gmail.com](mailto:parthapratimray1986@gmail.com)
[ppray@cus.ac.in](mailto:ppray@cus.ac.in)

GitHub:
[https://github.com/ParthaPRay](https://github.com/ParthaPRay)

---

## Disclaimer

AgroInput-XAI provides model-based predictive and explainability results. The outputs should be interpreted as data-driven patterns and not as direct causal estimates of fertilizer or pesticide response. Actual crop response may depend on field-level soil conditions, crop variety, irrigation, pest pressure, crop-stage weather, management practices, and local agronomic context.

---


```
```

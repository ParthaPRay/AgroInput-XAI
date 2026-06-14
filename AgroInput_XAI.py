#!/usr/bin/env python
# coding: utf-8

# Explainable Climate–Soil–Input Efficiency Modelling for Yield-Stable Fertilizer and Pesticide Use in Indian Cropping Systems
# 
# AgroInput-XAI: Explainable Modelling of Crop Yield, Fertilizer-Use Efficiency, and Pesticide-Use Efficiency under Soil--Weather Variability
# 
# 
# Interpretable Machine Learning for Climate–Soil–Input Efficiency Assessment in Indian Crop Production Systems
# 
# 
# 
# Partha Pratim Ray, Sikkim University, India, parthapratimray1986@gmail.com, ppray@cus.ac.in

# https://www.kaggle.com/datasets/anshumish/crop-yield-data-with-soil-and-weather-dataset/

# In[2]:


# Cell 1 — Install required packages
get_ipython().system('pip -q install shap lime optuna xgboost lightgbm catboost scikit-posthocs statsmodels openpyxl')


# In[3]:


# Cell 2 — Import libraries and global settings

import os
import re
import json
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from IPython.display import display

from scipy import stats
from scipy.stats import shapiro, kruskal, mannwhitneyu, spearmanr, pearsonr

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    classification_report,
    confusion_matrix
)

from sklearn.inspection import permutation_importance, PartialDependenceDisplay

from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

import shap
from lime.lime_tabular import LimeTabularExplainer

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

plt.rcParams["figure.dpi"] = 600
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

OUTPUT_DIR = "AgroInput_XAI_outputs"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
TAB_DIR = os.path.join(OUTPUT_DIR, "tables")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

for d in [OUTPUT_DIR, FIG_DIR, TAB_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

print("Libraries loaded successfully.")
print("Output folder:", OUTPUT_DIR)


# In[4]:


#Cell 3 — Helper functions for clean figures and tables

def clean_colnames(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.lower()
    )
    return df


def clean_text_columns(df):
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].str.replace(r"\s+", " ", regex=True)
    return df


def save_fig(name):
    path_png = os.path.join(FIG_DIR, f"{name}.png")
    path_pdf = os.path.join(FIG_DIR, f"{name}.pdf")
    plt.tight_layout()
    plt.savefig(path_png, dpi=600, bbox_inches="tight")
    plt.savefig(path_pdf, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path_png}")
    print(f"Saved: {path_pdf}")


def save_table(df, name):
    path_csv = os.path.join(TAB_DIR, f"{name}.csv")
    path_xlsx = os.path.join(TAB_DIR, f"{name}.xlsx")
    df.to_csv(path_csv, index=False)
    df.to_excel(path_xlsx, index=False)
    print(f"Saved table: {path_csv}")
    print(f"Saved table: {path_xlsx}")
    display(df)


def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100
    return {
        "MAE": mae,
        "RMSE": rmse,
        "Median_AE": medae,
        "R2": r2,
        "Explained_Variance": evs,
        "MAPE_percent": mape
    }


def classification_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro_Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro_Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted_F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Cohen_Kappa": cohen_kappa_score(y_true, y_pred)
    }


def top_n_table(df, col, n=15):
    out = df[col].value_counts().head(n).reset_index()
    out.columns = [col, "count"]
    out["percentage"] = out["count"] / len(df) * 100
    return out


print("Helper functions ready.")


# In[5]:


#Cell 4 — Load the three CSV files

crop_path = "crop_yield.csv"
weather_path = "state_weather_data_1997_2020.csv"
soil_path = "state_soil_data.csv"

crop = pd.read_csv(crop_path)
weather = pd.read_csv(weather_path)
soil = pd.read_csv(soil_path)

crop = clean_text_columns(clean_colnames(crop))
weather = clean_text_columns(clean_colnames(weather))
soil = clean_text_columns(clean_colnames(soil))

print("Crop yield shape:", crop.shape)
print("Weather shape:", weather.shape)
print("Soil shape:", soil.shape)

print("\nCrop columns:")
print(crop.columns.tolist())

print("\nWeather columns:")
print(weather.columns.tolist())

print("\nSoil columns:")
print(soil.columns.tolist())

display(crop.head())
display(weather.head())
display(soil.head())


# In[6]:


#Cell 6 — Merge crop, weather, and soil data

df = crop.merge(weather, on=["state", "year"], how="left")
df = df.merge(soil, on="state", how="left")

print("Merged dataset shape:", df.shape)

merge_audit = pd.DataFrame({
    "column": df.columns,
    "missing_count": df.isna().sum().values,
    "missing_percent": (df.isna().sum().values / len(df)) * 100
})

save_table(merge_audit, "merged_missing_value_audit")

display(df.head())


# In[7]:


# Cell 7 — Feature engineering

#Important: production should not be used as a predictor for yield, because yield is generally derived from production and area.
#Using production creates target leakage.

df = df.copy()

eps = 1e-6

df["fertilizer_per_ha"] = df["fertilizer"] / (df["area"] + eps)
df["pesticide_per_ha"] = df["pesticide"] / (df["area"] + eps)

df["log_area"] = np.log1p(df["area"])
df["log_fertilizer"] = np.log1p(df["fertilizer"])
df["log_pesticide"] = np.log1p(df["pesticide"])
df["log_fertilizer_per_ha"] = np.log1p(df["fertilizer_per_ha"])
df["log_pesticide_per_ha"] = np.log1p(df["pesticide_per_ha"])
df["log_yield"] = np.log1p(df["yield"])

df["npk_sum"] = df["n"] + df["p"] + df["k"]
df["n_p_ratio"] = df["n"] / (df["p"] + eps)
df["n_k_ratio"] = df["n"] / (df["k"] + eps)
df["p_k_ratio"] = df["p"] / (df["k"] + eps)

df["rainfall_temp_ratio"] = df["total_rainfall_mm"] / (df["avg_temp_c"] + eps)
df["humidity_temp_ratio"] = df["avg_humidity_percent"] / (df["avg_temp_c"] + eps)

df["fertilizer_use_efficiency"] = df["yield"] / (df["fertilizer_per_ha"] + eps)
df["pesticide_use_efficiency"] = df["yield"] / (df["pesticide_per_ha"] + eps)

df["log_fue"] = np.log1p(df["fertilizer_use_efficiency"])
df["log_pue"] = np.log1p(df["pesticide_use_efficiency"])

print("Feature-engineered dataset shape:", df.shape)
display(df.head())

engineered_cols = [
    "fertilizer_per_ha", "pesticide_per_ha",
    "log_yield", "npk_sum", "n_p_ratio", "n_k_ratio", "p_k_ratio",
    "rainfall_temp_ratio", "humidity_temp_ratio",
    "fertilizer_use_efficiency", "pesticide_use_efficiency"
]

save_table(df[engineered_cols].describe().T.reset_index().rename(columns={"index": "feature"}),
           "engineered_feature_descriptive_statistics")


# In[8]:


# Cell 8 — Create sustainable efficiency classes

# This creates four agronomic decision classes:

# High yield + low fertilizer = efficient
# High yield + high fertilizer = intensive
# Low yield + high fertilizer = inefficient
# Low yield + low fertilizer = low-input low-output


yield_median = df["yield"].median()
fert_median = df["fertilizer_per_ha"].median()

def efficiency_class(row):
    high_yield = row["yield"] >= yield_median
    high_fert = row["fertilizer_per_ha"] >= fert_median

    if high_yield and not high_fert:
        return "Efficient"
    elif high_yield and high_fert:
        return "Intensive"
    elif (not high_yield) and high_fert:
        return "Inefficient"
    else:
        return "Low-input low-output"

df["efficiency_class"] = df.apply(efficiency_class, axis=1)

class_table = df["efficiency_class"].value_counts().reset_index()
class_table.columns = ["efficiency_class", "count"]
class_table["percentage"] = class_table["count"] / len(df) * 100

save_table(class_table, "efficiency_class_distribution")

plt.figure(figsize=(8, 5))
plt.bar(class_table["efficiency_class"], class_table["count"])
plt.title("Distribution of sustainable input-efficiency classes")
plt.xlabel("Efficiency class")
plt.ylabel("Count")
plt.xticks(rotation=30, ha="right")
save_fig("efficiency_class_distribution")

print("Tabular values below the plot:")
display(class_table)


# In[9]:


# Cell 9 — Dataset structure: states, crops, seasons, years

structure_summary = pd.DataFrame({
    "item": ["states", "crops", "seasons", "years", "records"],
    "count": [
        df["state"].nunique(),
        df["crop"].nunique(),
        df["season"].nunique(),
        df["year"].nunique(),
        len(df)
    ]
})

save_table(structure_summary, "dataset_structure_summary")

for col in ["state", "crop", "season", "year"]:
    tab = top_n_table(df, col, n=20)
    save_table(tab, f"top_{col}_distribution")


# In[10]:


# Cell 11 — State distribution plot

state_counts = df["state"].value_counts().reset_index()
state_counts.columns = ["state", "count"]

plt.figure(figsize=(9, 8))
plt.barh(state_counts["state"][::-1], state_counts["count"][::-1])
plt.title("Records by state")
plt.xlabel("Count")
plt.ylabel("State")
save_fig("records_by_state")

print("Tabular values below the plot:")
display(state_counts)


# In[11]:


# Cell 12 — Season-wise yield summary

season_yield = (
    df.groupby("season")
    .agg(
        records=("yield", "size"),
        mean_yield=("yield", "mean"),
        median_yield=("yield", "median"),
        std_yield=("yield", "std"),
        mean_fertilizer_per_ha=("fertilizer_per_ha", "mean"),
        mean_pesticide_per_ha=("pesticide_per_ha", "mean")
    )
    .reset_index()
    .sort_values("mean_yield", ascending=False)
)

save_table(season_yield, "season_wise_yield_input_summary")

plt.figure(figsize=(8, 5))
plt.bar(season_yield["season"], season_yield["mean_yield"])
plt.title("Mean yield by season")
plt.xlabel("Season")
plt.ylabel("Mean yield")
plt.xticks(rotation=30, ha="right")
save_fig("mean_yield_by_season")

print("Tabular values below the plot:")
display(season_yield)


# In[12]:


# Cell 13 — Year-wise yield and input trends

year_trend = (
    df.groupby("year")
    .agg(
        records=("yield", "size"),
        mean_yield=("yield", "mean"),
        median_yield=("yield", "median"),
        mean_fertilizer_per_ha=("fertilizer_per_ha", "mean"),
        mean_pesticide_per_ha=("pesticide_per_ha", "mean"),
        mean_rainfall=("total_rainfall_mm", "mean"),
        mean_temperature=("avg_temp_c", "mean"),
        mean_humidity=("avg_humidity_percent", "mean")
    )
    .reset_index()
)

save_table(year_trend, "year_wise_trend_summary")

plt.figure(figsize=(9, 5))
plt.plot(year_trend["year"], year_trend["mean_yield"], marker="o")
plt.title("Mean crop yield trend over years")
plt.xlabel("Year")
plt.ylabel("Mean yield")
plt.grid(True, alpha=0.3)
save_fig("mean_yield_trend_by_year")

print("Tabular values below the plot:")
display(year_trend)


# In[13]:


# Cell 14 — Fertilizer and pesticide intensity trends

plt.figure(figsize=(9, 5))
plt.plot(year_trend["year"], year_trend["mean_fertilizer_per_ha"], marker="o", label="Fertilizer per ha")
plt.plot(year_trend["year"], year_trend["mean_pesticide_per_ha"], marker="s", label="Pesticide per ha")
plt.title("Mean input intensity trends over years")
plt.xlabel("Year")
plt.ylabel("Mean input per ha")
plt.legend()
plt.grid(True, alpha=0.3)
save_fig("input_intensity_trends_by_year")

print("Tabular values below the plot:")
display(year_trend[["year", "mean_fertilizer_per_ha", "mean_pesticide_per_ha"]])


# In[14]:


# Cell 15 — Descriptive statistics of numerical variables

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

desc = df[numeric_cols].describe().T.reset_index().rename(columns={"index": "feature"})
desc["skewness"] = df[numeric_cols].skew().values
desc["kurtosis"] = df[numeric_cols].kurtosis().values

save_table(desc, "full_numerical_descriptive_statistics")


# In[15]:


# Cell 16 — Distribution plots for key variables

import seaborn as sns

key_vars = [
    "yield", "log_yield",
    "fertilizer_per_ha", "log_fertilizer_per_ha",
    "pesticide_per_ha", "log_pesticide_per_ha",
    "avg_temp_c", "total_rainfall_mm", "avg_humidity_percent",
    "n", "p", "k", "ph"
]

distribution_summary = []

for col in key_vars:
    temp = df[col].dropna()
    distribution_summary.append({
        "feature": col,
        "count": temp.shape[0],
        "mean": temp.mean(),
        "median": temp.median(),
        "std": temp.std(),
        "min": temp.min(),
        "max": temp.max(),
        "skewness": temp.skew(),
        "kurtosis": temp.kurtosis()
    })

    plt.figure(figsize=(7, 5))
    # Using sns.histplot with kde=True overlays the smooth curve automatically
    sns.histplot(temp, bins=40, edgecolor="black", alpha=0.8, kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    save_fig(f"distribution_{col}")
    plt.show()

distribution_summary = pd.DataFrame(distribution_summary)
save_table(distribution_summary, "distribution_summary_key_variables")


# In[16]:


# Cell 17 — Correlation analysis

corr_features = [
    "yield", "log_yield",
    "fertilizer_per_ha", "pesticide_per_ha",
    "avg_temp_c", "total_rainfall_mm", "avg_humidity_percent",
    "n", "p", "k", "ph",
    "npk_sum", "n_p_ratio", "n_k_ratio", "p_k_ratio"
]

corr_df = df[corr_features].corr(method="spearman")

corr_long = (
    corr_df.reset_index()
    .melt(id_vars="index", var_name="variable_2", value_name="spearman_correlation")
    .rename(columns={"index": "variable_1"})
)

save_table(corr_long, "spearman_correlation_long_table")

plt.figure(figsize=(10, 8))
plt.imshow(corr_df, aspect="auto")
plt.colorbar(label="Spearman correlation")
plt.xticks(range(len(corr_features)), corr_features, rotation=90)
plt.yticks(range(len(corr_features)), corr_features)
plt.title("Spearman correlation heatmap")
save_fig("spearman_correlation_heatmap")

print("Tabular values below the plot:")
display(corr_df)


# In[17]:


# Cell 18 — Top crops by mean yield and input efficiency

crop_summary = (
    df.groupby("crop")
    .agg(
        records=("yield", "size"),
        mean_yield=("yield", "mean"),
        median_yield=("yield", "median"),
        mean_fertilizer_per_ha=("fertilizer_per_ha", "mean"),
        mean_pesticide_per_ha=("pesticide_per_ha", "mean"),
        mean_fue=("fertilizer_use_efficiency", "mean"),
        mean_pue=("pesticide_use_efficiency", "mean")
    )
    .reset_index()
)

crop_summary_filtered = crop_summary[crop_summary["records"] >= 20].copy()

top_yield_crops = crop_summary_filtered.sort_values("mean_yield", ascending=False).head(20)
top_fue_crops = crop_summary_filtered.sort_values("mean_fue", ascending=False).head(20)

save_table(crop_summary_filtered.sort_values("mean_yield", ascending=False),
           "crop_level_yield_input_efficiency_summary")

plt.figure(figsize=(9, 7))
plt.barh(top_yield_crops["crop"][::-1], top_yield_crops["mean_yield"][::-1])
plt.title("Top 20 crops by mean yield")
plt.xlabel("Mean yield")
plt.ylabel("Crop")
save_fig("top_20_crops_by_mean_yield")

print("Tabular values below the plot:")
display(top_yield_crops)


# In[18]:


# Cell 19 — State-wise input efficiency

state_summary = (
    df.groupby("state")
    .agg(
        records=("yield", "size"),
        mean_yield=("yield", "mean"),
        median_yield=("yield", "median"),
        mean_fertilizer_per_ha=("fertilizer_per_ha", "mean"),
        mean_pesticide_per_ha=("pesticide_per_ha", "mean"),
        mean_fue=("fertilizer_use_efficiency", "mean"),
        mean_pue=("pesticide_use_efficiency", "mean"),
        mean_rainfall=("total_rainfall_mm", "mean"),
        mean_temp=("avg_temp_c", "mean"),
        soil_n=("n", "mean"),
        soil_p=("p", "mean"),
        soil_k=("k", "mean"),
        soil_ph=("ph", "mean")
    )
    .reset_index()
)

save_table(state_summary.sort_values("mean_fue", ascending=False),
           "state_level_yield_input_efficiency_summary")

top_states_fue = state_summary.sort_values("mean_fue", ascending=False).head(15)

plt.figure(figsize=(9, 6))
plt.barh(top_states_fue["state"][::-1], top_states_fue["mean_fue"][::-1])
plt.title("Top 15 states by mean fertilizer-use efficiency")
plt.xlabel("Mean fertilizer-use efficiency")
plt.ylabel("State")
save_fig("top_15_states_by_fue")

print("Tabular values below the plot:")
display(top_states_fue)


# In[19]:


# Cell 20 — Statistical normality tests

normality_features = [
    "yield", "log_yield",
    "fertilizer_per_ha", "log_fertilizer_per_ha",
    "pesticide_per_ha", "log_pesticide_per_ha",
    "avg_temp_c", "total_rainfall_mm",
    "avg_humidity_percent", "n", "p", "k", "ph"
]

normality_results = []

for col in normality_features:
    # Drops missing values and includes all available rows
    x = df[col].dropna()
    stat, pval = shapiro(x)
    normality_results.append({
        "feature": col,
        "shapiro_statistic": stat,
        "p_value": pval,
        "normal_at_0_05": pval > 0.05
    })

normality_results = pd.DataFrame(normality_results)
save_table(normality_results, "shapiro_wilk_normality_tests")


# In[20]:


# Cell 21 — Kruskal-Wallis tests by season and efficiency class

def kruskal_by_group(df, value_col, group_col):
    groups = []
    labels = []
    for name, sub in df.groupby(group_col):
        vals = sub[value_col].dropna()
        if len(vals) > 5:
            groups.append(vals)
            labels.append(name)
    if len(groups) >= 2:
        stat, pval = kruskal(*groups)
    else:
        stat, pval = np.nan, np.nan

    return {
        "value": value_col,
        "group": group_col,
        "groups_tested": len(groups),
        "kruskal_statistic": stat,
        "p_value": pval,
        "significant_at_0_05": pval < 0.05 if not np.isnan(pval) else None,
        "significant_at_0_01": pval < 0.01 if not np.isnan(pval) else None
    }

kruskal_tests = []
for value_col in ["yield", "log_yield", "fertilizer_per_ha", "pesticide_per_ha", "fertilizer_use_efficiency"]:
    for group_col in ["season", "efficiency_class"]:
        kruskal_tests.append(kruskal_by_group(df, value_col, group_col))

kruskal_results = pd.DataFrame(kruskal_tests)
save_table(kruskal_results, "kruskal_wallis_tests")


# In[21]:


# Cell 22 — Pairwise post-hoc tests for season-wise yield

import scikit_posthocs as sp

posthoc_season_yield = sp.posthoc_dunn(
    df,
    val_col="log_yield",
    group_col="season",
    p_adjust="bonferroni"
)

posthoc_season_yield = posthoc_season_yield.reset_index().rename(columns={"index": "season"})
save_table(posthoc_season_yield, "posthoc_dunn_log_yield_by_season")


# In[22]:


# Cell 23 — Spearman association tests with yield

association_features = [
    "fertilizer_per_ha", "pesticide_per_ha",
    "avg_temp_c", "total_rainfall_mm", "avg_humidity_percent",
    "n", "p", "k", "ph",
    "npk_sum", "n_p_ratio", "n_k_ratio", "p_k_ratio",
    "rainfall_temp_ratio", "humidity_temp_ratio"
]

assoc_results = []

for col in association_features:
    x = df[col]
    y = df["yield"]
    mask = x.notna() & y.notna()
    rho, pval = spearmanr(x[mask], y[mask])
    assoc_results.append({
        "feature": col,
        "target": "yield",
        "spearman_rho": rho,
        "p_value": pval,
        "significant_at_0_05": pval < 0.05
    })

assoc_results = pd.DataFrame(assoc_results).sort_values("spearman_rho", ascending=False)
save_table(assoc_results, "spearman_association_with_yield")

plt.figure(figsize=(8, 6))
plt.barh(assoc_results["feature"][::-1], assoc_results["spearman_rho"][::-1])
plt.axvline(0, linestyle="--")
plt.title("Spearman association of features with yield")
plt.xlabel("Spearman rho")
plt.ylabel("Feature")
save_fig("spearman_association_with_yield")

print("Tabular values below the plot:")
display(assoc_results)


# In[23]:


# Cell 24 — Multicollinearity analysis using VIF

vif_features = [
    "fertilizer_per_ha", "pesticide_per_ha",
    "avg_temp_c", "total_rainfall_mm", "avg_humidity_percent",
    "n", "p", "k", "ph",
    "npk_sum"
]

vif_data = df[vif_features].replace([np.inf, -np.inf], np.nan).dropna()
vif_data_scaled = pd.DataFrame(
    StandardScaler().fit_transform(vif_data),
    columns=vif_features
)

vif_results = []

for i, col in enumerate(vif_data_scaled.columns):
    vif_results.append({
        "feature": col,
        "VIF": variance_inflation_factor(vif_data_scaled.values, i)
    })

vif_results = pd.DataFrame(vif_results).sort_values("VIF", ascending=False)
save_table(vif_results, "variance_inflation_factor_results")


# In[24]:


# Cell 25 — Regression modelling dataset

#Target:

#        log_yield

# Important: production is excluded to avoid leakage.

target_reg = "log_yield"

categorical_features = ["crop", "season", "state"]

numeric_features = [
    "year",
    "log_area",
    "log_fertilizer_per_ha",
    "log_pesticide_per_ha",
    "avg_temp_c",
    "total_rainfall_mm",
    "avg_humidity_percent",
    "n", "p", "k", "ph",
    "npk_sum",
    "n_p_ratio", "n_k_ratio", "p_k_ratio",
    "rainfall_temp_ratio", "humidity_temp_ratio"
]

model_df = df[categorical_features + numeric_features + [target_reg, "yield"]].replace([np.inf, -np.inf], np.nan).dropna()

X = model_df[categorical_features + numeric_features]
y = model_df[target_reg]

print("Regression modelling dataset shape:", X.shape)
display(X.head())


# In[25]:


# Cell 26 — Preprocessing pipeline

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)

print("Preprocessor ready.")


# In[26]:


# Cell 27 — Random train-test split for yield prediction

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# In[27]:


# Cell 28 — Regression model benchmark

regressors = {
    "Ridge": Ridge(random_state=RANDOM_STATE),
    "ElasticNet": ElasticNet(random_state=RANDOM_STATE),
    "RandomForest": RandomForestRegressor(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    "XGBoost": XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "CatBoost": CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_seed=RANDOM_STATE,
        verbose=0
    ),
    "MLP": MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        random_state=RANDOM_STATE
    )
}

reg_results = []
reg_predictions = {}

for name, model in regressors.items():
    print(f"Training {name}...")
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    pred_log = pipe.predict(X_test)

    metrics = regression_metrics(y_test, pred_log)
    metrics["Model"] = name
    reg_results.append(metrics)
    reg_predictions[name] = {
        "pipeline": pipe,
        "prediction_log": pred_log
    }

reg_results_df = pd.DataFrame(reg_results).sort_values("RMSE")
save_table(reg_results_df, "regression_model_benchmark_random_split")

plt.figure(figsize=(9, 6))
plt.barh(reg_results_df["Model"][::-1], reg_results_df["R2"][::-1])
plt.title("Regression model comparison for log-yield prediction")
plt.xlabel("R²")
plt.ylabel("Model")
save_fig("regression_model_comparison_r2")

print("Tabular values below the plot:")
display(reg_results_df)


# In[28]:


# Cell 29 — Individual actual vs predicted plots for all regression models

actual_yield = np.expm1(y_test)
all_models = reg_results_df["Model"].tolist()

for model_name in all_models:
    # 1. Extract pipeline and predictions for the current model
    pipe = reg_predictions[model_name]["pipeline"]
    pred_log = reg_predictions[model_name]["prediction_log"]
    pred_yield = np.expm1(pred_log)

    # 2. Create the evaluation dataframe
    actual_pred_df = pd.DataFrame({
        "actual_yield": actual_yield,
        "predicted_yield": pred_yield,
        "actual_log_yield": y_test,
        "predicted_log_yield": pred_log,
        "absolute_error_yield": np.abs(actual_yield - pred_yield)
    })

    # 3. Save files to disk
    summary_df = actual_pred_df.describe().T.reset_index().rename(columns={"index": "metric"})
    save_table(summary_df, f"{model_name.lower()}_regression_actual_predicted_summary")
    save_table(actual_pred_df.head(20), f"{model_name.lower()}_regression_actual_predicted_head20")

    # 4. Generate and display the individual plot
    plt.figure(figsize=(6, 6))
    plt.scatter(actual_pred_df["actual_yield"], actual_pred_df["predicted_yield"], alpha=0.35, color="teal")

    # Identity line (perfect prediction reference line)
    min_val = min(actual_pred_df["actual_yield"].min(), actual_pred_df["predicted_yield"].min())
    max_val = max(actual_pred_df["actual_yield"].max(), actual_pred_df["predicted_yield"].max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="crimson", linewidth=2)

    plt.title(f"Actual vs Predicted Yield: {model_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Actual Yield")
    plt.ylabel("Predicted Yield")
    plt.grid(True, linestyle=":", alpha=0.6)

    # Save individual figure
    save_fig(f"actual_vs_predicted_yield_{model_name.lower()}")
    plt.show()

    # 5. Print the corresponding tabular statistics below the current plot
    print("=" * 60)
    print(f"TABULAR VALUES FOR MODEL: {model_name}")
    print("=" * 60)

    print("\n[1/2] Descriptive Statistics Summary:")
    display(actual_pred_df.describe().T)

    print("\n[2/2] First 20 Predicted vs Actual Rows:")
    display(actual_pred_df.head(20))
    print("\n" + "\n" + "_"*80 + "\n")


# In[29]:


# Cell 30 — Residual analysis for all regression models

all_models = reg_results_df["Model"].tolist()
actual_yield = np.expm1(y_test)

for model_name in all_models:
    # 1. Extract predictions for the current model
    pred_log = reg_predictions[model_name]["prediction_log"]

    # 2. Create a fresh dataframe for the current model's residual analysis
    model_df = pd.DataFrame({
        "actual_log_yield": y_test,
        "predicted_log_yield": pred_log
    })

    # Calculate the residual in the log-transformed scale
    model_df["residual_log"] = model_df["actual_log_yield"] - model_df["predicted_log_yield"]

    # 3. Create and save the summary table
    residual_summary = model_df["residual_log"].describe().reset_index()
    residual_summary.columns = ["statistic", "value"]
    save_table(residual_summary, f"residual_summary_{model_name.lower()}")

    # 4. Generate and display the individual residual histogram
    plt.figure(figsize=(7, 5))
    plt.hist(model_df["residual_log"], bins=40, color="steelblue", edgecolor="black", alpha=0.8)

    # Reference line at 0 (where actual equals predicted)
    plt.axvline(x=0, color="crimson", linestyle="--", linewidth=1.5, label="Zero Error")

    plt.title(f"Residual Distribution: {model_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Residual in log-yield")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)

    # Save the individual figure
    save_fig(f"residual_distribution_{model_name.lower()}")
    plt.show()

    # 5. Print the corresponding tabular values below the current plot
    print("=" * 60)
    print(f"RESIDUAL SUMMARY FOR MODEL: {model_name}")
    print("=" * 60)
    display(residual_summary)
    print("\n" + "_"*80 + "\n")


# In[30]:


# Cell 31 — Year-wise temporal validation, corrected

# Check whether year exists in the main merged dataset
print("Available columns in df:")
print(df.columns.tolist())

if "year" not in df.columns:
    raise KeyError("Column 'year' is not found in df. Please check whether it is named differently, such as 'crop_year' or 'Year'.")

# Rebuild modelling dataframe safely with year included
target_reg = "log_yield"

categorical_features = ["crop", "season", "state"]

numeric_features = [
    "year",
    "log_area",
    "log_fertilizer_per_ha",
    "log_pesticide_per_ha",
    "avg_temp_c",
    "total_rainfall_mm",
    "avg_humidity_percent",
    "n", "p", "k", "ph",
    "npk_sum",
    "n_p_ratio", "n_k_ratio", "p_k_ratio",
    "rainfall_temp_ratio", "humidity_temp_ratio"
]

required_cols = categorical_features + numeric_features + [target_reg]

missing_cols = [col for col in required_cols if col not in df.columns]

if len(missing_cols) > 0:
    raise KeyError(f"These required columns are missing from df: {missing_cols}")

model_df_time = (
    df[required_cols]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .copy()
)

# Make sure year is numeric
model_df_time["year"] = pd.to_numeric(model_df_time["year"], errors="coerce")
model_df_time = model_df_time.dropna(subset=["year"])
model_df_time["year"] = model_df_time["year"].astype(int)

print("Temporal modelling dataset shape:", model_df_time.shape)
print("Year range:", model_df_time["year"].min(), "to", model_df_time["year"].max())

# Use 2016 as cutoff if available
train_year_max = 2016

train_mask = model_df_time["year"] <= train_year_max
test_mask = model_df_time["year"] > train_year_max

X_train_time = model_df_time.loc[train_mask, categorical_features + numeric_features]
y_train_time = model_df_time.loc[train_mask, target_reg]

X_test_time = model_df_time.loc[test_mask, categorical_features + numeric_features]
y_test_time = model_df_time.loc[test_mask, target_reg]

print("Temporal train years:",
      X_train_time["year"].min(), "-", X_train_time["year"].max())

print("Temporal test years:",
      X_test_time["year"].min(), "-", X_test_time["year"].max())

print("Temporal train shape:", X_train_time.shape)
print("Temporal test shape:", X_test_time.shape)

if len(X_train_time) == 0 or len(X_test_time) == 0:
    raise ValueError("Temporal split failed. Check whether years before and after 2016 exist in the dataset.")

temporal_results = []

for name, model in regressors.items():
    print(f"Temporal validation: {name}")

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train_time, y_train_time)
    pred = pipe.predict(X_test_time)

    metrics = regression_metrics(y_test_time, pred)
    metrics["Model"] = name
    metrics["Validation"] = "Train <= 2016, Test > 2016"
    temporal_results.append(metrics)

temporal_results_df = pd.DataFrame(temporal_results).sort_values("RMSE")

save_table(temporal_results_df, "temporal_validation_regression_results")

plt.figure(figsize=(9, 6))
plt.barh(temporal_results_df["Model"][::-1], temporal_results_df["R2"][::-1])
plt.title("Temporal validation: model comparison")
plt.xlabel("R²")
plt.ylabel("Model")
save_fig("temporal_validation_regression_r2")

print("Tabular values below the plot:")
display(temporal_results_df)


# In[31]:


# Cell 32 — Leave-one-state-out validation, corrected

# Make sure required columns exist
target_reg = "log_yield"

categorical_features = ["crop", "season", "state"]

numeric_features = [
    "year",
    "log_area",
    "log_fertilizer_per_ha",
    "log_pesticide_per_ha",
    "avg_temp_c",
    "total_rainfall_mm",
    "avg_humidity_percent",
    "n", "p", "k", "ph",
    "npk_sum",
    "n_p_ratio", "n_k_ratio", "p_k_ratio",
    "rainfall_temp_ratio", "humidity_temp_ratio"
]

required_cols = categorical_features + numeric_features + [target_reg]

missing_cols = [col for col in required_cols if col not in df.columns]

if len(missing_cols) > 0:
    raise KeyError(f"These required columns are missing from df: {missing_cols}")

# Rebuild safe modelling dataframe
model_df_state = (
    df[required_cols]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .copy()
)

model_df_state["year"] = pd.to_numeric(model_df_state["year"], errors="coerce")
model_df_state = model_df_state.dropna(subset=["year"])
model_df_state["year"] = model_df_state["year"].astype(int)

print("Leave-one-state-out modelling dataset shape:", model_df_state.shape)
print("Number of states:", model_df_state["state"].nunique())
print("States:")
print(sorted(model_df_state["state"].unique()))

# Base model for leave-one-state-out validation
base_model_for_group_test = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

state_group_results = []

unique_states = sorted(model_df_state["state"].unique())

for state_name in unique_states:
    train_mask = model_df_state["state"] != state_name
    test_mask = model_df_state["state"] == state_name

    X_tr = model_df_state.loc[train_mask, categorical_features + numeric_features]
    y_tr = model_df_state.loc[train_mask, target_reg]

    X_te = model_df_state.loc[test_mask, categorical_features + numeric_features]
    y_te = model_df_state.loc[test_mask, target_reg]

    # Skip states with too few samples
    if len(X_te) < 20:
        print(f"Skipping {state_name}: only {len(X_te)} records")
        continue

    print(f"Training without state: {state_name} | Test records: {len(X_te)}")

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", base_model_for_group_test)
    ])

    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)

    metrics = regression_metrics(y_te, pred)
    metrics["left_out_state"] = state_name
    metrics["test_records"] = len(X_te)
    state_group_results.append(metrics)

state_group_results_df = pd.DataFrame(state_group_results)

if state_group_results_df.empty:
    raise ValueError("No state had enough records for leave-one-state-out validation.")

state_group_results_df = state_group_results_df.sort_values("RMSE")

save_table(state_group_results_df, "leave_one_state_out_validation_results")

plt.figure(figsize=(9, 8))
plt.barh(
    state_group_results_df["left_out_state"][::-1],
    state_group_results_df["R2"][::-1]
)
plt.title("Leave-one-state-out validation using XGBoost")
plt.xlabel("R²")
plt.ylabel("Left-out state")
save_fig("leave_one_state_out_validation_r2")

print("Tabular values below the plot:")
display(state_group_results_df)


# In[32]:


# Cell 33 — Efficiency-class classification dataset


target_cls = "efficiency_class"

cls_features = categorical_features + numeric_features

cls_df = df[cls_features + [target_cls]].replace([np.inf, -np.inf], np.nan).dropna()

X_cls = cls_df[cls_features]
y_cls = cls_df[target_cls]

label_encoder = LabelEncoder()
y_cls_encoded = label_encoder.fit_transform(y_cls)

print("Classes:")
for i, c in enumerate(label_encoder.classes_):
    print(i, "=", c)

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls,
    y_cls_encoded,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_cls_encoded
)

print("Classification train shape:", X_train_cls.shape)
print("Classification test shape:", X_test_cls.shape)


# In[33]:


# Cell 34 — Classification model benchmark

classifiers = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
    ),
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "CatBoost": CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_seed=RANDOM_STATE,
        verbose=0
    ),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE, class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        random_state=RANDOM_STATE
    )
}

cls_results = []
cls_predictions = {}

for name, model in classifiers.items():
    print(f"Training classifier: {name}")
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train_cls, y_train_cls)
    pred = pipe.predict(X_test_cls)

    metrics = classification_metrics(y_test_cls, pred)
    metrics["Model"] = name
    cls_results.append(metrics)

    cls_predictions[name] = {
        "pipeline": pipe,
        "prediction": pred
    }

cls_results_df = pd.DataFrame(cls_results).sort_values("Macro_F1", ascending=False)
save_table(cls_results_df, "classification_model_benchmark_efficiency_class")

plt.figure(figsize=(9, 6))
plt.barh(cls_results_df["Model"][::-1], cls_results_df["Macro_F1"][::-1])
plt.title("Classification model comparison for input-efficiency class")
plt.xlabel("Macro F1-score")
plt.ylabel("Model")
save_fig("classification_model_comparison_macro_f1")

print("Tabular values below the plot:")
display(cls_results_df)


# In[34]:


# Cell 35 — Confusion matrix and classification report for all models

all_classifiers = cls_results_df["Model"].tolist()

for model_name in all_classifiers:
    # 1. Extract predictions for the current model
    pred = cls_predictions[model_name]["prediction"]

    # 2. Compute confusion matrix and format as a DataFrame
    cm = confusion_matrix(y_test_cls, pred)
    cm_df = pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    )

    # 3. Compute detailed classification report
    report_dict = classification_report(
        y_test_cls,
        pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).T.reset_index().rename(columns={"index": "class"})

    # 4. Save results to disk
    save_table(cm_df.reset_index().rename(columns={"index": "actual_class"}), f"confusion_matrix_{model_name.lower()}")
    save_table(report_df, f"classification_report_{model_name.lower()}")

    # 5. Generate and display the individual confusion matrix plot
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, aspect="auto", cmap="Blues")  # Switched to a cleaner colormap for confusion matrices
    plt.colorbar(label="Count")

    plt.xticks(range(len(label_encoder.classes_)), label_encoder.classes_, rotation=45, ha="right")
    plt.yticks(range(len(label_encoder.classes_)), label_encoder.classes_)

    plt.title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")

    # Text annotations over cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Dynamic text color adjustment based on cell intensity
            text_color = "white" if cm[i, j] > (cm.max() / 2) else "black"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=text_color, fontweight="bold")

    plt.tight_layout()
    save_fig(f"confusion_matrix_{model_name.lower()}")
    plt.show()

    # 6. Display tabular evaluations directly under the plot
    print("=" * 70)
    print(f"EVALUATION TABLES FOR MODEL: {model_name}")
    print("=" * 70)

    print("\n[1/2] Confusion Matrix DataFrame:")
    display(cm_df)

    print("\n[2/2] Classification Report Details:")
    display(report_df)
    print("\n" + "_"*90 + "\n")


# In[35]:


# Cell 36 — SHAP explainability for best regression model, corrected

# Ensure best model information exists
if "best_reg_model_name" not in globals():
    best_reg_model_name = reg_results_df.sort_values("RMSE").iloc[0]["Model"]

if "best_reg_pipe" not in globals():
    best_reg_pipe = reg_predictions[best_reg_model_name]["pipeline"]

print("Explaining best regression model:", best_reg_model_name)

# Ensure train-test split exists
if "X_train" not in globals() or "X_test" not in globals():
    raise NameError("X_train or X_test is missing. Please run Cell 27 first.")

# Transform data using trained preprocessing pipeline
X_train_trans = best_reg_pipe.named_steps["preprocess"].transform(X_train)
X_test_trans = best_reg_pipe.named_steps["preprocess"].transform(X_test)

# Recover feature names
feature_names_num = numeric_features

feature_names_cat = (
    best_reg_pipe
    .named_steps["preprocess"]
    .named_transformers_["cat"]
    .named_steps["onehot"]
    .get_feature_names_out(categorical_features)
    .tolist()
)

feature_names_all = feature_names_num + feature_names_cat

X_train_trans_df = pd.DataFrame(X_train_trans, columns=feature_names_all)
X_test_trans_df = pd.DataFrame(X_test_trans, columns=feature_names_all)

model_only = best_reg_pipe.named_steps["model"]

# Use small sample for CPU-friendly SHAP
sample_size = min(1000, X_test_trans_df.shape[0])
X_shap_sample = X_test_trans_df.sample(sample_size, random_state=RANDOM_STATE)

background_size = min(500, X_train_trans_df.shape[0])
X_shap_background = X_train_trans_df.sample(background_size, random_state=RANDOM_STATE)

print("SHAP sample shape:", X_shap_sample.shape)
print("SHAP background shape:", X_shap_background.shape)

# SHAP explainer selection
try:
    # Works well for tree-based models
    explainer = shap.Explainer(model_only, X_shap_background)
    shap_values = explainer(X_shap_sample)
    print("SHAP values computed using shap.Explainer.")
except Exception as e1:
    print("shap.Explainer failed. Trying TreeExplainer...")
    print("Reason:", e1)

    try:
        explainer = shap.TreeExplainer(model_only)
        shap_values_raw = explainer.shap_values(X_shap_sample)

        class SimpleShapObject:
            def __init__(self, values, data, feature_names):
                self.values = values
                self.data = data
                self.feature_names = feature_names

        shap_values = SimpleShapObject(
            values=shap_values_raw,
            data=X_shap_sample.values,
            feature_names=X_shap_sample.columns.tolist()
        )

        print("SHAP values computed using TreeExplainer.")
    except Exception as e2:
        print("TreeExplainer also failed.")
        print("Reason:", e2)
        raise


# In[36]:


# Cell 37 — SHAP global bar plot, corrected

if hasattr(shap_values, "values"):
    shap_array = shap_values.values
else:
    shap_array = shap_values

# If SHAP returns 3D array, average across output dimension
if len(np.array(shap_array).shape) == 3:
    shap_array_2d = np.abs(shap_array).mean(axis=2)
else:
    shap_array_2d = shap_array

plt.figure()
shap.summary_plot(
    shap_array_2d,
    X_shap_sample,
    plot_type="bar",
    max_display=25,
    show=False
)
plt.title(f"SHAP global importance: {best_reg_model_name}")
save_fig("shap_global_bar_best_regression_model")

shap_importance = pd.DataFrame({
    "feature": X_shap_sample.columns,
    "mean_abs_shap": np.abs(shap_array_2d).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

save_table(shap_importance, "shap_global_importance_best_regression_model")

print("Tabular values below the plot:")
display(shap_importance.head(30))


# In[37]:


# Cell 38 — SHAP beeswarm plot, corrected

plt.figure()
shap.summary_plot(
    shap_array_2d,
    X_shap_sample,
    max_display=25,
    show=False
)
plt.title(f"SHAP beeswarm plot: {best_reg_model_name}")
save_fig("shap_beeswarm_best_regression_model")

print("Tabular values below the plot:")
display(shap_importance.head(30))


# In[38]:


# Cell 39 — Permutation importance for all regression models

all_models = reg_results_df["Model"].tolist()

for model_name in all_models:
    print(f"Calculating permutation importance for {model_name}...")

    # 1. Extract the specific trained pipeline for the current model
    pipe = reg_predictions[model_name]["pipeline"]

    # 2. Compute permutation importance
    perm = permutation_importance(
        pipe,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    # 3. Format results into a sorted DataFrame
    perm_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std
    }).sort_values("importance_mean", ascending=False)

    # 4. Save the full table to disk
    save_table(perm_df, f"permutation_importance_{model_name.lower()}")

    # 5. Generate and display the individual horizontal bar plot (Top 20 features)
    plt.figure(figsize=(8, 6))
    top_perm = perm_df.head(20)

    # Invert the order [::-1] so the highest importance appears at the top of the horizontal bar chart
    plt.barh(top_perm["feature"][::-1], top_perm["importance_mean"][::-1], color="teal", edgecolor="black", alpha=0.8)

    plt.title(f"Permutation Importance: {model_name} (Top 20 Features)", fontsize=12, fontweight="bold")
    plt.xlabel("Mean decrease in performance (Negative RMSE)")
    plt.ylabel("Feature")
    plt.grid(True, linestyle=":", alpha=0.6)

    # Save the individual figure
    save_fig(f"permutation_importance_{model_name.lower()}")
    plt.show()

    # 6. Print the corresponding tabular values below the current plot
    print("=" * 70)
    print(f"PERMUTATION IMPORTANCE VALUES FOR MODEL: {model_name}")
    print("=" * 70)
    display(top_perm)
    print("\n" + "_"*80 + "\n")


# In[39]:


# Cell 40 — Partial dependence plots for top numerical features

top_numeric_for_pdp = [
    f for f in perm_df["feature"].tolist()
    if f in numeric_features
][:6]

pdp_tables = []

for feature in top_numeric_for_pdp:
    print("PDP for:", feature)

    fig, ax = plt.subplots(figsize=(7, 5))
    PartialDependenceDisplay.from_estimator(
        best_reg_pipe,
        X_test,
        features=[feature],
        ax=ax,
        grid_resolution=30
    )
    plt.title(f"Partial dependence of log-yield on {feature}")
    save_fig(f"pdp_{feature}_best_regression_model")

    # Manual tabular PDP approximation
    grid = np.linspace(X_test[feature].quantile(0.05), X_test[feature].quantile(0.95), 30)
    pdp_vals = []
    X_temp = X_test.copy()

    for val in grid:
        X_temp[feature] = val
        pred = best_reg_pipe.predict(X_temp)
        pdp_vals.append(np.mean(pred))

    pdp_df = pd.DataFrame({
        "feature": feature,
        "grid_value": grid,
        "mean_predicted_log_yield": pdp_vals,
        "mean_predicted_yield": np.expm1(pdp_vals)
    })

    pdp_tables.append(pdp_df)
    print("Tabular values below the plot:")
    display(pdp_df)

all_pdp_df = pd.concat(pdp_tables, ignore_index=True)
save_table(all_pdp_df, "partial_dependence_manual_tables")


# In[40]:


# Cell 41 — ALE-style local effect analysis

# This is an ALE-style binned effect summary. It is simpler than a full ALE package but useful for publication discussion.

def ale_style_table(model, X_data, feature, bins=10):
    X_temp = X_data.copy()
    qs = np.quantile(X_temp[feature].dropna(), np.linspace(0, 1, bins + 1))
    qs = np.unique(qs)

    rows = []
    baseline_pred = model.predict(X_temp).mean()

    for i in range(len(qs) - 1):
        low, high = qs[i], qs[i + 1]
        mid = (low + high) / 2
        X_mod = X_temp.copy()
        X_mod[feature] = mid
        pred_mean = model.predict(X_mod).mean()
        rows.append({
            "feature": feature,
            "bin_low": low,
            "bin_high": high,
            "bin_mid": mid,
            "mean_predicted_log_yield": pred_mean,
            "effect_vs_baseline": pred_mean - baseline_pred,
            "mean_predicted_yield": np.expm1(pred_mean)
        })

    return pd.DataFrame(rows)

ale_tables = []

for feature in top_numeric_for_pdp:
    ale_df = ale_style_table(best_reg_pipe, X_test, feature, bins=10)
    ale_tables.append(ale_df)

    plt.figure(figsize=(7, 5))
    plt.plot(ale_df["bin_mid"], ale_df["effect_vs_baseline"], marker="o")
    plt.axhline(0, linestyle="--")
    plt.title(f"ALE-style effect of {feature} on log-yield")
    plt.xlabel(feature)
    plt.ylabel("Effect vs baseline prediction")
    plt.grid(True, alpha=0.3)
    save_fig(f"ale_style_{feature}_best_regression_model")

    print("Tabular values below the plot:")
    display(ale_df)

all_ale_df = pd.concat(ale_tables, ignore_index=True)
save_table(all_ale_df, "ale_style_effect_tables")


# In[41]:


# Cell 42 — LIME local explanation for all regression models, corrected

# 1. Extract and transform data once using a reference model's preprocessor
# (Assuming all pipelines share an identically structured preprocessor pipeline)
reference_model_name = reg_results_df["Model"].iloc[0]
reference_pipe = reg_predictions[reference_model_name]["pipeline"]

X_train_trans = reference_pipe.named_steps["preprocess"].transform(X_train)
X_test_trans = reference_pipe.named_steps["preprocess"].transform(X_test)

# Recover feature names
feature_names_num = numeric_features
feature_names_cat = (
    reference_pipe
    .named_steps["preprocess"]
    .named_transformers_["cat"]
    .named_steps["onehot"]
    .get_feature_names_out(categorical_features)
    .tolist()
)
feature_names_all = feature_names_num + feature_names_cat

X_train_trans_df = pd.DataFrame(X_train_trans, columns=feature_names_all)
X_test_trans_df = pd.DataFrame(X_test_trans, columns=feature_names_all)

# Initialize the LIME explainer with the shared training structure
lime_explainer_reg = LimeTabularExplainer(
    training_data=X_train_trans_df.values,
    feature_names=feature_names_all,
    mode="regression",
    random_state=RANDOM_STATE
)

sample_index = 0
sample_actual_log = y_test.iloc[sample_index]
all_models = reg_results_df["Model"].tolist()

# 2. Loop through every model to generate individual LIME explanations
for model_name in all_models:
    print(f"Generating LIME explanation for model: {model_name}...")

    pipe = reg_predictions[model_name]["pipeline"]
    model_only = pipe.named_steps["model"]

    # Define custom prediction function mapped to the current model architecture
    def predict_transformed_for_lime(x_array):
        x_df = pd.DataFrame(x_array, columns=feature_names_all)
        return model_only.predict(x_df)

    # Compute prediction locally for the current model
    sample_pred_log = predict_transformed_for_lime(
        X_test_trans_df.iloc[[sample_index]].values
    )[0]

    # Compute the instance feature weights
    lime_exp = lime_explainer_reg.explain_instance(
        X_test_trans_df.values[sample_index],
        predict_transformed_for_lime,
        num_features=15
    )

    # Format and save local explanation weights table
    lime_reg_table = pd.DataFrame(
        lime_exp.as_list(),
        columns=["feature_condition", "lime_weight"]
    )
    save_table(lime_reg_table, f"lime_local_explanation_regression_{model_name.lower()}_sample_{sample_index}")

    # Generate and display the explanation plot
    fig = lime_exp.as_pyplot_figure()
    plt.title(f"LIME Local Explanation: {model_name} (Sample {sample_index})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig(f"lime_local_explanation_regression_{model_name.lower()}_sample_{sample_index}")
    plt.show()

    # Compute error and metrics summary table
    prediction_summary = pd.DataFrame([{
        "sample_index": sample_index,
        "actual_log_yield": sample_actual_log,
        "predicted_log_yield": sample_pred_log,
        "actual_yield": np.expm1(sample_actual_log),
        "predicted_yield": np.expm1(sample_pred_log),
        "absolute_log_error": abs(sample_actual_log - sample_pred_log),
        "absolute_yield_error": abs(np.expm1(sample_actual_log) - np.expm1(sample_pred_log))
    }])
    save_table(prediction_summary, f"lime_regression_{model_name.lower()}_sample_prediction_summary")

    # Print the corresponding tables below the current model's plot
    print("=" * 75)
    print(f"LIME EXPLANATION DATA FOR MODEL: {model_name}")
    print("=" * 75)

    print("\n[1/2] Feature Condition Weights Table:")
    display(lime_reg_table)

    print("\n[2/2] Local Prediction Summary:")
    display(prediction_summary)
    print("\n" + "_"*90 + "\n")


# In[42]:


# Cell 43 — Non-SHAP explainability for classification models
# Fast and publication-friendly alternative to SHAP for Colab Free

import os
import gc
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from lime.lime_tabular import LimeTabularExplainer

def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name))

# Safety checks
if "cls_predictions" not in globals():
    raise NameError("cls_predictions is missing. Please run Cell 34 first.")

if "X_test_cls" not in globals() or "y_test_cls" not in globals():
    raise NameError("X_test_cls or y_test_cls is missing. Please run Cell 33 first.")

if "label_encoder" not in globals():
    raise NameError("label_encoder is missing. Please run Cell 33 first.")

print("Classification models available:")
print(list(cls_predictions.keys()))

# Select best classification model using Macro F1
best_cls_model_name = cls_results_df.sort_values("Macro_F1", ascending=False).iloc[0]["Model"]
best_cls_pipe = cls_predictions[best_cls_model_name]["pipeline"]
best_cls_pred = cls_predictions[best_cls_model_name]["prediction"]

print("Best classification model:", best_cls_model_name)

# ---------------------------------------------------------------------
# 1. Permutation importance for all classification models
# ---------------------------------------------------------------------

all_perm_cls_tables = []

# n_repeats=5 is fast and acceptable; use 10 only if runtime allows
N_REPEATS = 5

for model_name, model_item in cls_predictions.items():

    print("\n" + "=" * 80)
    print("Running permutation importance for:", model_name)
    print("=" * 80)

    safe_model = safe_name(model_name)
    pipe = model_item["pipeline"]

    perm_cls = permutation_importance(
        pipe,
        X_test_cls,
        y_test_cls,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        scoring="f1_macro",
        n_jobs=-1
    )

    perm_cls_df = pd.DataFrame({
        "Model": model_name,
        "feature": X_test_cls.columns,
        "importance_mean": perm_cls.importances_mean,
        "importance_std": perm_cls.importances_std
    }).sort_values("importance_mean", ascending=False)

    all_perm_cls_tables.append(perm_cls_df)

    save_table(
        perm_cls_df,
        f"permutation_importance_classification_{safe_model}"
    )

    top_perm_cls = perm_cls_df.head(20)

    plt.figure(figsize=(8, 6))
    plt.barh(
        top_perm_cls["feature"][::-1],
        top_perm_cls["importance_mean"][::-1]
    )
    plt.title(f"Permutation importance: {model_name}")
    plt.xlabel("Mean decrease in macro F1")
    plt.ylabel("Feature")
    save_fig(f"permutation_importance_classification_{safe_model}")

    print("Top permutation importance features:")
    display(top_perm_cls)

    gc.collect()
    plt.close("all")

all_perm_cls_df = pd.concat(all_perm_cls_tables, ignore_index=True)

save_table(
    all_perm_cls_df,
    "all_models_permutation_importance_classification"
)

# ---------------------------------------------------------------------
# 2. Cross-model permutation importance
# ---------------------------------------------------------------------

cross_model_perm = (
    all_perm_cls_df
    .groupby("feature")
    .agg(
        mean_importance_across_models=("importance_mean", "mean"),
        median_importance_across_models=("importance_mean", "median"),
        std_importance_across_models=("importance_mean", "std"),
        model_count=("Model", "nunique")
    )
    .reset_index()
    .sort_values("mean_importance_across_models", ascending=False)
)

save_table(
    cross_model_perm,
    "cross_model_mean_permutation_importance_classification"
)

top_cross_perm = cross_model_perm.head(25)

plt.figure(figsize=(9, 7))
plt.barh(
    top_cross_perm["feature"][::-1],
    top_cross_perm["mean_importance_across_models"][::-1]
)
plt.title("Top classification predictors averaged across models")
plt.xlabel("Mean permutation importance across models")
plt.ylabel("Feature")
save_fig("cross_model_mean_permutation_importance_classification")

print("Cross-model important features:")
display(top_cross_perm)

# ---------------------------------------------------------------------
# 3. Model-wise rank matrix from permutation importance
# ---------------------------------------------------------------------

ranked_perm = all_perm_cls_df.copy()

ranked_perm["rank"] = (
    ranked_perm
    .groupby("Model")["importance_mean"]
    .rank(ascending=False, method="dense")
)

ranked_perm_top = ranked_perm[ranked_perm["rank"] <= 20].copy()

perm_rank_pivot = ranked_perm_top.pivot_table(
    index="feature",
    columns="Model",
    values="rank",
    aggfunc="min"
)

save_table(
    perm_rank_pivot.reset_index(),
    "model_wise_top20_permutation_feature_rank_matrix_classification"
)

print("Model-wise permutation rank matrix. Lower rank means higher importance.")
display(perm_rank_pivot)

# ---------------------------------------------------------------------
# 4. Model-native importance for tree-based models
# ---------------------------------------------------------------------

native_importance_tables = []

for model_name, model_item in cls_predictions.items():

    pipe = model_item["pipeline"]
    model_only = pipe.named_steps["model"]

    if hasattr(model_only, "feature_importances_"):

        print("\nExtracting model-native importance for:", model_name)

        # Get transformed feature names
        feature_names_num_cls = numeric_features

        feature_names_cat_cls = (
            pipe
            .named_steps["preprocess"]
            .named_transformers_["cat"]
            .named_steps["onehot"]
            .get_feature_names_out(categorical_features)
            .tolist()
        )

        feature_names_all_cls = feature_names_num_cls + feature_names_cat_cls

        native_df = pd.DataFrame({
            "Model": model_name,
            "feature": feature_names_all_cls,
            "native_importance": model_only.feature_importances_
        }).sort_values("native_importance", ascending=False)

        native_importance_tables.append(native_df)

        safe_model = safe_name(model_name)

        save_table(
            native_df,
            f"native_feature_importance_classification_{safe_model}"
        )

        top_native = native_df.head(25)

        plt.figure(figsize=(9, 7))
        plt.barh(
            top_native["feature"][::-1],
            top_native["native_importance"][::-1]
        )
        plt.title(f"Model-native feature importance: {model_name}")
        plt.xlabel("Native importance")
        plt.ylabel("Feature")
        save_fig(f"native_feature_importance_classification_{safe_model}")

        display(top_native)

if len(native_importance_tables) > 0:
    all_native_importance_df = pd.concat(native_importance_tables, ignore_index=True)

    save_table(
        all_native_importance_df,
        "all_tree_models_native_feature_importance_classification"
    )
else:
    all_native_importance_df = pd.DataFrame()
    print("No model-native tree importance available.")

# ---------------------------------------------------------------------
# 5. LIME local explanation for best classifier only
# ---------------------------------------------------------------------

print("\nRunning LIME for best classifier only:", best_cls_model_name)

pipe = best_cls_pipe
model_only = pipe.named_steps["model"]

X_train_cls_trans = pipe.named_steps["preprocess"].transform(X_train_cls)
X_test_cls_trans = pipe.named_steps["preprocess"].transform(X_test_cls)

feature_names_num_cls = numeric_features

feature_names_cat_cls = (
    pipe
    .named_steps["preprocess"]
    .named_transformers_["cat"]
    .named_steps["onehot"]
    .get_feature_names_out(categorical_features)
    .tolist()
)

feature_names_all_cls = feature_names_num_cls + feature_names_cat_cls

X_train_cls_trans_df = pd.DataFrame(
    X_train_cls_trans,
    columns=feature_names_all_cls
)

X_test_cls_trans_df = pd.DataFrame(
    X_test_cls_trans,
    columns=feature_names_all_cls
)

lime_explainer_cls = LimeTabularExplainer(
    training_data=X_train_cls_trans_df.values,
    feature_names=feature_names_all_cls,
    class_names=label_encoder.classes_,
    mode="classification",
    random_state=RANDOM_STATE
)

def predict_proba_transformed_for_lime(x_array):
    x_df = pd.DataFrame(x_array, columns=feature_names_all_cls)

    if hasattr(model_only, "predict_proba"):
        return model_only.predict_proba(x_df)
    else:
        preds = model_only.predict(x_df)
        out = np.zeros((len(preds), len(label_encoder.classes_)))
        for i, p in enumerate(preds):
            out[i, int(p)] = 1
        return out

# Choose one correctly classified and one incorrectly classified sample if possible
pred_test = best_cls_pipe.predict(X_test_cls)
correct_indices = np.where(pred_test == y_test_cls)[0]
wrong_indices = np.where(pred_test != y_test_cls)[0]

selected_indices = []

if len(correct_indices) > 0:
    selected_indices.append(int(correct_indices[0]))

if len(wrong_indices) > 0:
    selected_indices.append(int(wrong_indices[0]))

# If all predictions are correct or no wrong sample exists, add another sample
if len(selected_indices) == 1 and len(X_test_cls) > 1:
    selected_indices.append(1)

lime_all_rows = []

for sample_index_cls in selected_indices:

    print("\nLIME sample index:", sample_index_cls)

    lime_exp_cls = lime_explainer_cls.explain_instance(
        X_test_cls_trans_df.values[sample_index_cls],
        predict_proba_transformed_for_lime,
        num_features=15
    )

    lime_cls_table = pd.DataFrame(
        lime_exp_cls.as_list(),
        columns=["feature_condition", "lime_weight"]
    )

    actual_class = label_encoder.inverse_transform([y_test_cls[sample_index_cls]])[0]
    predicted_class = label_encoder.inverse_transform([pred_test[sample_index_cls]])[0]

    lime_cls_table.insert(0, "Model", best_cls_model_name)
    lime_cls_table.insert(1, "sample_index", sample_index_cls)
    lime_cls_table.insert(2, "actual_class", actual_class)
    lime_cls_table.insert(3, "predicted_class", predicted_class)

    lime_all_rows.append(lime_cls_table)

    save_table(
        lime_cls_table,
        f"lime_local_explanation_best_classifier_sample_{sample_index_cls}"
    )

    fig = lime_exp_cls.as_pyplot_figure()
    plt.title(
        f"LIME explanation: {best_cls_model_name}, sample {sample_index_cls}\n"
        f"Actual: {actual_class}, Predicted: {predicted_class}"
    )
    save_fig(f"lime_local_explanation_best_classifier_sample_{sample_index_cls}")

    display(lime_cls_table)

if len(lime_all_rows) > 0:
    all_lime_cls_df = pd.concat(lime_all_rows, ignore_index=True)

    save_table(
        all_lime_cls_df,
        "lime_local_explanations_best_classifier"
    )

print("Non-SHAP classification explainability completed.")


# In[43]:


# Cell 43A — Permutation importance for all classification models

from sklearn.inspection import permutation_importance

all_perm_cls_tables = []

for model_name, model_item in cls_predictions.items():
    print("\n" + "=" * 80)
    print("Running permutation importance for:", model_name)
    print("=" * 80)

    safe_model = safe_name(model_name)
    pipe = model_item["pipeline"]

    perm_cls = permutation_importance(
        pipe,
        X_test_cls,
        y_test_cls,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="f1_macro",
        n_jobs=-1
    )

    perm_cls_df = pd.DataFrame({
        "Model": model_name,
        "feature": X_test_cls.columns,
        "importance_mean": perm_cls.importances_mean,
        "importance_std": perm_cls.importances_std
    }).sort_values("importance_mean", ascending=False)

    all_perm_cls_tables.append(perm_cls_df)

    save_table(
        perm_cls_df,
        f"permutation_importance_classification_{safe_model}"
    )

    top_perm_cls = perm_cls_df.head(20)

    plt.figure(figsize=(8, 6))
    plt.barh(
        top_perm_cls["feature"][::-1],
        top_perm_cls["importance_mean"][::-1]
    )
    plt.title(f"Permutation importance: {model_name}")
    plt.xlabel("Mean decrease in macro F1")
    plt.ylabel("Feature")
    save_fig(f"permutation_importance_classification_{safe_model}")

    print("Tabular values below the plot:")
    display(top_perm_cls)

all_perm_cls_df = pd.concat(all_perm_cls_tables, ignore_index=True)

save_table(
    all_perm_cls_df,
    "all_models_permutation_importance_classification"
)

cross_model_perm = (
    all_perm_cls_df
    .groupby("feature")
    .agg(
        mean_importance_across_models=("importance_mean", "mean"),
        median_importance_across_models=("importance_mean", "median"),
        model_count=("Model", "nunique")
    )
    .reset_index()
    .sort_values("mean_importance_across_models", ascending=False)
)

save_table(
    cross_model_perm,
    "cross_model_mean_permutation_importance_classification"
)

top_cross_perm = cross_model_perm.head(25)

plt.figure(figsize=(9, 7))
plt.barh(
    top_cross_perm["feature"][::-1],
    top_cross_perm["mean_importance_across_models"][::-1]
)
plt.title("Top permutation features averaged across classification models")
plt.xlabel("Mean permutation importance across models")
plt.ylabel("Feature")
save_fig("cross_model_mean_permutation_importance_classification")

print("Tabular values below the plot:")
display(top_cross_perm)


# In[44]:


# Cell 43B — LIME local explanation for all classification models

from lime.lime_tabular import LimeTabularExplainer

all_lime_cls_tables = []

sample_index_cls = 0

for model_name, model_item in cls_predictions.items():
    print("\n" + "=" * 80)
    print("Running LIME for classification model:", model_name)
    print("=" * 80)

    safe_model = safe_name(model_name)
    pipe = model_item["pipeline"]
    model_only = pipe.named_steps["model"]

    # Transform train and test data using this model's fitted preprocessor
    X_train_cls_trans = pipe.named_steps["preprocess"].transform(X_train_cls)
    X_test_cls_trans = pipe.named_steps["preprocess"].transform(X_test_cls)

    X_train_cls_trans_df = pd.DataFrame(
        X_train_cls_trans,
        columns=feature_names_all_cls
    )

    X_test_cls_trans_df = pd.DataFrame(
        X_test_cls_trans,
        columns=feature_names_all_cls
    )

    lime_explainer_cls = LimeTabularExplainer(
        training_data=X_train_cls_trans_df.values,
        feature_names=feature_names_all_cls,
        class_names=label_encoder.classes_,
        mode="classification",
        random_state=RANDOM_STATE
    )

    def predict_proba_transformed_for_lime(x_array):
        x_df = pd.DataFrame(x_array, columns=feature_names_all_cls)

        if hasattr(model_only, "predict_proba"):
            return model_only.predict_proba(x_df)
        else:
            preds = model_only.predict(x_df)
            out = np.zeros((len(preds), len(label_encoder.classes_)))
            for i, p in enumerate(preds):
                out[i, int(p)] = 1
            return out

    try:
        lime_exp_cls = lime_explainer_cls.explain_instance(
            X_test_cls_trans_df.values[sample_index_cls],
            predict_proba_transformed_for_lime,
            num_features=15
        )

        lime_cls_table = pd.DataFrame(
            lime_exp_cls.as_list(),
            columns=["feature_condition", "lime_weight"]
        )

        lime_cls_table.insert(0, "Model", model_name)
        all_lime_cls_tables.append(lime_cls_table)

        save_table(
            lime_cls_table,
            f"lime_local_explanation_classification_{safe_model}_sample_{sample_index_cls}"
        )

        fig = lime_exp_cls.as_pyplot_figure()
        plt.title(f"LIME local explanation: {model_name}")
        save_fig(f"lime_local_explanation_classification_{safe_model}_sample_{sample_index_cls}")

        model_pred = pipe.predict(X_test_cls.iloc[[sample_index_cls]])[0]

        actual_class = label_encoder.inverse_transform([y_test_cls[sample_index_cls]])[0]
        predicted_class = label_encoder.inverse_transform([model_pred])[0]

        lime_prediction_summary = pd.DataFrame([{
            "Model": model_name,
            "sample_index": sample_index_cls,
            "actual_class": actual_class,
            "predicted_class": predicted_class
        }])

        save_table(
            lime_prediction_summary,
            f"lime_classification_prediction_summary_{safe_model}"
        )

        print("Tabular values below the LIME plot:")
        display(lime_cls_table)
        display(lime_prediction_summary)

    except Exception as e:
        print(f"LIME failed for {model_name}")
        print("Reason:", e)

if len(all_lime_cls_tables) > 0:
    all_lime_cls_df = pd.concat(all_lime_cls_tables, ignore_index=True)

    save_table(
        all_lime_cls_df,
        "all_models_lime_local_explanations_classification"
    )
else:
    print("No LIME classification explanations were generated.")


# In[45]:


# Cell 44— Permutation importance for best classifier

perm_cls = permutation_importance(
    best_cls_pipe,
    X_test_cls,
    y_test_cls,
    n_repeats=10,
    random_state=RANDOM_STATE,
    scoring="f1_macro",
    n_jobs=-1
)

perm_cls_df = pd.DataFrame({
    "feature": X_test_cls.columns,
    "importance_mean": perm_cls.importances_mean,
    "importance_std": perm_cls.importances_std
}).sort_values("importance_mean", ascending=False)

save_table(perm_cls_df, "permutation_importance_best_classifier")

plt.figure(figsize=(8, 6))
top_perm_cls = perm_cls_df.head(20)
plt.barh(top_perm_cls["feature"][::-1], top_perm_cls["importance_mean"][::-1])
plt.title(f"Permutation importance for efficiency classification: {best_cls_model_name}")
plt.xlabel("Mean decrease in macro F1")
plt.ylabel("Feature")
save_fig("permutation_importance_best_classifier")

print("Tabular values below the plot:")
display(top_perm_cls)


# In[47]:


# Cell 45 — LIME local explanation for classification, corrected

from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Select best classification model if not already defined
# ---------------------------------------------------------

if "best_cls_model_name" not in globals():
    best_cls_model_name = cls_results_df.sort_values(
        "Macro_F1",
        ascending=False
    ).iloc[0]["Model"]

if "best_cls_pipe" not in globals():
    best_cls_pipe = cls_predictions[best_cls_model_name]["pipeline"]

if "best_cls_pred" not in globals():
    best_cls_pred = cls_predictions[best_cls_model_name]["prediction"]

# This line fixes your error
cls_model_only = best_cls_pipe.named_steps["model"]

print("Best classification model for LIME:", best_cls_model_name)

# ---------------------------------------------------------
# 2. Transform train and test data using best classifier pipeline
# ---------------------------------------------------------

X_train_cls_trans = best_cls_pipe.named_steps["preprocess"].transform(X_train_cls)
X_test_cls_trans = best_cls_pipe.named_steps["preprocess"].transform(X_test_cls)

feature_names_num_cls = numeric_features

feature_names_cat_cls = (
    best_cls_pipe
    .named_steps["preprocess"]
    .named_transformers_["cat"]
    .named_steps["onehot"]
    .get_feature_names_out(categorical_features)
    .tolist()
)

feature_names_all_cls = feature_names_num_cls + feature_names_cat_cls

X_train_cls_trans_df = pd.DataFrame(
    X_train_cls_trans,
    columns=feature_names_all_cls
)

X_test_cls_trans_df = pd.DataFrame(
    X_test_cls_trans,
    columns=feature_names_all_cls
)

# ---------------------------------------------------------
# 3. Create LIME explainer
# ---------------------------------------------------------

lime_explainer_cls = LimeTabularExplainer(
    training_data=X_train_cls_trans_df.values,
    feature_names=feature_names_all_cls,
    class_names=label_encoder.classes_,
    mode="classification",
    random_state=RANDOM_STATE
)

# ---------------------------------------------------------
# 4. Prediction function for LIME
# ---------------------------------------------------------

def predict_proba_transformed_for_lime(x_array):
    x_df = pd.DataFrame(x_array, columns=feature_names_all_cls)

    if hasattr(cls_model_only, "predict_proba"):
        return cls_model_only.predict_proba(x_df)
    else:
        preds = cls_model_only.predict(x_df)
        out = np.zeros((len(preds), len(label_encoder.classes_)))
        for i, p in enumerate(preds):
            out[i, int(p)] = 1
        return out

# ---------------------------------------------------------
# 5. Explain one local sample
# ---------------------------------------------------------

sample_index_cls = 0

lime_exp_cls = lime_explainer_cls.explain_instance(
    X_test_cls_trans_df.values[sample_index_cls],
    predict_proba_transformed_for_lime,
    num_features=15
)

lime_cls_table = pd.DataFrame(
    lime_exp_cls.as_list(),
    columns=["feature_condition", "lime_weight"]
)

save_table(
    lime_cls_table,
    "lime_local_explanation_classification_sample_0"
)

fig = lime_exp_cls.as_pyplot_figure()
plt.title(
    f"LIME local explanation for efficiency-class prediction: {best_cls_model_name}"
)
save_fig("lime_local_explanation_classification_sample_0")

actual_class = label_encoder.inverse_transform(
    [y_test_cls[sample_index_cls]]
)[0]

predicted_class = label_encoder.inverse_transform(
    [best_cls_pred[sample_index_cls]]
)[0]

print("Actual class:", actual_class)
print("Predicted class:", predicted_class)

prediction_summary = pd.DataFrame([{
    "model": best_cls_model_name,
    "sample_index": sample_index_cls,
    "actual_class": actual_class,
    "predicted_class": predicted_class
}])

save_table(
    prediction_summary,
    "lime_classification_sample_0_prediction_summary"
)

display(lime_cls_table)
display(prediction_summary)


# In[48]:


# Cell 46 — Counterfactual-style sensitivity analysis

# This tests how predicted yield changes when fertilizer intensity is modified for selected samples.

sensitivity_feature = "log_fertilizer_per_ha"

sensitivity_rows = []

base_samples = X_test.sample(min(100, len(X_test)), random_state=RANDOM_STATE).copy()

quantiles = np.quantile(X_test[sensitivity_feature], [0.1, 0.25, 0.5, 0.75, 0.9])

for q in quantiles:
    X_mod = base_samples.copy()
    X_mod[sensitivity_feature] = q
    pred_log = best_reg_pipe.predict(X_mod)

    sensitivity_rows.append({
        "modified_feature": sensitivity_feature,
        "feature_value_log_scale": q,
        "feature_value_original_scale": np.expm1(q),
        "mean_predicted_log_yield": np.mean(pred_log),
        "mean_predicted_yield": np.expm1(np.mean(pred_log))
    })

sensitivity_df = pd.DataFrame(sensitivity_rows)
save_table(sensitivity_df, "counterfactual_style_fertilizer_sensitivity")

plt.figure(figsize=(7, 5))
plt.plot(
    sensitivity_df["feature_value_original_scale"],
    sensitivity_df["mean_predicted_yield"],
    marker="o"
)
plt.title("Counterfactual-style fertilizer sensitivity")
plt.xlabel("Fertilizer per ha")
plt.ylabel("Mean predicted yield")
plt.grid(True, alpha=0.3)
save_fig("counterfactual_style_fertilizer_sensitivity")

print("Tabular values below the plot:")
display(sensitivity_df)


# In[50]:


# Cell 47 — Crop-specific model performance, corrected

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Select best regression model if not already defined
# ---------------------------------------------------------

if "best_reg_model_name" not in globals():
    best_reg_model_name = reg_results_df.sort_values("RMSE").iloc[0]["Model"]

if "best_reg_pipe" not in globals():
    best_reg_pipe = reg_predictions[best_reg_model_name]["pipeline"]

# This line fixes the error
if "best_pred_log" not in globals():
    best_pred_log = best_reg_pipe.predict(X_test)

print("Best regression model used for crop-specific error analysis:", best_reg_model_name)

# ---------------------------------------------------------
# 2. Build crop-specific test metadata
# ---------------------------------------------------------

test_meta = X_test.copy()

test_meta["actual_log_yield"] = y_test.values
test_meta["predicted_log_yield"] = best_pred_log

test_meta["actual_yield"] = np.expm1(test_meta["actual_log_yield"])
test_meta["predicted_yield"] = np.expm1(test_meta["predicted_log_yield"])

test_meta["abs_error"] = np.abs(
    test_meta["actual_yield"] - test_meta["predicted_yield"]
)

# ---------------------------------------------------------
# 3. Crop-wise performance table
# ---------------------------------------------------------

crop_perf = (
    test_meta.groupby("crop")
    .agg(
        test_records=("abs_error", "size"),
        mean_actual_yield=("actual_yield", "mean"),
        mean_predicted_yield=("predicted_yield", "mean"),
        mean_abs_error=("abs_error", "mean"),
        median_abs_error=("abs_error", "median"),
        std_abs_error=("abs_error", "std")
    )
    .reset_index()
)

crop_perf = (
    crop_perf[crop_perf["test_records"] >= 5]
    .sort_values("mean_abs_error", ascending=False)
)

save_table(
    crop_perf,
    "crop_specific_prediction_error_analysis"
)

# ---------------------------------------------------------
# 4. Plot top crops by prediction error
# ---------------------------------------------------------

top_error_crops = crop_perf.head(20)

plt.figure(figsize=(9, 7))
plt.barh(
    top_error_crops["crop"][::-1],
    top_error_crops["mean_abs_error"][::-1]
)
plt.title(f"Top crops by prediction error: {best_reg_model_name}")
plt.xlabel("Mean absolute error")
plt.ylabel("Crop")
save_fig("top_crop_prediction_errors")

print("Tabular values below the plot:")
display(top_error_crops)


# In[51]:


# Cell 48 — State-specific model performance

state_perf = (
    test_meta.groupby("state")
    .agg(
        test_records=("abs_error", "size"),
        mean_actual_yield=("actual_yield", "mean"),
        mean_predicted_yield=("predicted_yield", "mean"),
        mean_abs_error=("abs_error", "mean")
    )
    .reset_index()
)

state_perf = state_perf[state_perf["test_records"] >= 5].sort_values("mean_abs_error", ascending=False)
save_table(state_perf, "state_specific_prediction_error_analysis")

plt.figure(figsize=(9, 8))
plt.barh(state_perf["state"][::-1], state_perf["mean_abs_error"][::-1])
plt.title("State-wise prediction error")
plt.xlabel("Mean absolute error")
plt.ylabel("State")
save_fig("state_wise_prediction_error")

print("Tabular values below the plot:")
display(state_perf)


# In[52]:


# Cell 49 — Statistical comparison of model errors

model_error_rows = []

for model_name, item in reg_predictions.items():
    pred = item["prediction_log"]
    abs_error = np.abs(y_test.values - pred)
    for e in abs_error:
        model_error_rows.append({
            "model": model_name,
            "absolute_log_error": e
        })

model_error_df = pd.DataFrame(model_error_rows)

groups = [
    sub["absolute_log_error"].values
    for _, sub in model_error_df.groupby("model")
]

kw_stat, kw_p = kruskal(*groups)

model_error_test = pd.DataFrame([{
    "test": "Kruskal-Wallis across regression model absolute log-errors",
    "statistic": kw_stat,
    "p_value": kw_p,
    "significant_at_0_05": kw_p < 0.05
}])

save_table(model_error_test, "model_error_kruskal_wallis_test")

model_error_summary = (
    model_error_df.groupby("model")
    .agg(
        mean_abs_log_error=("absolute_log_error", "mean"),
        median_abs_log_error=("absolute_log_error", "median"),
        std_abs_log_error=("absolute_log_error", "std")
    )
    .reset_index()
    .sort_values("mean_abs_log_error")
)

save_table(model_error_summary, "model_error_distribution_summary")

plt.figure(figsize=(9, 6))
plt.barh(model_error_summary["model"][::-1], model_error_summary["mean_abs_log_error"][::-1])
plt.title("Mean absolute log-error by regression model")
plt.xlabel("Mean absolute log-error")
plt.ylabel("Model")
save_fig("mean_absolute_log_error_by_model")

print("Tabular values below the plot:")
display(model_error_summary)


# In[53]:


# Cell 50 — Save final merged dataset and key result tables, corrected and expanded

import os
import pandas as pd

merged_path = os.path.join(
    OUTPUT_DIR,
    "merged_crop_soil_weather_input_dataset.csv"
)

df.to_csv(merged_path, index=False)

summary_excel_path = os.path.join(
    OUTPUT_DIR,
    "AgroInput_XAI_all_key_results.xlsx"
)

# Helper: write only if variable exists and is a non-empty DataFrame
def write_if_exists(writer, var_name, sheet_name):
    if var_name in globals():
        obj = globals()[var_name]

        if isinstance(obj, pd.DataFrame) and not obj.empty:
            # Excel sheet names must be <=31 characters
            safe_sheet = sheet_name[:31]
            obj.to_excel(writer, sheet_name=safe_sheet, index=False)
            print(f"Written sheet: {safe_sheet}")
        else:
            print(f"Skipped {var_name}: not a non-empty DataFrame")
    else:
        print(f"Skipped {var_name}: variable not found")

with pd.ExcelWriter(summary_excel_path, engine="openpyxl") as writer:

    # Dataset audit and descriptive analysis
    write_if_exists(writer, "audit_df", "dataset_audit")
    write_if_exists(writer, "merge_audit", "merged_missing_audit")
    write_if_exists(writer, "structure_summary", "structure")
    write_if_exists(writer, "desc", "descriptive_stats")
    write_if_exists(writer, "distribution_summary", "distribution_summary")

    # Agronomic summaries
    write_if_exists(writer, "class_table", "efficiency_class_dist")
    write_if_exists(writer, "season_yield", "season_summary")
    write_if_exists(writer, "year_trend", "year_trend")
    write_if_exists(writer, "crop_summary_filtered", "crop_summary")
    write_if_exists(writer, "state_summary", "state_summary")

    # Statistical testing
    write_if_exists(writer, "normality_results", "normality")
    write_if_exists(writer, "kruskal_results", "kruskal")
    write_if_exists(writer, "posthoc_season_yield", "posthoc_season")
    write_if_exists(writer, "assoc_results", "spearman_yield")
    write_if_exists(writer, "vif_results", "vif_results")

    # Regression modelling
    write_if_exists(writer, "reg_results_df", "regression_random")
    write_if_exists(writer, "temporal_results_df", "temporal_validation")
    write_if_exists(writer, "state_group_results_df", "leave_state_out")

    # Regression explainability
    write_if_exists(writer, "shap_importance", "shap_regression")
    write_if_exists(writer, "perm_df", "perm_regression")
    write_if_exists(writer, "all_pdp_df", "pdp_regression")
    write_if_exists(writer, "all_ale_df", "ale_regression")

    # Classification modelling
    write_if_exists(writer, "cls_results_df", "classification")

    # Non-SHAP classification explainability
    write_if_exists(writer, "all_perm_cls_df", "perm_classification")
    write_if_exists(writer, "cross_model_perm", "cross_perm_cls")
    write_if_exists(writer, "perm_rank_pivot", "perm_rank_matrix")
    write_if_exists(writer, "all_native_importance_df", "native_tree_importance")
    write_if_exists(writer, "all_lime_cls_df", "lime_classification")

    # Sensitivity and subgroup error analysis
    write_if_exists(writer, "sensitivity_df", "fertilizer_sensitivity")
    write_if_exists(writer, "crop_perf", "crop_error_analysis")
    write_if_exists(writer, "state_perf", "state_error_analysis")
    write_if_exists(writer, "model_error_test", "model_error_test")
    write_if_exists(writer, "model_error_summary", "model_error_summary")

print("Saved merged dataset:", merged_path)
print("Saved summary Excel:", summary_excel_path)


# In[54]:


# Cell 51 — Zip all results for download

import shutil

zip_path = shutil.make_archive("AgroInput_XAI_outputs", "zip", OUTPUT_DIR)

print("Created zip file:", zip_path)

from google.colab import files
files.download(zip_path)


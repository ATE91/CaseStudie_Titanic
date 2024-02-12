
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def calc_corr_target_num(df, rel_num_cols, target_col):
    num_correlations = {}

    for col in rel_num_cols:
        correlation_tmp = df[col].corr(df[target_col], method='pearson')
        num_correlations[col] = correlation_tmp

    df_num_correlations = pd.DataFrame.from_dict(num_correlations, orient='index', columns=['Correlation'])

    return df_num_correlations


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def calc_corr_target_cat(df, rel_cat_cols, target_col):
    df_cat_correlations = pd.DataFrame(index=rel_cat_cols, columns=['Correlation'])

    for col in rel_cat_cols:
        correlation_tmp = cramers_v(df[col], df[target_col])
        df_cat_correlations.loc[col, 'Correlation'] = correlation_tmp

    return df_cat_correlations


def print_VIF_unprepared(df, rel_num_cols ,rel_cat_cols):

    # VIF test - >10 mögliche multikollinearität
    print("VIF test - >10 mögliche multikollinearität")
    X_num = df[rel_num_cols].copy()
    X_num = add_constant(X_num)  # add a constant to the model for the intercept

    # Convert all columns to numeric and replace inf with NaN
    X_num = X_num.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)

    # categorical columns
    X_cat = pd.get_dummies(df[rel_cat_cols])

    # entferne erste dummy spalte je variable (um perfekte multikollinearität zu vermeiden)
    cols_tmp = X_cat.columns.tolist()
    for c in rel_cat_cols:
        matching = [ct for ct in cols_tmp if c in ct]
        # entferne erste dummy spalte je variable
        cols_tmp.remove(matching[0])


    X_cat = X_cat[cols_tmp]

    # Kombiniere numerical und categorical (now numerical) features
    X_combined = pd.concat([X_num, X_cat], axis=1)
    X_combined = add_constant(X_combined)  # add a constant to the model for the intercept

    # N/A Zeilen entfernen für VIF Berechnung
    X_tmp_cleaned = X_combined.dropna()
    X_tmp_cleaned = X_tmp_cleaned.astype({
        'SibSp': 'float64',
        'Parch': 'float64',
        'Cabin_String_N': 'float64'
    })

    print(f"X_combined: {X_combined.shape}, X_tmp_cleaned: {X_tmp_cleaned.shape}")

    #print(X_tmp_cleaned.dtypes)
    # Convert boolean columns to integers
    for column in X_tmp_cleaned.columns:
        if X_tmp_cleaned[column].dtype == bool:
            X_tmp_cleaned[column] = X_tmp_cleaned[column].astype(int)


    # Calculate VIF for each feature
    vif_combined = pd.DataFrame()
    vif_combined["Feature"] = X_tmp_cleaned.columns
    vif_combined["VIF"] = [variance_inflation_factor(X_tmp_cleaned.values, i) for i in range(X_tmp_cleaned.shape[1])]

    return print(vif_combined)

def print_VIF(df, nominal_cols):

    X_train_df_tmp = df.copy()

    # entferne erste dummy spalte je nominal_cols variable (um perfekte multikollinearität zu vermeiden)
    cols_tmp = df.columns.tolist()
    for c in nominal_cols:
        matching = [ct for ct in cols_tmp if c in ct]
        cols_tmp.remove(matching[0])

    X_train_df_tmp = X_train_df_tmp[cols_tmp]

    # Calculate VIF for each feature
    vif_combined = pd.DataFrame()
    vif_combined["Feature"] = cols_tmp
    vif_combined["VIF"] = [variance_inflation_factor(X_train_df_tmp.values, i) for i in range(X_train_df_tmp.shape[1])]

    return print(vif_combined)


def calc_corr_triangle(df):
    # Prüfe Multikollinearität
    df_correlation_matrix = df.corr(method='pearson')

    # Getting the Upper Triangle of the co-relation matrix
    matrix = np.triu(df_correlation_matrix)

    # Plotting the heatmap
    plt.figure(figsize=(12, 12))
    sns.color_palette("vlag", as_cmap=True)
    sns.heatmap(df_correlation_matrix, annot=True, mask=matrix, cmap="vlag", center=0, linewidths=0.5)
    return plt.show()
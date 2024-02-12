
import pandas as pd

def imputation_by_avg(df, col2fill, col2groupby):
    df[col2fill] = df.groupby(col2groupby)[col2fill].transform(lambda x: x.fillna(x.mean()))
    return df[col2fill]

def imputation_by_median(df, col2fill, col2groupby):
    df[col2fill] = df.groupby(col2groupby)[col2fill].transform(lambda x: x.fillna(x.median()))
    return df[col2fill]

def imputation_by_mode(df, col2fill, col2groupby):
    df[col2fill] = df.groupby(col2groupby)[col2fill].transform(lambda x: x.fillna(x.mode()[0]))
    return df[col2fill]

def remove_nominal_feature_value(df, nominal_cols):
    cols_tmp = df.columns.tolist()
    for c in nominal_cols:
        matching = [ct for ct in cols_tmp if c in ct]
        try:
            cols_tmp.remove(matching[0])
            print(f"For {c} - removed: {matching[0]}")
        except:
            print(f"Cannot find: {c} in df")
            print(f"Matching: {matching}")
            print(f"Columns: {cols_tmp}")

    df_tmp = df[cols_tmp]
    return df_tmp


def imputation_of_dataframe(df):

    total_nan_sum = df.isna().sum().sum()

    df["Age"] = imputation_by_median(df, "Age", "Pclass")
    df["Fare"] = imputation_by_median(df, "Fare", "Pclass")
    df["Cabin_Level"] = df["Cabin_Level"].fillna("No_Info")
    df["Cabin_String_3"] = df["Cabin_String_3"].fillna("No_Info")
    df["Embarked"] = imputation_by_mode(df, "Embarked", "Pclass")

    print(f"Total NaNs before: {total_nan_sum}, after: {df.isna().sum().sum()}")

    return df
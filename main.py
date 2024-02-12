## Import der installierten Module

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import re
from io import StringIO
import json
import warnings
from datetime import datetime

## Import der eigenen Funktionen

from custom_modules.data_prep import prep_df_dict_format, \
                              get_df_info, \
                              get_df_info_all, \
                              prep_df_format,  \
                              extract_names,  \
                              extract_title_group,  \
                              extract_cabin_level,  \
                              extract_cabin_string_n,  \
                              extract_ticket_string_n
from custom_modules.data_preprocess import imputation_by_median, \
                              imputation_by_mode
from custom_modules.model_util import save_model, load_model




if __name__ == "__main__":

    # Start of the Inference Script
    ts_start = datetime.now()
    print(f"Start of Inference Script: {ts_start}")

    # path to the config.json file
    config_file_path = 'config.json'

    # ppen the config.json file and load its content
    with open(config_file_path, 'r') as file:
        config = json.load(file)

    input_data_path = config['input_data_dir']
    output_data_path = config['output_data_dir']
    plot_path = config['plot_dir']
    model_path = config['model_dir']
    random_state = config['random_state']


    # Load Data
    print("\nLoad Data")
    df_test = pd.read_csv(input_data_path+'test.csv')
    df_gs = pd.read_csv(input_data_path+'gender_submission.csv')
    ids_test = df_gs["PassengerId"]

    print(f"\nTest Data: {df_test.shape}")
    print(f"gender_submission Data: {df_gs.shape}\n")


    # Data Preprocessing
    print("Data Preprocessing\n")
    X_test = df_test.copy()
    X_test = prep_df_format(X_test)
    X_test = extract_names(X_test)
    X_test = extract_title_group(X_test)
    X_test = extract_cabin_level(X_test)
    X_test = extract_cabin_string_n(X_test)
    X_test = extract_ticket_string_n(X_test)


    # Imputation
    # Befülle NA Werte relativ zur Pclass (Annahme ist, das Passagiere der ersten Klasse andere Werte haben als Passagiere der dritten Klasse)
    X_test["Age"] = imputation_by_median(X_test, "Age", "Pclass")
    X_test["Fare"] = imputation_by_median(X_test, "Fare", "Pclass")
    X_test["Cabin_Level"] = X_test["Cabin_Level"].fillna("No_Info")
    X_test["Cabin_String_3"] = X_test["Cabin_String_3"].fillna("No_Info")
    X_test["Embarked"] = imputation_by_mode(X_test, "Embarked", "Pclass")


    # Scaling and Encoding
    print("Scaling and Encoding\n")
    ordinal_cols = ["Pclass", "Cabin_Level", "Title_group"]
    nominal_cols = ["Sex", "Embarked", "Cabin_String_3", "Ticket_String_6"]
    numeric_cols = ["Age", "SibSp", "Parch", "Fare"]

    preprocessor = load_model("preprocessor", model_path)
    X_test_prepared = preprocessor.transform(X_test)
    new_nominal_cols = preprocessor.transformers_[1][1].named_steps["encoder"].get_feature_names_out(nominal_cols).tolist()
    new_cols = ordinal_cols + new_nominal_cols + numeric_cols
    X_test_df = pd.DataFrame(X_test_prepared, columns=new_cols)
    # encoder müsste neu ohne cabin_cols trainiert werden (nicht der Fall, daher vorerst erstmal so)
    cabin_cols = ['Cabin_String_3_0', 'Cabin_String_3_1', 'Cabin_String_3_No_Info', 'Cabin_Level']
    X_test_df = X_test_df.drop(cabin_cols, axis=1)

    # Für result_df
    label_encoder = load_model("label_encoder", model_path)
    y_test = df_gs["Survived"].astype("object")
    y_test = label_encoder.transform(y_test)

    # Load Model
    print("Load Model\n")
    logreg_model = load_model("logreg_baseline", model_path)

    # Predictions
    print("Predictions\n")
    y_pred_lr= logreg_model.predict(X_test_df)
    y_pred_lr_proba = logreg_model.predict_proba(X_test_df)[:,1]

    # Save Predictions
    print("Save Predictions\n")
    result_df = pd.DataFrame({'PassengerId': ids_test, 
                              'Survived': y_test, 
                              'Prediction': y_pred_lr, 
                              'Pred_Prob': logreg_model.predict_proba(X_test_df)[:,1]})

    df_merged = df_test.merge(result_df, on='PassengerId', how='left')

    # End of Inference Script
    ts_end = datetime.now()
    ts_string = str(ts_end).replace(" ", "_").replace(":", "-").replace(".", "_")
    df_merged.to_csv(output_data_path+f'logreg_predictions_{ts_string}.csv', sep=';', index=False)

    print(f"End of Inference Script: {ts_end}")
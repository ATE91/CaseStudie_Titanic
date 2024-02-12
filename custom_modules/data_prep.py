
import numpy as np
import pandas as pd

import re

def get_df_info(df):
    info_data = [] 
    
    for column in df.columns:
        dtype = str(df[column].dtype)
        non_null_count = df[column].notnull().sum()
        null_count = df[column].isnull().sum()
        total_count = df.shape[0]  # Anzahl der Zeilen im DataFrame
        info_data.append({'Column': column, 'Non-Null Count': non_null_count, 'Null Count': null_count, 'Dtype': dtype, 'Total Count': total_count})
    
    info_df = pd.DataFrame(info_data)
    
    return info_df

def get_df_info_all(dfs_dict):    
    dfs_names = list(dfs_dict.keys())
    #all_columns = sorted(set(column for df in dfs_names for column in dfs_dict[df].columns))

    info_columns = ['Column', 'Non-Null Count', 'Null Count', 'Total Count', 'Dtype']
    combined_info_df = pd.DataFrame(columns=info_columns)

    for df in dfs_names:
        info_df = get_df_info(dfs_dict[df])
        info_df['DataFrame'] = f'{df}'
        combined_info_df = pd.concat([combined_info_df, info_df], ignore_index=True)

    pivot_table = combined_info_df.pivot(index='Column', columns='DataFrame', values=info_columns[1:])

    return pivot_table

def prep_df_dict_format(dfs_dict):

    dfs_names = list(dfs_dict.keys())

    int_columns_to_format = ["SibSp", "Parch"]
    float_columns_to_format = ["Age", "Fare"]
    object_columns_to_format = ["Cabin", "Embarked", "Name", "PassengerId", "Pclass", "Sex", "Survived", "Ticket"]

    print(f"len_columns: {len(int_columns_to_format)+len(float_columns_to_format)+len(object_columns_to_format)}")

    for col in int_columns_to_format:
        for df in dfs_names:
            try:
                dfs_dict[df][col] = dfs_dict[df][col].astype('Int64')
            except:
                print(f"Error: {col} in {df}")
    for col in float_columns_to_format:
        for df in dfs_names:
            try:
                dfs_dict[df][col] = dfs_dict[df][col].astype('float64')
            except:
                print(f"Error: {col} in {df}")
    for col in object_columns_to_format:
        for df in dfs_names:
            try:
                dfs_dict[df][col] = dfs_dict[df][col].astype('object')
            except:
                print(f"Error: {col} in {df}")

    return dfs_dict

def prep_df_format(df):

    int_columns_to_format = ["SibSp", "Parch"]
    float_columns_to_format = ["Age", "Fare"]
    object_columns_to_format = ["Cabin", "Embarked", "Name", "PassengerId", "Pclass", "Sex", "Survived", "Ticket"]

    print(f"len_columns: {len(int_columns_to_format)+len(float_columns_to_format)+len(object_columns_to_format)}")

    for col in int_columns_to_format:
        try:
            df[col] = df[col].astype('Int64')
        except:
            print(f"Error: {col} in df with int_columns_to_format")
    for col in float_columns_to_format:
        try:
            df[col] = df[col].astype('float64')
        except:
            print(f"Error: {col} in df with float_columns_to_format")
    for col in object_columns_to_format:
        try:
            df[col] = df[col].astype('object')
        except:
            print(f"Error: {col} in df with object_columns_to_format")

    return df

def create_age_group(df):
    df['age_group'] = pd.cut(df['Age'], 
                             bins=[0, 4, 12, 17, 30, 60, 999], 
                             labels=['Baby', 'Child', 'Teenager', 'Young Adults', 'Adults', 'Senior'])
    return df


def extract_names(df):
    # Vorname, Titel, Nachname
    df['Lastname'] = df['Name'].str.split(',', expand=True)[0].str.strip()
    df['Title'] = df['Name'].str.split(',', expand=True)[1].str.split('.', expand=True)[0].str.strip()
    df['Firstname'] = df['Name'].str.split(',', expand=True)[1].str.split('.', expand=True)[1].str.strip()

    return df

def extract_title_group(df):
    Title_dict = {'Mr': 'Civilian', 
                  'Miss': 'Civilian',  
                  'Mrs': 'Civilian', 
                  'Master': 'Civilian',  
                  'Dr': 'MediReli',  
                  'Rev': 'MediReli', 
                  'Mlle': 'Civilian', 
                  'Major': 'Military', 
                  'Col': 'Military', 
                  'the Countess': 'Noble', 
                  'Capt': 'Military', 
                  'Ms': 'Civilian', 
                  'Sir': 'Noble', 
                  'Lady': 'Noble', 
                  'Mme': 'Civilian', 
                  'Don': 'Noble', 
                  'Jonkheer': 'Noble',
                  'Dona': 'Noble'
                 }

    df["Title_group"] = df["Title"].map(Title_dict)
    return df

def check_cabin_format(df):
    cabin_format = []
    for cabin in df['Cabin']:
        if pd.isnull(cabin):
            cabin_format.append(False)
        else:
            cabin = cabin.replace(" ", "")  # Remove whitespaces from cabin string
            match = re.match(r'^[A-Z0-9]+$', cabin)
            cabin_format.append(bool(match))
    return cabin_format

def extract_cabin_level(df):
    df['Cabin_Format'] = check_cabin_format(df)
    df.loc[df['Cabin_Format'], 'Cabin_Level'] = df.loc[df['Cabin_Format'], 'Cabin'].str[0]
    return df

def extract_cabin_string_n(df):
    cabin_strings_N = []
    for cabin in df["Cabin"]:
        if (pd.isnull(cabin)) | (cabin == "nan"):
            cabin_strings_N.append(999)
        else:
            cabin_strings_N.append(len(cabin))

    df["Cabin_String_N"] = cabin_strings_N
    df['Cabin_String_3'] = df['Cabin_String_N'].apply(
        lambda x: "1" if x == 3 else ("999" if x == 999 else "0")
    )

    df["Cabin_String_N"] = df["Cabin_String_N"].replace(999, np.nan)
    df["Cabin_String_3"] = df["Cabin_String_3"].replace("999", np.nan)
    return df

def extract_ticket_string_n(df):
    df['Ticket_String_6'] = df['Ticket'].apply(lambda x: "1" if len(str(x)) == 6 and str(x).isdigit() else "0")
    return df

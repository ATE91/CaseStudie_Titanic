
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree


def categorical_column_distribution(df, column, color="blue", label="All", alpha=0.7, density=True, dropna=False):   

    value_counts = df[column].value_counts(normalize=density, dropna=dropna)

    sns.barplot(x=value_counts.index.astype(str), y=value_counts.values, color=color, alpha=alpha, label=label)
    #sns.despine()
    #plt.xticks(rotation=90)
    plt.legend().remove()
    return plt


def plot_column_distribution_single(df, column, target=True, alpha=0.7, density=True):

    sns.set_style("ticks")
    fig = plt.figure(figsize=(10, 4))

    if target:
        df_pos = df[df.Survived == 1]
        df_neg = df[df.Survived == 0]
        print(f"df_pos: {df_pos.shape}, df_neg: {df_neg.shape}")
        print(f"df_pos: {df_pos[column].dtype}, df_neg: {df_neg[column].dtype}")

        if df[column].dtype == 'object':
            categorical_column_distribution(df_pos, column, color="blue", label="Survived", alpha=0.7, density=density, dropna=False)
            categorical_column_distribution(df_neg, column, color="red", label="Not Survived", alpha=0.7, density=density, dropna=False)
        else:
            plt.hist(df_pos[column], color="blue", alpha=alpha, bins=10, density=density, label='Survived')
            plt.hist(df_neg[column], color="red", alpha=alpha, bins=10, density=density, label='Not Survived')

    else:
        if df[column].dtype == 'object':
            categorical_column_distribution(df, column, color="blue", label="All", alpha=0.7, density=density, dropna=False)
        else:
            plt.hist(df[column], color="blue", alpha=alpha, bins=10, density=density, label='All')

    fig.legend(loc='upper right')
    plt.xticks(rotation=90) 
    sns.despine()

    return plt


def categorical_column_distribution_ax(df, column, color="blue", label="All", alpha=0.7, density=True, dropna=False, ax=None):   

    value_counts = df[column].value_counts(normalize=density, dropna=dropna)
    values_n = len(value_counts)
    avg_string_n = np.mean([len(i) for i in value_counts.index])

    if avg_string_n > 6:
        x_labels = list(range(values_n)).astype(str)
    else :
        x_labels = value_counts.index.astype(str)

    sns.barplot(x=x_labels, y=value_counts.values, color=color, alpha=alpha, label=label, ax=ax)
    #sns.despine()
    #plt.xticks(rotation=90)
    plt.legend().remove()

    return plt

def plot_column_distribution_all(df_1, 
                                 df_2, 
                                 cat_cols, 
                                 num_cols, 
                                 density=False, 
                                 target=False,
                                 dropna=False):
    
    #warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    df_1 = df_1
    df_2 = df_2
    df_columns = num_cols + cat_cols
    n_columns = len(df_columns)

    if target == False:
        label_1 = "df_train"
        label_2 = "df_test"

    else:
        label_1 = "Survived"
        label_2 = "Not Survived"

    fig, axes = plt.subplots(int(n_columns/2) , 2, figsize=(10, int(n_columns*2)))

    color_dict = {0: 'blue', 1: 'orange'}

    for c, col in enumerate(df_columns):
        i = c // 2
        j = c % 2
        ax = axes[i, j]

        #print(f"\nProcessing {col} {i} {j} {c} {ax} | dtype: {df_train[col].dtype}\n")

        if col in cat_cols:
            try:
                categorical_column_distribution_ax(df_1, 
                                                   col, 
                                                   color=color_dict[0], 
                                                   label=label_1, 
                                                   alpha=0.7, 
                                                   density=density, 
                                                   dropna=dropna,
                                                   ax=ax)

            except:
                pass
                #ax.text(0.5, 0.3, f'N/A ({label_1}) {col}', fontsize=12, ha='center', label=label_1)
            try:
                categorical_column_distribution_ax(df_2, 
                                   col, 
                                   color=color_dict[1], 
                                   label=label_2, 
                                   alpha=0.7, 
                                   density=density, 
                                   dropna=dropna,
                                   ax=ax)
            except:
                pass
                #ax.text(0.5, 0.6, f'N/A ({label_2}) {col}', fontsize=12, ha='center', label=label_2)
                
        elif col in num_cols:
            try:
                ax.hist(df_1[col], color=color_dict[0], alpha=0.7, density=density, dropna=dropna, label=label_1)
            except:
                pass
                #ax.text(0.5, 0.3, f'N/A ({label_1}) {col}', fontsize=12, ha='center', label=label_1)
            try:
                ax.hist(df_2[col], color=color_dict[1], alpha=0.7, density=density, dropna=dropna, label=label_2)
            except:
                pass
                #ax.text(0.5, 0.6, f'N/A ({label_2}) {col}', fontsize=12, ha='center', label=label_2)

        else:
            print(f"\nCannot Process {col} {i} {j} {c} {ax} | dtype: {df_1[col].dtype}\n")

        ax.set_title(f'{col}')
        #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.legend().remove()
        ax.tick_params(axis='x', rotation=90)
        
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')


    plt.show()


def plot_decision_tree(rfc, tree_i, feature_names, class_names, plot_path="plots/"):
    
    tree2visualize = rfc.estimators_[tree_i]
    plt.figure(figsize=(20, 20))
    plot_tree(tree2visualize, 
              filled=True, 
              feature_names=feature_names, 
              class_names=class_names, #["Not S", "S"], 
              rounded=True, 
              proportion=False, 
              precision=2)

    plt.savefig(plot_path+f"RFC_decision_tree_{tree_i}.png", format='png', bbox_inches='tight', dpi=300)
    plt.show()  
    return plt

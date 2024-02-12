
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, \
                            confusion_matrix, \
                            roc_auc_score, \
                            roc_curve, \
                            accuracy_score

import os
from joblib import dump, load
import shap


def save_model(model, name, path):
    filename = str(path) + str(name) + '.joblib'
    dump(model, filename)
    return print(f"Model saved as {filename}")

def load_model(name, path):
    filename = str(path) + str(name) + '.joblib'
    model = load(filename)
    print(f"Model loaded form {filename}")
    return model


def get_accuracy_score(model, X_train, X_test, y_train, y_test):

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_accuracy = round(accuracy_score(y_train, y_pred_train), 4)
    test_accuracy  = round(accuracy_score(y_test, y_pred_test), 4)
    accuracy_gap = round(abs(train_accuracy - test_accuracy), 4)

    print('\nTraining Accurarcy:', train_accuracy)
    print('Testing Accuracy:', test_accuracy)
    print('Train-Test Gap   :', accuracy_gap)

    return train_accuracy, test_accuracy, accuracy_gap


def get_cm_score(y_test, y_pred_test):

    # confusion matrix
    cm = confusion_matrix(y_true = y_test, y_pred = y_pred_test)
    cm_tn, \
    cm_fp, \
    cm_fn, \
    cm_tp = cm.ravel()

    print(f"""
    \nTrue Negatives : {cm_tn}
    False Positives: {cm_fp}
    False Negatives: {cm_fn}
    True Positives : {cm_tp}
    """)

    return cm


def get_cr_score(y_test, y_pred_test):

    # classification report
    cr = classification_report(y_test, y_pred_test, output_dict=True)
    cr_df = pd.DataFrame(cr).transpose()
    cr_df = cr_df.round(2)
    print("\nClassification Report")
    print(cr_df)

    return cr_df


def get_roc_score(model, X_test, y_test, plot_path="plots/"):

    model_string = str(type(model)).split(".")[-1][:-2]
    # ROC Curve 
    y_test_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    auc = roc_auc_score(y_test, y_test_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {round(auc,2)}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(plot_path+f'{model_string}_ROC.png', dpi=300, bbox_inches='tight')
    plt.show()

    return auc


def get_model_coef_df(model, X_train):
    print("Intercept:", model.intercept_)
    print("Coefficients:")

    coef_dict = {}
    for feature, coef in zip(X_train.columns, model.coef_[0]):
        #print(f"{feature}: {coef}")
        coef_dict[str(feature)] = coef

    coef_df = pd.DataFrame(list(coef_dict.items()), columns=['Feature', 'Coefficient'])

    coef_df.sort_values(by='Coefficient', ascending=False)

    return coef_df


def get_shap_explainer(model, X_train, model_type='logreg'):
    if model_type == 'logreg':
        shap_explainer = shap.Explainer(model, X_train)
    elif model_type == 'rfc':
        feature_names = X_train.columns.tolist()
        shap_explainer = shap.TreeExplainer(model, feature_names=feature_names)
    else:
        print("model_type not supported")
    return shap_explainer

def get_shap_importance(shap_values):
    return shap.plots.bar(shap_values)

def get_shap_beeswarm(shap_values):
    return shap.plots.beeswarm(shap_values)

def get_shap_waterfall(shap_values):
    return shap.plots.waterfall(shap_values[0])

def get_shap_waterfall_single(shap_values, df_i):
    shap_values_i = shap_values[df_i.index]
    return shap.plots.waterfall(shap_values_i[0])

def plot_shap_waterfall_single(shap_explainer, df_i):

    shap_values_i = shap_explainer(df_i)

    return shap.plots.waterfall(shap_values_i[0])


def get_cv_param_scoring(random_search_cv, plot_path="plots/"):

    cv_results_df = pd.DataFrame(random_search_cv.cv_results_)
    parameters = [col for col in cv_results_df.columns if col.startswith('param_')]
    best_params = random_search_cv.best_params_

    fig, axs = plt.subplots(5, 1, figsize=(5, 15))
    # Loop über die parameter und plotte die scores
    for i, param in enumerate(parameters[:5]):  
        param_tmp = param[6:]
        if param == 'param_max_features':
            results_tmp = cv_results_df[param].astype(str)
            best_param = str(best_params[param_tmp])
        else:
            results_tmp = cv_results_df[param]
            best_param = best_params[param_tmp]

        axs[i].scatter(results_tmp, cv_results_df['mean_test_score'])
        axs[i].axvline(best_param, color='red', linestyle='--')
        axs[i].set_title(f'Score for {param}')
        axs[i].set_xlabel(param)
        axs[i].set_ylabel('Test Score')
        axs[i].tick_params(axis='x', labelrotation=45)  # Rotate labels to avoid overlap


    print("Best Parameters:", best_params)
    plt.tight_layout()
    plt.savefig(plot_path+"RandomForestClassifier_cv_results_params.png", format='png', bbox_inches='tight', dpi=300)
    plt.show()
    
    return cv_results_df
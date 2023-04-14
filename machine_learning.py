''' Author: Alex Cohen Dambrós Lopes

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Code used to apply the machine learning models
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

'''

# ============= Imports =============

import os
import time
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from scipy.stats import ks_2samp

import warnings
from warnings import simplefilter

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# ============= Warnings =============

warnings.simplefilter("ignore")
simplefilter(action='ignore', category=FutureWarning)

# ============= General Functions =============


def save_model(clf, name_model, sufix):
    
    """
    Description:
        Saves a trained model to disk.

    Parameters:
        clf : object
            Trained model object to be saved.
        model_name: str
            Model name.
        suffix: str
            "_local" or "_global"

    Return:
        None.
    """

    # Path
    path = os.path.join(os.getcwd(), f'Saved_models\\{name_model}')

    # Create the directory
    os.makedirs(path, exist_ok=True)

    # Saving
    file_name = os.path.join(path, f'{name_model+sufix}.pkl')

    if os.path.isfile(file_name):
        # If the file already exists, remove the old file
        os.remove(file_name)

    joblib.dump(clf, file_name)


def compute_ks(y_test, y_pred_proba):
    
    """
    Description:
        Kolmogorov-Smirnov value obtained from ground-truth targets (y_true) and their probabilities (y_prob_positive).

    Params:
        y_true (pd.Series): Ground-truth labels
        y_prob_positive (pd.Series): The probabilities of

    Return:
        ks (float): The KS rate.
    """

    vals = list(zip(y_test, y_pred_proba))
    positives = []
    negatives = []
    for a, b in vals:
        if a == 0:
            negatives.append(b)
        else:
            positives.append(b)

    ks = 100.0 * stats.ks_2samp(positives, negatives)[0]
    return ks

# ============= Classifier Function =============


def classifier_function(name_model, sufix, clf, parameters, cv, X_train, y_train, X_test, y_test):
    
    """
    Description: 
        This function takes as input a model name, a suffix, a model, a set of parameters to be tested by the classifier, 
        a cv object that represents cross-validation, sets of training and test data X_train, y_train, X_test, and y_test. 
        The function trains the classifier using the training set, selects the best set of parameters using cross-validation, 
        makes predictions on the test data, and calculates various model performance metrics such as precision, recall, accuracy, F1-score, area under ROC curve (AUC) and 
        the KS statistic. Finally, it saves the trained model using the save_model() function and returns the performance metrics.

    Input parameters:
        name_model: name of the model to be saved;
        suffix: suffix that will be added to the saved model file name;
        clf: a sklearn-like classifier;
        parameters: a dictionary containing the parameters that will be tested by the classifier;
        cv: an object that represents the cross-validation;
        X_train: training dataset;
        y_train: class labels corresponding to the training data;
        X_test: test dataset;
        y_test: class labels corresponding to the test data.

    Return:
        precision: calculated precision for the model;
        recall: calculated recall for the model;
        acc: calculated accuracy for the model;
        f1: F1-score calculated for the model;
        auc: area under the calculated ROC curve for the model;
        ks: KS statistic calculated for the model.
    """
    
    clf = GridSearchCV(clf, parameters, cv=cv, scoring='accuracy', n_jobs=-1)

    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Generates the positive class probabilities for the test examples
    probs = clf.predict_proba(X_test)[:, 1]

    # Accuracy
    acc = metrics.accuracy_score(y_test, y_pred) * 100
    print("\nAccuracy: %.2f" % (acc))
    # Precision
    precision = metrics.precision_score(
        y_test, y_pred, average='weighted') * 100
    print("Precision: %.2f" % (precision))
    # Recall
    recall = metrics.recall_score(y_test, y_pred, average='weighted') * 100
    print("Recall: %.2f" % (recall))
    # F1
    f1 = metrics.f1_score(y_test, y_pred, average='weighted') * 100
    print("F1: %.2f" % (f1))
    # AUC
    auc = roc_auc_score(y_test, probs)
    print("AUC: %.2f" % (auc))
    # KS
    try:
        ks = compute_ks(y_test, probs)
        print("KS: %.2f" % ks)
    except:
        ks = None
        print("Não foi possível calcular o KS.")

    # ============= Save Model =============
    save_model(clf, name_model, sufix)

    return precision, recall, acc, f1, auc, ks

# ============= Saving the Results =============


def saving_the_results(dict_result, sufix):
    
    """
    Description:
        Saves the results of a model in an Excel file.

    Parameters:
        dict_result : dict
            Results dictionary to be saved in Excel file. The dictionary keys
            will be the columns of the Excel file.
        suffix: str
            Suffix to add to the Excel file name.

    Return:
        None.
    """

    if not dict_result:
        raise ValueError("The dictionary of results is empty.")

    path = Path.cwd() / "Model_Results"
    path.mkdir(parents=True, exist_ok=True)

    file_name = f"Result_{sufix}.xlsx"
    file_path = path / file_name

    if file_path.is_file():
        file_path.unlink()

    df_results = pd.DataFrame.from_dict(dict_result, orient="index")

    with pd.ExcelWriter(file_path) as writer:
        df_results.to_excel(writer, index=True)


if __name__ == '__main__':

    # ============= Read Databases =============

    local_view = pd.read_csv(
        "Preprocessed\preprocessed_local_view.csv", sep=",")
    global_view = pd.read_csv(
        "Preprocessed\preprocessed_global_view.csv", sep=",")

    local_view.drop(["Unnamed: 0"], axis=1, inplace=True)
    global_view.drop(["Unnamed: 0"], axis=1, inplace=True)

    dropna_list = [local_view, global_view]

    for var in dropna_list:
        var.dropna(inplace=True)

    print("\n============================================================================================================")
    print("Checking base balance: ")

    targets = pd.concat([local_view[['label']].rename(columns={'label': 'target_local'}), global_view[[
                        'label']].rename(columns={'label': 'target_global'})], axis=0, ignore_index=True)
    counts = targets.apply(pd.Series.value_counts).fillna(0).astype(int)

    print(counts)

    # ============= Separating into X and y =============

    X_local = local_view.iloc[:, :-1]
    X_global = global_view.iloc[:, :-1]

    y_local = local_view['label']
    y_global = global_view['label']

    # ============= Separating into training and testing =============

    X_train_local, X_test_local, y_train_local, y_test_local = train_test_split(
        X_local, y_local, test_size=.3, random_state=42, stratify=y_local)

    X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
        X_global, y_global, test_size=.3, random_state=42, stratify=y_global)

    # ============= All models and parameters =============

    models_and_parameters = {
        'SVM': {
            'clf': SVC(probability=True, random_state=42),
            'parameters': {
                'C': [1, 3, 5, 10, 15],
                'kernel': ['linear', 'rbf'],
                'tol': [1e-3, 1e-4]
            },
        },
        'AdaBoostClassifier': {
            'clf': AdaBoostClassifier(random_state=42),
            'parameters': {
                'n_estimators': range(60, 220, 40)
            },
        },
    }

    results_local = {}
    results_global = {}
    cv = StratifiedKFold(10, random_state=42, shuffle=True)

    start_time = time.time()

    # Loops through all models within the dictionary, performs training and returns results
    for name, model in models_and_parameters.items():
        print("\n============================================================================================================")
        print("Model:", name)

        print("\n-Local")
        precision_local, recall_local, acc_local, f1_local, auc_local, ks_local = classifier_function(
            name, '_local', model['clf'], model['parameters'], cv, X_train_local, y_train_local, X_test_local, y_test_local)

        print("\n-Global")
        precision_global, recall_global, acc_global, f1_global, auc_global, ks_global = classifier_function(
            name, '_global', model['clf'], model['parameters'], cv, X_train_global, y_train_global, X_test_global, y_test_global)

        results_local[name + '_local'] = {
            'accuracy': acc_local,
            'precision': precision_local,
            'recall': recall_local,
            'f1': f1_local,
            'auc': auc_local,
            'ks': ks_local
        }
        results_global[name + '_global'] = {
            'accuracy': acc_global,
            'precision': precision_global,
            'recall': recall_global,
            'f1': f1_global,
            'auc': auc_global,
            'ks': ks_global
        }

    # marks the end of the runtime
    end_time = time.time()

    # Calculates execution time in seconds
    execution_time = end_time - start_time
    print(f"Runtime: {execution_time:.2f} seconds")

    saving_the_results(results_local, "local")
    saving_the_results(results_global, "global")

# ============= Command to load a saved model =============
# lr_model = joblib.load('./Saved_models/model.pk1')

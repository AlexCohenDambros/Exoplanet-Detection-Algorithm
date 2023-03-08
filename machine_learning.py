''' Author: Alex Cohen Dambrós Lopes 

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Code used to apply the machine learning models
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

'''

# ============= Imports =============

import numpy as np
import pandas as pd
from scipy import stats

import warnings
from warnings import simplefilter

import xgboost as xgb
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


# ============= Warnings =============

warnings.simplefilter("ignore")
simplefilter(action='ignore', category=FutureWarning)


# ============= Read Databases =============

local_view = pd.read_csv("Preprocessed\preprocessed_local_view.csv", sep=",")
global_view = pd.read_csv("Preprocessed\preprocessed_global_view.csv", sep=",")

local_view.drop(["Unnamed: 0"], axis=1, inplace=True)
global_view.drop(["Unnamed: 0"], axis=1, inplace=True)

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


# ============= General Functions =============

def compute_ks(y_test, y_pred_proba):
    """
    Description:
    Kolmogorov-Smirnov value obtained from ground-truth 
    targets (y_true) and
    their probabilities (y_prob_positive).
    Params:
    y_true (pd.Series): Ground-truth labels
    y_prob_positive (pd.Series): The probabilities of 
    TARGET=1
    Output:
    ks (float): The KS rate
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


# ============= All models and parameters =============

models_and_parameters = {
    'SVM': {
        'clf': SVC(probability=True, random_state=42),
        'parameters': {
            'C': [1, 3, 5, 10, 100, 200],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': [0.1, 0.01, 0.001, 0.0001, 'scale'],
            'tol': [1e-3, 1e-4]
        },
    },

    'XGBClassifier': {
        'clf': xgb.XGBClassifier(objective="binary:logistic", random_state=42),
        'parameters': {
            'max_depth': range(2, 10, 1),
            'n_estimators': range(60, 220, 40),
            'learning_rate': [0.1, 0.01, 0.05],
            "min_child_weight": [1, 5]
        },
    },

    'AdaBoostClassifier': {
        'clf': AdaBoostClassifier(n_estimators=100, random_state=42),
        'parameters': {
            'n_estimators': range(60, 220, 40)
        },
    },

}

# ============= Classifier Functions =============


def classifier_function(clf, parameters, cv, X_train, y_train, X_test, y_test):

    clf = GridSearchCV(clf, parameters, cv=cv, scoring='accuracy', n_jobs=-1)

    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Precision
    precision = metrics.precision_score(
        y_test, y_pred, average='weighted') * 100
    print("\nPrecision: %.2f" % (precision))
    # Recall
    recall = metrics.recall_score(y_test, y_pred, average='weighted') * 100
    print("Recall: %.2f" % (recall))
    # Accuracy
    acc = metrics.accuracy_score(y_test, y_pred) * 100
    print("Accuracy: %.2f" % (acc))
    # F1
    f1 = metrics.f1_score(y_test, y_pred, average='weighted') * 100
    print("F1: %.2f" % (f1))
    # KS
    ks = compute_ks(y_test, clf.predict_proba(X_test)[:, 1])
    print("KS: %.2f" % (ks))

    return precision, recall, acc, f1, ks


# ============= Main =============
results_local = {}
results_global = {}
cv = StratifiedKFold(10, random_state=1, shuffle=True)

# Loops through all models within the dictionary, performs training and returns results
for name, model in models_and_parameters.items():

    precision_local, recall_local, acc_local, f1_local, ks_local = classifier_function(
        model['clf'], model['parameters'], cv, X_train_local, y_train_local, X_test_local, y_test_local)
    precision_global, recall_global, acc_global, f1_global, ks_global = classifier_function(
        model['clf'], model['parameters'], cv, X_train_global, y_train_global, X_test_global, y_test_global)

    results_local[name] = {
        'precision': precision_local,
        'recall': recall_local,
        'accuracy': acc_local,
        'f1': f1_local,
        'ks': ks_local
    }
    results_global[name] = {
        'precision': precision_global,
        'recall': recall_global,
        'accuracy': acc_global,
        'f1': f1_global,
        'ks': ks_global
    }

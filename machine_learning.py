''' Author: Alex Cohen Dambrós Lopes 

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Code used to apply the machine learning models
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

'''

# ============= Imports =============
import pandas as pd
import numpy as np
from scipy import stats

import warnings
from warnings import simplefilter

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics 


# ============= Warnings =============
warnings.simplefilter("ignore")
simplefilter(action='ignore', category=FutureWarning)


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

# ============= All parameters =============
parameters_svm = [
    {'C': [1, 5, 10, 100, 200], 'kernel': ['linear'],
     'C': [1, 5, 10, 50, 100, 200], 'kernel': ['poly'],
     'C': [1, 5, 10, 100, 150, 500], 'gamma': [0.1, 0.01, 0.001, 0.0001, 'scale'], 'kernel':['rbf']
     },
]

# ============= Classifier Functions =============
def classifier_svm(parameters, X_train, y_train, X_test, y_test):
    
    clf = SVC(probability=True, random_state=42)
    clf = GridSearchCV(clf, parameters, scoring='accuracy', n_jobs=-1)
    
    clf = clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    # Precision
    print("\nPrecision: %.2f" % (metrics.precision_score(y_test, y_pred, average='weighted') * 100))
    # Recall
    print("Recall: %.2f" % (metrics.recall_score(y_test, y_pred, average='weighted') * 100))
    # Accuracy
    print("Accuracy: %.2f" % (metrics.accuracy_score(y_test, y_pred) * 100))
    # F1
    print("F1: %.2f" % (metrics.f1_score(y_test, y_pred, average='weighted') * 100))
    # KS
    print("KS: %.2f" % (compute_ks(y_test, clf.predict_proba(X_test)[:,1])))
    
    
    
    


''' Author: Alex Cohen Dambrós Lopes

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Code used to apply the machine learning models
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

'''

# ============= Imports =============

import time
import numpy as np
import pandas as pd
from scipy import stats

import warnings
from warnings import simplefilter


import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score

from General_Functions import save_models_results

np.random.seed(42)

# ============= Warnings =============

warnings.simplefilter("ignore")
simplefilter(action='ignore', category=FutureWarning)

# ============= General Functions =============

def pre_processing_onehotencoder(df_encoder):
  
    try: 
        X = df_encoder.drop(['label'], axis = 1)
    except:
        X = df_encoder.copy()
 
    numerical_columns = X.select_dtypes(include=['Int64', 'int32', 'float32', 'float64']).columns
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
    nan_columns = [i[0] for i in list(X[numerical_columns].isna().all().items()) if i[1]]

    remove_columns = nan_columns
    
    numerical_columns = list(set(numerical_columns)-set(remove_columns))
    categorical_columns = list(set(categorical_columns)-set(remove_columns))

    # Features dataset
    features = list(numerical_columns) + list(categorical_columns)
    
    numeric_imputer_transformer = Pipeline(steps=[
        ('Median Imputer', SimpleImputer(strategy='median', add_indicator=False)),
        ('Standard Scaler', StandardScaler())
    ])

    categorical_modeimputer_transformer = Pipeline(steps=[
        ('ModeImputer', SimpleImputer(strategy='constant', fill_value = 'Missing')),
        ('OneHotEncoder', OneHotEncoder(sparse=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
        ('Numerical - MeanImputer StandardScaler', numeric_imputer_transformer, numerical_columns),
        ('Categorical - ModeImputer OneHotEncoder', categorical_modeimputer_transformer, categorical_columns),
        ('AllNAN - Drop', 'drop', nan_columns)
    ])

    features_update = features.copy()
    return preprocessor.fit(df_encoder[features_update])


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

# ============= LSTM =============

def model_LSTM(vector_size):
    
    # Creating the LSTM (Long Short Term Memory) network
    
    model = Sequential()
    # model.add(Embedding(vocab_size, vector_size))
    # model.add(tf.keras.layers.LSTM(embedding_dim, dropout = 0.25 , return_sequences=True))
    # model.add(tf.keras.layers.LSTM(embedding_dim, dropout = 0.25))
    # model.add(tf.keras.layers.Dense(64, activation='relu'))
    # model.add(tf.keras.layers.Dense(2, activation='softmax'))
    # model.add(Dense(units=1)) 
    
    # Compiling the LSTM
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # LSTM training
    model.fit(X_train_local, y_train_local, epochs=20, validation_data=(X_test_local, y_test_local), verbose=2)

    model.summary()

# ============= Classifier Function =============

def run_classifier_models(name_model, sufix, clf, parameters, cv, X_train, y_train, X_test, y_test, preprocessor):
    
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
        y_pred: Is a variable that is used to store the predictions (predictions) generated by the model.
    """
    
    clf = GridSearchCV(clf, parameters, cv=cv, scoring='accuracy', n_jobs=-1)
    
    # Smote balancing
    # smote = SMOTE()  # Create a SMOTE instance
    # X_resampled, y_resampled = smote.fit_resample(X_train, y_train)  # Apply SMOTE to data
    
    clf = make_pipeline(preprocessor, clf).fit(X_train, y_train)
    
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
 
    save_models_results.save_model(clf, name_model, sufix)

    return precision, recall, acc, f1, auc, ks, y_pred


def classifier_function(dict_model_parameters):
    
    """
    Description: 
       This function is used to train the classification models passed as parameters.

    Input parameters:
       dict_model_parameters: Dictionary containing the models and parameters for training.

    Return:
       None.
    """
    
    if not isinstance(dict_model_parameters, dict):
        raise TypeError("Parameter must be a dictionary.")
    
    if len(dict_model_parameters) == 0:
        raise ValueError("Dictionary of parameters is empty.")
    
    results_local = {}
    results_global = {}
    cv = StratifiedKFold(10, random_state=42, shuffle=True)

    start_time = time.time()

    # Loops through all models within the dictionary, performs training and returns results
    for name, model in dict_model_parameters.items():
        print("\n============================================================================================================")
        print("Model:", name)

        print("\n-Local")
        precision_local, recall_local, acc_local, f1_local, auc_local, ks_local, y_pred_local = run_classifier_models(
            name, '_local', model['clf'], model['parameters'], cv, X_train_local, y_train_local, X_test_local, y_test_local, preprocessor_local)

        print("\n-Global")
        precision_global, recall_global, acc_global, f1_global, auc_global, ks_global, y_pred_global = run_classifier_models(
            name, '_global', model['clf'], model['parameters'], cv, X_train_global, y_train_global, X_test_global, y_test_global, preprocessor_global)

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
    print(f"\nRuntime: {execution_time:.2f} seconds")

    save_models_results.saving_the_results(results_local, y_test_local, y_pred_local)
    save_models_results.saving_the_results(results_global, y_test_global, y_pred_global)

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
    
    # ============= Transform target column values ​​into 0 and 1 =============
    
    target_map = {'CONFIRMED': 0, 'FALSE POSITIVE': 1}
    local_view['label'] = local_view['label'].map(target_map)
    global_view['label'] = global_view['label'].map(target_map)
    
    # ============= preprocessor data =============
    preprocessor_local = pre_processing_onehotencoder(local_view)
    preprocessor_global = pre_processing_onehotencoder(global_view)
    
    # ============= Separating into X and y =============

    X_local = local_view.iloc[:, :-1]
    X_global = global_view.iloc[:, :-1]

    y_local = local_view['label']
    y_global = global_view['label']

    # ============= Separating into training and testing =============

    X_train_local, X_test_local, y_train_local, y_test_local = train_test_split(
        X_local, y_local, test_size= 0.3, random_state=42, stratify=y_local)

    X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
        X_global, y_global, test_size= 0.3, random_state=42, stratify=y_global)
    
    
    # ============= All models and parameters of classification models =============

    models_and_parameters_C = {
        'AdaBoostClassifier': {
            'clf': AdaBoostClassifier(random_state=42),
            'parameters': {
                'n_estimators': range(60, 220, 40)
            },
        },
        'XGBClassifier': {
            'clf': xgb.XGBClassifier(objective = "binary:logistic", random_state=42),
            'parameters': {
                'min_child_weight': [1, 5, 10],
                'gamma': [0.5, 1, 1.5, 2, 5],
                'max_depth': [3, 4, 5]
            }
        },
        # 'SVM': {
        #     'clf': SVC(probability=True, random_state=42),
        #     'parameters': {
        #         'C': [1, 3, 5, 10, 15],
        #         'kernel': ['linear', 'rbf'],
        #         'tol': [1e-3, 1e-4]
        #     },
        # },
        # 'MLPClassifier': {
        #     'clf': MLPClassifier(random_state=42),
        #     'parameters': {
        #         'solver': ['lbfgs', 'sgd', 'adam'], 
        #         'max_iter': [1000,1300,1500,2000], 
        #         'alpha': 10.0 ** -np.arange(1, 10), 
        #         'hidden_layer_sizes':np.arange(10, 15),
        #         'tol': [1e-3, 1e-4]
        #     },
        # },
    }
    
    # ============= Running classifier models =============
    
    classifier_function(models_and_parameters_C)
    
    # ============= Running LSTM =============
    
    
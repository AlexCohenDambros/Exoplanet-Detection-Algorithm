''' Author: Alex Cohen Dambrós Lopes

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Code used to apply the machine learning models
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

'''

# ============= Imports =============
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

import time
import warnings
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning

import xgboost as xgb

from scipy import stats

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score

from General_Functions import save_models_results


np.random.seed(42)

# ============= Warnings =============

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)
simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)

# ============= General Functions =============

def preProcessingOneHotEncoder(df_encoder):
  
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


def computeKs(y_test, y_pred_proba):
    
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

def plotTrainHistoryLSTM(history, title):
    
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()
  
def createTimeSteps(length):
    
  time_steps = []
  
  for i in range(-length, 0, 1):
    time_steps.append(i)
    
  return time_steps
  
def plotPredsLSTM(plot_data, delta=0):
    
    labels = ['History', 'True Future','LSTM Prediction']
    marker = ['.-', 'gX', 'ro' , 'bo']
    time_steps = createTimeSteps(plot_data[0].shape[0])
    
    future = delta

    plt.title('Predictions')
    for i, x in enumerate(plot_data):
      if i:
        plt.plot(future, plot_data[i], marker[i], markersize=10,
                label=labels[i])
      else:
        plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt

# ============= LSTM =============

def methodLSTM(X_train, y_train, X_test, y_test):
    
    # Defining the datasets 
    
    BATCH_SIZE = 256
    BUFFER_SIZE = 10000

    train_univariate = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_univariate = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
    
    # Creating the architecture
    simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=(X_train.shape[1], y_train.shape[2])),
    tf.keras.layers.Dense(1)
    ])

    simple_lstm_model.compile(optimizer='adam', loss='mae')

    simple_lstm_model.summary()
    
    
    # Training LSTM
    
    EPOCHS = 10
     
    lstm_log = simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                      validation_data=val_univariate, validation_steps=50)
    
    plotTrainHistoryLSTM(lstm_log, 'LSTM Training and validation loss')
    
    future = 5
    
    for x, y in val_univariate.take(5):
        plot = plotPredsLSTM([x[0].numpy(), y[0].numpy(), simple_lstm_model.predict(x)[0]], future)
        plot.show()
    
    # Calculates the accuracy  

    err_lstm=0

    for x, y in val_univariate.take(10):
        err_lstm += abs(y[0].numpy() - simple_lstm_model.predict(x)[0])

    err_lstm = err_lstm/10
    
    print(err_lstm)

# ============= Classifier Function =============

def runClassifierModels(name_model, sufix, clf, parameters, cv, X_train, y_train, X_test, y_test, preprocessor):
    
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
        ks = computeKs(y_test, probs)
        print("KS: %.2f" % ks)
    except:
        ks = None
        print("Não foi possível calcular o KS.")
 
    save_models_results.save_model(clf, name_model, sufix)

    return precision, recall, acc, f1, auc, ks, y_pred


def classifiers(dict_model_parameters):
    
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
        precision_local, recall_local, acc_local, f1_local, auc_local, ks_local, y_pred_local = runClassifierModels(
            name, '_local', model['clf'], model['parameters'], cv, X_train_local, y_train_local, X_test_local, y_test_local, preprocessor_local)

        print("\n-Global")
        precision_global, recall_global, acc_global, f1_global, auc_global, ks_global, y_pred_global = runClassifierModels(
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
    preprocessor_local = preProcessingOneHotEncoder(local_view)
    preprocessor_global = preProcessingOneHotEncoder(global_view)
    
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
    
    
    # Smote balancing
    smote = SMOTE()  # Create a SMOTE instance
    X_train_local, y_train_local = smote.fit_resample(X_train_local, y_train_local)  # Apply SMOTE to data local
    X_train_global, y_train_global = smote.fit_resample(X_train_global, y_train_global)  # Apply SMOTE to data global
    
    
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
        'SVM': {
            'clf': SVC(probability=True, random_state=42),
            'parameters': {
                'C': [1, 3, 5, 10, 15],
                'kernel': ['linear', 'rbf'],
                'tol': [1e-3, 1e-4]
            },
        },
        'MLPClassifier': {
            'clf': MLPClassifier(random_state=42),
            'parameters': {
                'solver': ['sgd', 'adam'], 
                'max_iter': [1000, 1300, 1500, 2000], 
                'alpha': 10.0 ** -np.arange(1, 10), 
                'hidden_layer_sizes':np.arange(10, 15),
                'tol': [1e-3, 1e-4]
            },
        },
    }
    
    # ============= Running classifier models =============
    
    # classifiers(models_and_parameters_C)
    
    # ============= Running LSTM =============
    
    '''Tamanho da Janela do Historico'''
    univariate_past_history = 30  #30 observacoes anteriores
    future = univariate_future_target = 5  #a proxima observação

    # x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
    #                                         univariate_past_history,
    #                                         univariate_future_target)
    # x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
    #                                     univariate_past_history,
    #                                     univariate_future_target)
    
    
    # Test LSTM data local
    methodLSTM(X_train_local, y_train_local, X_test_local, y_test_local)
    
    # Test LSTM data global
    methodLSTM(X_train_global, y_train_global, X_test_global, y_test_global)
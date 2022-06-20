import time

import warnings
from warnings import simplefilter

import pandas as pd
import numpy as np

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from pyts.classification import TimeSeriesForest

from lightkurve import search_lightcurve


warnings.filterwarnings("ignore", category=RuntimeWarning)
simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

np.random.seed(42)

dataframe = pd.read_csv("datasKepler.csv")
print('Total curves: %d \n' % (dataframe.shape[0]))

dataframe = dataframe.dropna()
dataframe = dataframe[dataframe.koi_disposition != 'CANDIDATE']
dataframe = dataframe.reset_index(drop=True)

print('False Positive: %d | Confirmed: %d\n' % ((dataframe.koi_disposition ==
      'FALSE POSITIVE').sum(), (dataframe.koi_disposition == 'CONFIRMED').sum()))

percentageNegativePositive = (
    (dataframe.koi_disposition == 'FALSE POSITIVE').sum()*100) / dataframe.shape[0]
print('False positive: %.2f %% Confirmed: %.2f %% \n' %
      (percentageNegativePositive, 100 - percentageNegativePositive))


dataframe["koi_disposition"] = dataframe["koi_disposition"].apply(
    lambda x: 2 if x == 'CONFIRMED' else 1 if x == 'CANDIDATE' else 0)

print(dataframe.head())


# ======== Parameters of Machine Learning functions ========

parametersSVM = [
    {
        'C': [1, 5, 10, 50, 100, 150, 500],
        'gamma': [0.1, 0.01, 0.001, 0.0001, 'scale'],
        'kernel':['rbf', 'poly', 'linear', 'sigmoid', 'precomputed'],
        'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        'random_state': [42]
    }
]

parametersTSF = [
    {
        'n_estimators': range(100, 1000, 100)
    }    
]


# ======== Functions ========

def supportVectorMachine(parameters, folds):

    model = SVC(probability=True, random_state=42)

    model = GridSearchCV(model, parameters,
                          scoring='accuracy', cv=folds,  n_jobs=-1)

    model = model.fit(X_train, y_train)

    best_parameters = model.best_params_

    # test using test base
    predicted = model.predict(X_test)

    result = model_selection.cross_val_score(
        model, X_train, y_train, cv=folds, n_jobs=-1)

    # calculates accuracy based on test
    score = result.mean()

    # calculate the confusion matrix
    matrix = confusion_matrix(y_test, predicted)

    print("\nFinished!")
    
    return score, matrix, best_parameters


def timeSeriesForest(parameters, folds):

    model = TimeSeriesForest(random_state=42)

    model = GridSearchCV(model, parameters,
                          scoring='accuracy', cv=folds,  n_jobs=-1)

    model = model.fit(X_train, y_train)

    best_parameters = model.best_params_

    # test using test base
    predicted = model.predict(X_test)

    result = model_selection.cross_val_score(
        model, X_train, y_train, cv=folds, n_jobs=-1)

    # calculates accuracy based on test
    score = result.mean()

    # calculate the confusion matrix
    matrix = confusion_matrix(y_test, predicted)

    print("\nFinished!")

    return score, matrix, best_parameters

# ================================= Main =================================

folds = 10  # Number of folds used in cross-validation

start_time = time.time()

if __name__ == '__main__':
    for line, column in dataframe.iterrows():
        period, t0, duration = column[4], column[5], column[6]

        curve = search_lightcurve(
            str(column[1]), author='Kepler', cadence="long")

        if (curve != None):

            allCurve = curve.download_all()

            try:
                allCurveClean = allCurve.normalize().remove_nans().remove_outliers()
            except:
                allCurveClean = allCurve.stitch()

        break

    tempo = time.time() - start_time

    print('\nTime used for learning: %f seconds\n' % tempo)
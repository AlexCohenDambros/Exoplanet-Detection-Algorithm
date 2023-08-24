''' Author: Alex Cohen Dambrós Lopes

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Code used to apply the machine learning models
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

'''

# ============= Imports =============
from __future__ import absolute_import, division, print_function, unicode_literals
from GeneralFunctions import save_models_results
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from scipy import stats
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from warnings import simplefilter
import warnings
import time

import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


np.random.seed(42)

# ============= Warnings =============

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)
simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)

# ============= General Functions =============


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


def plot_train_history_LSTM(history, title):

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


def create_time_steps(length):

    time_steps = []

    for i in range(-length, 0, 1):
        time_steps.append(i)

    return time_steps


def plot_preds_LSTM(plot_data, delta=0):

    labels = ['History', 'True Future', 'LSTM Prediction']
    marker = ['.-', 'gX', 'ro', 'bo']
    time_steps = create_time_steps(plot_data[0].shape[0])

    future = delta

    plt.title('Predictions')
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(
            ), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt


def univariate_data(dataset, start_index, end_index, history_size, target_size):

    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)

# ============= LSTM =============

def method_LSTM(x_train_uni, y_train_uni, x_val_uni, y_val_uni):

    # Defining the datasets

    BATCH_SIZE = 256 # quantidade de amostra para cada passagem no algortimo de treinamento
    BUFFER_SIZE = 10000 # Gerenciar melhor a memória

    train_univariate = tf.data.Dataset.from_tensor_slices(
        (x_train_uni, y_train_uni))
    train_univariate = train_univariate.cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
    val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

    # Creating the architecture
    simple_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(8, input_shape=(
            x_train_uni.shape[1], y_train_uni.shape[2])),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    simple_lstm_model.compile(optimizer='adam', loss='binary_crossentropy')

    simple_lstm_model.summary()

    # Training LSTM

    EPOCHS = 10

    lstm_log = simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                                     validation_data=val_univariate, validation_steps=50)

    plot_train_history_LSTM(lstm_log, 'LSTM Training and validation loss')

    future = 1

    for x, y in val_univariate.take(5):
        plot = plot_preds_LSTM(
            [x[0].numpy(), y[0].numpy(), simple_lstm_model.predict(x)[0]], future)
        plot.show()

    # Calculates the accuracy

    err_lstm = 0

    for x, y in val_univariate.take(10):
        err_lstm += abs(y[0].numpy() - simple_lstm_model.predict(x)[0])

    err_lstm = err_lstm/10

    print(err_lstm)

# ============= Classifier Function =============


def run_classifier_models(name_model, sufix, clf, parameters, cv, X_train, y_train, X_test, y_test):
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

    save_models_results.save_model(clf, name_model, sufix)

    return precision, recall, acc, f1, auc, ks, y_pred


def defining_classifiers(dict_model_parameters, X_train, y_train, X_test, y_test, data_vision):
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

    results = {}
    cv = StratifiedKFold(10, random_state=42, shuffle=True)

    start_time = time.time()

    # Loops through all models within the dictionary, performs training and returns results
    for name, model in dict_model_parameters.items():
        print("\n============================================================================================================")
        print("Model:", name)

        print(f"\n- Running models on data in a {data_vision} vision")
        data_vision = "_" + data_vision

        precision, recall, acc, f1, auc, ks, y_pred = run_classifier_models(
            name, data_vision, model['clf'], model['parameters'], cv, X_train, y_train, X_test, y_test)

        results[name + data_vision] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'ks': ks
        }

    # marks the end of the runtime
    end_time = time.time()

    # Calculates execution time in seconds
    execution_time = end_time - start_time
    print(f"\nRuntime: {execution_time:.2f} seconds")

    save_models_results.saving_the_results(results, y_test, y_pred)

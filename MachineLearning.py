import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from matplotlib import widgets
import numpy as np
from sklearn import model_selection, naive_bayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import time
from lightkurve import search_lightcurve 
import warnings

np.random.seed(42)
warnings.filterwarnings('ignore')

dataframe = pd.read_csv("datasKepler.csv")
print('Total de curvas: %d \n' % (dataframe.shape[0]))

dataframe = dataframe.dropna()
dataframe = dataframe[dataframe.koi_disposition != 'CANDIDATE']
dataframe = dataframe.reset_index(drop=True)

print('Falsos positivos: %d | Confirmados: %d\n' % ((dataframe.koi_disposition == 'FALSE POSITIVE').sum(), (dataframe.koi_disposition == 'CONFIRMED').sum()))

percentageNegativePositive = ((dataframe.koi_disposition == 'FALSE POSITIVE').sum()*100) / dataframe.shape[0]
print('Falsos positivos: %.2f %% Confirmados: %.2f %% \n' % (percentageNegativePositive, 100 - percentageNegativePositive))


dataframe["koi_disposition"] = dataframe["koi_disposition"].apply(lambda x: 2 if x == 'CONFIRMED' else 1 if x == 'CANDIDATE' else 0 )

print(dataframe.head())


# ======== Parametros das funcoes de Machine Learning ========

parametrosKNN = [
    {'n_neighbors': list(range(0, 30)),
     'weights': ['uniform', 'distance'],
     'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
     'p': [1, 2],
     'leaf_size': list(range(5, 60, 5))},
]

parametrosDecisionTrees = [
    {'max_depth': list(range(3, 60, 3)),
     'min_samples_split': list(range(0, 21, 3)),
     'criterion': ['entropy', 'gini', 'log_loss'],
     'splitter':['best', 'random']},
]

parametrosNaiveBayes = [
    {
        'var_smoothing': [1e-9, 1e-8, 1e-5, 1e-4, 1e-3]
    }
]


# ======== Funcoes de Machine Learning ========

# ================================= Função KNN =================================
def kNN(folds, parameters):

    modelo = KNeighborsClassifier()

    resultado = model_selection.cross_val_score(modelo, X, y, cv = folds)

    gs = GridSearchCV(modelo, parameters, scoring='r2', cv = folds, n_jobs=5)

    gs = gs.fit(X, y)

    print(gs.best_params_)

    # Definindo a técnica a ser utilizada
    modelo = gs.best_estimator_

    resultado = model_selection.cross_val_score(modelo, X, y, cv = folds)

    # Mostrando a acurácia média e desvio padrão.
    print("\nCross Validation Results %d folds:" % folds)
    print("Mean Accuracy: %.5f" % resultado.mean())
    print("Mean Std: %.5f" % resultado.std())

    # Calculando a predição para exemplo de teste
    y_pred = model_selection.cross_val_predict(modelo, X, y, cv = folds)

    # Calculando para cada instância de teste a probabilidade de cada classe
    predicted_proba = model_selection.cross_val_predict(
        modelo, X, y, cv = folds, method='predict_proba')

    # Calculando a precisão na base de teste
    precision = precision_score(y, y_pred, average='weighted')
    print("Precision = %.3f " % precision)

    # Calculando a revocação na base de teste
    recall = recall_score(y, y_pred, average='weighted')
    print("Recall = %.3f " % recall)

    # Calculando f1 na base de teste
    f1 = f1_score(y, y_pred, average='weighted')
    print("F1 = %.3f " % f1)

    # Exemplo mostrando o resultado previsto para a primeira instância de teste
    print(
        "Primeira instância na base de teste foi considerada como da classe: %d" % y_pred[0])

    # Exemplo abaixo mostrando para a primeira instância de teste a probabilidade de cada classe
    print("Probabilidade de cada classe para a primeira instância: ",
          predicted_proba[0])

    # Calculando a matriz de confusão
    print("Matriz de Confusão:")
    matrix = confusion_matrix(y, y_pred)
    print(matrix)

# ================================= Função de Árvore de Decisão =================================
def decisionTrees(folds, parameters):
    modelo = DecisionTreeClassifier(random_state=42)

    resultado = model_selection.cross_val_score(modelo, X, y, cv = folds)

    gs = GridSearchCV(modelo, parameters, scoring='r2', cv = folds, n_jobs=5)
    
    gs = gs.fit(X, y)

    print(gs.best_params_)

    # Definindo a técnica a ser utilizada
    modelo = gs.best_estimator_

    resultado = model_selection.cross_val_score(modelo, X, y, cv = folds)

    # Mostrando a acurácia média e desvio padrão.
    print("\nCross Validation Results %d folds:" % folds)
    print("Mean Accuracy: %.5f" % resultado.mean())
    print("Mean Std: %.5f" % resultado.std())

    # Calculando a predição para exemplo de teste
    y_pred = model_selection.cross_val_predict(modelo, X, y, cv = folds)

    # Calculando para cada instância de teste a probabilidade de cada classe
    predicted_proba = model_selection.cross_val_predict(
        modelo, X, y, cv = folds, method='predict_proba')

    # Calculando a precisão na base de teste
    precision = precision_score(y, y_pred, average='weighted')
    print("Precision = %.3f " % precision)

    # Calculando a revocação na base de teste
    recall = recall_score(y, y_pred, average='weighted')
    print("Recall = %.3f " % recall)

    # Calculando f1 na base de teste
    f1 = f1_score(y, y_pred, average='weighted')
    print("F1 = %.3f " % f1)

    # Exemplo mostrando o resultado previsto para a primeira instância de teste
    print(
        "Primeira instância na base de teste foi considerada como da classe: %d" % y_pred[0])

    # Exemplo abaixo mostrando para a primeira instância de teste a probabilidade de cada classe
    print("Probabilidade de cada classe para a primeira instância: ",
          predicted_proba[0])

    # Calculando a matriz de confusão
    print("Matriz de Confusão:")
    matrix = confusion_matrix(y, y_pred)
    print(matrix)

# ================================= Função Naive Bayes =================================
def naiveBayes(folds, parameters):
    modelo = GaussianNB()

    resultado = model_selection.cross_val_score(modelo, X, y, cv = folds)

    gs = GridSearchCV(modelo, parameters, scoring='r2', cv = folds, n_jobs=5)

    gs = gs.fit(X, y)

    print(gs.best_params_)

    # Definindo a técnica a ser utilizada
    modelo = gs.best_estimator_

    resultado = model_selection.cross_val_score(modelo, X, y, cv = folds)

    # Mostrando a acurácia média e desvio padrão.
    print("\nCross Validation Results %d folds:" % folds)
    print("Mean Accuracy: %.5f" % resultado.mean())
    print("Mean Std: %.5f" % resultado.std())

    # Calculando a predição para exemplo de teste
    y_pred = model_selection.cross_val_predict(modelo, X, y, cv = folds)

    # Calculando para cada instância de teste a probabilidade de cada classe
    predicted_proba = model_selection.cross_val_predict(
        modelo, X, y, cv = folds, method='predict_proba')

    # Calculando a precisão na base de teste
    precision = precision_score(y, y_pred, average='weighted')
    print("Precision = %.3f " % precision)

    # Calculando a revocação na base de teste
    recall = recall_score(y, y_pred, average='weighted')
    print("Recall = %.3f " % recall)

    # Calculando f1 na base de teste
    f1 = f1_score(y, y_pred, average='weighted')
    print("F1 = %.3f " % f1)

    # Exemplo mostrando o resultado previsto para a primeira instância de teste
    print(
        "Primeira instância na base de teste foi considerada como da classe: %d" % y_pred[0])

    # Exemplo abaixo mostrando para a primeira instância de teste a probabilidade de cada classe
    print("Probabilidade de cada classe para a primeira instância: ",
          predicted_proba[0])

    # Calculando a matriz de confusão
    print("Matriz de Confusão:")
    matrix = confusion_matrix(y, y_pred)
    print(matrix)
            
folds = 10 # Numero de Folds utilizados na validação cruzada 

curvas_locais = []
labels_locais = []
curvas_globais = []
labels_globais = []
start_time = time.time()


if __name__ == '__main__':
    for line, column in dataframe.iterrows():
        period, t0, duration = column[4], column[5], column[6]
        
        
        curve = search_lightcurve(str(column[1]), author='Kepler', cadence = "long")

        
        if (curve != None):
            
            allCurve = curve.download_all()
        
            try:
                allCurveClean = allCurve.normalize().remove_nans().remove_outliers()
            except:
                allCurveClean = allCurve.stitch()

        
        break
        

    tempo = time.time() - start_time   

    print('\nTempo utilizado para aprendizado: %f seconds\n' % tempo)
    

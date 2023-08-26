from deslib.dcs import OLA
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Carregar o conjunto de dados Iris como exemplo
iris = load_iris()
X = iris.data
y = iris.target

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar diferentes classificadores
clf_lr = LogisticRegression()
clf_rf = RandomForestClassifier()
clf_svm = SVC(probability=True)

clf_lr.fit(X_train, y_train)
clf_rf.fit(X_train, y_train)
clf_svm.fit(X_train, y_train)

# Criar uma lista de classificadores treinados
classifiers = [clf_lr, clf_rf, clf_svm]

# Criar uma instância da técnica de seleção dinâmica OLA
ola = OLA(classifiers)

# Ajustar o OLA aos dados de treinamento
ola.fit(X_train, y_train)

# Prever as classes usando a técnica de seleção dinâmica OLA
selected_predictions = ola.predict(X_test)

# Calcular a acurácia das previsões
acc_ola = accuracy_score(y_test, selected_predictions)
print("Acurácia usando OLA com diferentes classificadores:", acc_ola)

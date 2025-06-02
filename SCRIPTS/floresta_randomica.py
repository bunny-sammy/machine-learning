# Importações necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import os

# Variáveis globais
CATEGORICAS = ['local', 'tvcabo', 'debaut', 'cancel']

# Definições de caminho
base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
db_dir = os.path.join(base_dir, "DADOS")
os.makedirs(db_dir, exist_ok=True)

file_path = os.path.join(db_dir, 'p33.xlsx')
df = pd.read_excel(file_path, 'TECAL')

# Exibir as primeiras linhas do dataframe
print(df.head())

# Garantir que colunas categóricas sejam strings
for col in CATEGORICAS:
    df[col] = df[col].astype(str)

# Codificação das variáveis categóricas
label_encoders = {}
for col in CATEGORICAS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separar variáveis independentes e dependentes
X = df.drop(['id', 'cancel'], axis=1)
y = df['cancel']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar e treinar o modelo de Floresta Randômica
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Previsão
y_pred = clf.predict(X_test)

# Avaliação do modelo
print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

# Visualizar uma única árvore da floresta
plt.figure(figsize=(12, 8))
plot_tree(clf.estimators_[0], filled=True, feature_names=X.columns, class_names=label_encoders['cancel'].classes_)
plt.title("Árvore da Floresta Randômica (Primeira Árvore)")
plt.show()

# Plotar importância das features
plt.figure(figsize=(10, 6))
feature_importance = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
feature_importance.plot(kind='bar')
plt.title("Importância das Features na Floresta Randômica")
plt.xlabel("Features")
plt.ylabel("Importância")
plt.show()
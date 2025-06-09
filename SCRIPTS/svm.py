# Importações necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Variáveis globais
CATEGORICAS = ['local', 'tvcabo', 'debaut', 'cancel']

# Definições de caminho
base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
db_dir = os.path.join(base_dir, "DADOS")
os.makedirs(db_dir, exist_ok=True)

file_path = os.path.join(db_dir, 'p33_clean.xlsx')
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
X = df.drop(['cancel'], axis=1)
y = df['cancel']

# Escalonar as features (essencial para SVM)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar e treinar o modelo SVM
clf = SVC(kernel='rbf', C=1.0, max_iter=100, random_state=42)
clf.fit(X_train, y_train)

# Previsão
y_pred = clf.predict(X_test)

# Avaliação do modelo
print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

# 1. Heatmap da Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoders['cancel'].classes_, yticklabels=label_encoders['cancel'].classes_)
plt.title('Matriz de Confusão - SVM')
plt.xlabel('Previsão')
plt.ylabel('Verdadeiro')
plt.show()

# 2. Gráfico de Dispersão com PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)
plt.figure(figsize=(8, 6))
for status in np.unique(y_test):
    idx = y_test == status
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=label_encoders['cancel'].classes_[status], alpha=0.6)
plt.title('Dispersão das Classes (PCA) - SVM')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.show()

# 3. Curva de Acurácia vs. Parâmetro C
c_values = [0.1, 1, 10, 100]
scores = []
for c in c_values:
    svm = SVC(kernel='rbf', C=c, random_state=42)
    svm.fit(X_train, y_train)
    scores.append(accuracy_score(y_test, svm.predict(X_test)))
plt.figure(figsize=(8, 6))
plt.plot(c_values, scores, marker='o')
plt.xscale('log')
plt.title('Acurácia vs. Parâmetro C - SVM')
plt.xlabel('C (log scale)')
plt.ylabel('Acurácia')
plt.grid(True)
plt.show()
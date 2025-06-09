# Importações necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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

# Verificar valores nulos
# print("Valores nulos por coluna:\n", df.isnull().sum())

# Tratamento dos valores nulos
# df['IDADE'] = df['IDADE'].fillna(df['IDADE'].mean())
# df['RESID'] = df['RESID'].fillna('DESCONHECIDO')
# df['TMPRSD'] = df['TMPRSD'].fillna(df['TMPRSD'].mean())
# df['INSTRU'] = df['INSTRU'].fillna('NAO INFORMADO')

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

# Escalonar as features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar e treinar o modelo KNN
clf = KNeighborsClassifier(n_neighbors=69)
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
plt.title('Matriz de Confusão - KNN')
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
plt.title('Dispersão das Classes (PCA)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.show()

# 3. Curva de Acurácia vs. K
k_range = range(1, 21)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(accuracy_score(y_test, knn.predict(X_test)))
plt.figure(figsize=(8, 6))
plt.plot(k_range, scores, marker='o')
plt.title('Acurácia vs. Número de Vizinhos (K)')
plt.xlabel('K')
plt.ylabel('Acurácia')
plt.grid(True)
plt.show()
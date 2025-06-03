# neural_network.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Variáveis globais
CATEGORICAS = ['local', 'tvcabo', 'debaut', 'cancel']

# =============================================
# Definições de caminho
# =============================================
base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
db_dir = os.path.join(base_dir, "DADOS")
os.makedirs(db_dir, exist_ok=True)

# =============================================
# Carregamento da base
# =============================================
file_path = os.path.join(db_dir, 'p33.xlsx')
df = pd.read_excel(file_path, 'TECAL')

# Garantir que colunas categóricas sejam strings
for col in CATEGORICAS:
    df[col] = df[col].astype(str)

# Codificação das variáveis categóricas
label_encoders = {}
for col in CATEGORICAS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# =============================================
# Separação das variáveis independentes e dependente
# =============================================
X = df.drop(['id', 'cancel'], axis=1)
y = df['cancel']

# Normalização
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# =============================================
# Criação e treino da rede neural
# =============================================
mlp = MLPClassifier(
    hidden_layer_sizes=(int(input("Neurons:")),),
    activation='relu', 
    solver='adam',                
    max_iter=300, 
    random_state=42)

mlp.fit(X_train, y_train)

# =============================================
# Avaliação do modelo
# =============================================
y_pred = mlp.predict(X_test)

print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nAcurácia:", accuracy_score(y_test, y_pred))

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# =============================================
# Visualização - Heatmap da Matriz de Confusão
# =============================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=label_encoders['cancel'].classes_,
            yticklabels=label_encoders['cancel'].classes_)
plt.title('Matriz de Confusão - Rede Neural')
plt.xlabel('Previsão')
plt.ylabel('Verdadeiro')
plt.tight_layout()
plt.show()
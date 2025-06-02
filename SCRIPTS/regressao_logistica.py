import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Definindo diretórios
# Variáveis globais
CATEGORICAS = ['local', 'tvcabo', 'debaut', 'cancel']

# 1. Definições de caminho
base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
db_dir = os.path.join(base_dir, "DADOS")
os.makedirs(db_dir, exist_ok=True)

file_path = os.path.join(db_dir, 'p33.xlsx')
df = pd.read_excel(file_path, 'TECAL')

# 2. Remove linhas com valores ausentes em colunas relevantes
# df = df[['IDADE', 'TMPRSD', 'RNDTOT', 'STATUS']].dropna()

# 3. Codifica a variável de saída (STATUS): 'bom' → 0, 'mau' → 1
label_encoder = LabelEncoder()
df['cancel'] = label_encoder.fit_transform(df['cancel'])

# Garantir que colunas categóricas sejam strings
for col in CATEGORICAS:
    df[col] = df[col].astype(str)

# Codificação das variáveis categóricas
label_encoders = {}
for col in CATEGORICAS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 4. Define os preditores (X) e a variável alvo (y)
X = df.drop(['id', 'cancel'], axis=1) # colunas independentes
y = df['cancel']  # coluna dependente

# 5. Divide os dados em conjunto de treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Cria e treina o modelo de regressão logística
#modelo = LogisticRegression()
modelo = LogisticRegression(class_weight='balanced')
modelo.fit(X_train, y_train)

# 7. Faz previsões com os dados de teste
y_pred = modelo.predict(X_test)

# 8. Exibe métricas de avaliação
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))  # mostra erros/acertos

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))  # precisão, recall, f1-score

# 9. Mostra os coeficientes aprendidos para cada variável
print("\nCoeficientes do modelo:")
for var, coef in zip(X.columns, modelo.coef_[0]):
    print(f"{var}: {coef:.4f}")

# 10. Exibe o intercepto (bias)
print(f"\nIntercepto: {modelo.intercept_[0]:.4f}")



"""
# 11. Criação do gráfico da curva logística com base na IDADE
# Fixando valores médios para as demais variáveis
tmprsd_mean = df['TMPRSD'].mean()
rndtot_mean = df['RNDTOT'].mean()

# Geração de uma sequência de idades
idades = np.linspace(df['IDADE'].min(), df['IDADE'].max(), 300)

# Construção da matriz de entrada X para previsão
X_plot = pd.DataFrame({
    'IDADE': idades,
    'TMPRSD': tmprsd_mean,
    'RNDTOT': rndtot_mean
})

# Cálculo da probabilidade prevista para classe 1 (mau)
probs = modelo.predict_proba(X_plot)[:, 1]

# Gráfico
plt.figure(figsize=(10, 6))
plt.scatter(df['IDADE'], df['STATUS'], alpha=0.3, label='Observado (0=bom, 1=mau)')
plt.plot(idades, probs, color='red', label='Probabilidade prevista (classe "mau")')
plt.title("Curva de Regressão Logística em função da Idade")
plt.xlabel("Idade")
plt.ylabel("Probabilidade de ser 'mau'")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 11. Gráfico da curva logística com base em RNDTOT (Renda Total)
# Fixamos IDADE e TMPRSD na média
idade_mean = df['IDADE'].mean()
tmprsd_mean = df['TMPRSD'].mean()

# Geração de uma faixa de valores para RNDTOT
renda = np.linspace(df['RNDTOT'].min(), df['RNDTOT'].max(), 300)

# Construção do conjunto de entrada fixando idade e tempo de residência
X_plot = pd.DataFrame({
    'IDADE': idade_mean,
    'TMPRSD': tmprsd_mean,
    'RNDTOT': renda
})

# Cálculo das probabilidades previstas de ser "mau"
probs = modelo.predict_proba(X_plot)[:, 1]

# Gráfico
plt.figure(figsize=(10, 6))
plt.scatter(df['RNDTOT'], df['STATUS'], alpha=0.3, label='Observado (0=bom, 1=mau)')
plt.plot(renda, probs, color='red', label='Probabilidade prevista (classe "mau")')
plt.title("Curva de Regressão Logística em função da Renda Total")
plt.xlabel("Renda Total (RNDTOT)")
plt.ylabel("Probabilidade de ser 'mau'")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
"""
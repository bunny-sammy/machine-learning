import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import os

# Variáveis globais
CATEGORICAS = ['id', 'local', 'tvcabo', 'debaut', 'cancel']

# Definições de caminho
base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
db_dir = os.path.join(base_dir, "DADOS")
os.makedirs(db_dir, exist_ok=True)

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

# Remover linhas com outliers
df = df[(np.abs(stats.zscore(df.drop(CATEGORICAS, axis=1))) < 3).all(axis=1)]

# Remover features irrelevantes
df = df.drop(['id'], axis=1)
print(df.head())

# Salvar novo arquivo
df.to_excel('../DADOS/p33_clean.xlsx', index=False)
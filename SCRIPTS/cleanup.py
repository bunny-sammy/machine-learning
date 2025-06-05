import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# Variáveis globais
CATEGORICAS = ['local', 'tvcabo', 'debaut', 'cancel']

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

# Separar variáveis independentes e dependentes
df = df.drop(['id', 'renda', 'debaut'], axis=1)
print(df.head())
df.to_excel('p33_clean.xlsx', index=False)
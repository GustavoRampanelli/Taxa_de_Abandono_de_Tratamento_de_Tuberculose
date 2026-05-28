import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'tuberculosis-ltgu-prediction'

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report

print("Carregando bases de dados (Treinamento da Rede Neural - Escopo Acadêmico)...")
tr   = pd.read_csv(f'{DATA_DIR}/treino.csv',  low_memory=False)
t1df = pd.read_csv(f'{DATA_DIR}/teste1.csv',  low_memory=False)

def preparar(df):
    df = df.copy()
    # reingresso
    df['reingresso'] = (df['TRATAMENTO'].astype(str) == '3').astype(int)
    # dias notif -> inicio tratamento
    for col in ['DT_NOTIFIC', 'DT_INIC_TR']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col].astype(str), format='%Y%m%d', errors='coerce')
    if 'DT_NOTIFIC' in df.columns and 'DT_INIC_TR' in df.columns:
        df['dias_notif_trat'] = (df['DT_INIC_TR'] - df['DT_NOTIFIC']).dt.days.clip(0, 365)
    else:
        df['dias_notif_trat'] = np.nan
    # valor 9 = ignorado
    for col in ['AGRAVAIDS','AGRAVALCOO','AGRAVDIABE','AGRAVDOENC','AGRAVDROGA',
                'AGRAVTABAC','POP_RUA','POP_LIBER','POP_IMIG','BENEF_GOV','TRAT_SUPER']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').replace({9: np.nan, 9.0: np.nan})
    return df

print("Preparando dados...")
tr   = preparar(tr)
t1df = preparar(t1df)

S_NUM = ['reingresso','POP_RUA','AGRAVAIDS','POP_LIBER','AGRAVDROGA','idade_anos',
         'AGRAVALCOO','NU_COMU_EX','AGRAVDIABE','AGRAVTABAC','AGRAVDOENC',
         'POP_IMIG','BENEF_GOV','NU_CONTATO','TRAT_SUPER','dias_notif_trat']
S_NOM = ['CS_SEXO','HIV','TESTE_TUBE','CULTURA_ES','BACILOSC_E',
         'RAIOX_TORA','TRATAMENTO','CS_RACA','TEST_MOLEC']

S_NUM = [f for f in S_NUM if f in tr.columns]
S_NOM = [f for f in S_NOM if f in tr.columns]
SELECTED = S_NUM + S_NOM

for col in S_NOM:
    tr[col]   = tr[col].astype(str)
    t1df[col] = t1df[col].astype(str).fillna('nan')

num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', RobustScaler())])
nom_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=True, drop='first', max_categories=15))
])
pre = ColumnTransformer([('num', num_pipe, S_NUM), ('nom', nom_pipe, S_NOM)], remainder='drop')

print("Construindo e treinando a arquitetura da Rede Neural (MLP)...")
print("Pode levar alguns minutos devido ao tamanho da base de treino (562k registros).")

# Usando MLPClassifier (Multi-Layer Perceptron)
# Configuração base: 2 camadas ocultas com 64 e 32 neurônios.
# early_stopping=True ajuda a evitar overfitting e acelera o treino
mlp_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    batch_size=1024,
    learning_rate_init=0.001,
    max_iter=50,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
    verbose=True
)

pipe = Pipeline([
    ('pre', pre),
    ('clf', mlp_model)
])

pipe.fit(tr[SELECTED], tr['ltfu'])

print("\nRede Neural treinada com sucesso. Avaliando no Teste 1...")
y_prob = pipe.predict_proba(t1df[SELECTED])[:, 1]
y_pred = pipe.predict(t1df[SELECTED])

auc = roc_auc_score(t1df['ltfu'], y_prob)
f1  = f1_score(t1df['ltfu'], y_pred)

print("-" * 50)
print(f"RESULTADOS DA REDE NEURAL (MLPClassifier) NO TESTE 1")
print(f"ROC-AUC : {auc:.4f}")
print(f"F1-Score: {f1:.4f}")
print("-" * 50)
print("\nRelatório de Classificação:")
print(classification_report(t1df['ltfu'], y_pred, target_names=['Cura', 'Abandono']))

# Atualizando a tabela de resultados para refletir a Rede Neural
try:
    res = pd.read_csv('tuberculosis-ltfu-prediction-main/data/processed/resultados_modelos.csv', index_col=0)
    if 'Rede_Neural' not in res.index:
        res.loc['Rede_Neural'] = {'ROC-AUC': float(auc), 'F1': float(f1)}
        res = res.sort_values('ROC-AUC', ascending=False)
        res.to_csv('tuberculosis-ltfu-prediction-main/data/processed/resultados_modelos.csv')
        print("Tabela de 'resultados_modelos.csv' atualizada com a Rede Neural!")
except Exception as e:
    print(f"Não foi possível atualizar a tabela CSV: {e}")

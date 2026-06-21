import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'tuberculosis-ltgu-prediction'

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report

print("Carregando bases de dados (Fase 7 - Validacao Final)...")
tr   = pd.read_csv(f'{DATA_DIR}/treino.csv',  low_memory=False)
t1df = pd.read_csv(f'{DATA_DIR}/teste1.csv',  low_memory=False)
t2df = pd.read_csv(f'{DATA_DIR}/teste2.csv',  low_memory=False)

# Concatenar treino e teste1 para formar o novo conjunto de treinamento massivo
tr_full = pd.concat([tr, t1df], ignore_index=True)
print(f"Treino Original : {tr.shape}")
print(f"Teste 1         : {t1df.shape}")
print(f"Treino FULL     : {tr_full.shape} <-- Será usado para treinar agora")
print(f"Teste 2 (Holdout): {t2df.shape} <-- Base inviolada para validação")

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

print("\nPreparando dados...")
tr_full = preparar(tr_full)
t2df    = preparar(t2df)

S_NUM = ['reingresso','POP_RUA','AGRAVAIDS','POP_LIBER','AGRAVDROGA','idade_anos',
         'AGRAVALCOO','NU_COMU_EX','AGRAVDIABE','AGRAVTABAC','AGRAVDOENC',
         'POP_IMIG','BENEF_GOV','NU_CONTATO','TRAT_SUPER','dias_notif_trat']
S_NOM = ['CS_SEXO','HIV','TESTE_TUBE','CULTURA_ES','BACILOSC_E',
         'RAIOX_TORA','TRATAMENTO','CS_RACA','TEST_MOLEC']

S_NUM = [f for f in S_NUM if f in tr_full.columns]
S_NOM = [f for f in S_NOM if f in tr_full.columns]
SELECTED = S_NUM + S_NOM

for col in S_NOM:
    tr_full[col] = tr_full[col].astype(str)
    t2df[col]    = t2df[col].astype(str).fillna('nan')

num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', RobustScaler())])
nom_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=True, drop='first', max_categories=15))
])
pre = ColumnTransformer([('num', num_pipe, S_NUM), ('nom', nom_pipe, S_NOM)], remainder='drop')

neg   = int((tr_full['ltfu'] == 0).sum())
pos   = int((tr_full['ltfu'] == 1).sum())
ratio = round(neg / pos, 1)

pipe = Pipeline([
    ('pre', pre),
    ('clf', LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=ratio, subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42, verbose=-1
    ))
])

print(f"\nIniciando treinamento com a base ampliada ({len(tr_full):,} registros)...")
pipe.fit(tr_full[SELECTED], tr_full['ltfu'])
print("Treinamento finalizado.")

print("\nExecutando inferência no Teste 2 (Holdout inviolado)...")
y_prob = pipe.predict_proba(t2df[SELECTED])[:, 1]
y_pred = pipe.predict(t2df[SELECTED])

auc = roc_auc_score(t2df['ltfu'], y_prob)
f1  = f1_score(t2df['ltfu'], y_pred)

print("-" * 50)
print(f"RESULTADOS OFICIAIS DA VALIDACAO FINAL (Teste 2)")
print(f"ROC-AUC : {auc:.4f}")
print(f"F1-Score: {f1:.4f}")
print("-" * 50)
print("\nRelatório de Classificação (Precision / Recall):")
print(classification_report(t2df['ltfu'], y_pred, target_names=['Cura', 'Abandono']))

import pickle, json, os

OUT_DIR = 'tuberculosis-ltfu-prediction-main/data/processed'
os.makedirs(OUT_DIR, exist_ok=True)

# Salvar o modelo treinado final (com dados de treino e teste1) para Produção
with open(f'{OUT_DIR}/pipeline_final.pkl', 'wb') as f:
    pickle.dump(pipe, f)

# Metadados para a API
meta = {
    'modelo': 'LightGBM (Produção Fase 7)',
    'features_num': S_NUM,
    'features_nom': S_NOM,
    'features_all': SELECTED,
    'roc_auc_teste2': round(float(auc), 4),
    'f1_teste2': round(float(f1), 4),
    'n_treino': int(len(tr_full)),
    'prop_abandono_treino': round(float(tr_full['ltfu'].mean()), 4),
    'scale_pos_weight': ratio
}
with open(f'{OUT_DIR}/model_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print("\nDeploy Local Concluído! Arquivos 'pipeline_final.pkl' e 'model_metadata.json' atualizados para a nova versão robusta.")

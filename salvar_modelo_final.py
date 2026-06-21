import pandas as pd, numpy as np, pickle, json, os, warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'tuberculosis-ltgu-prediction'
OUT_DIR  = 'tuberculosis-ltfu-prediction-main/data/processed'
os.makedirs(OUT_DIR, exist_ok=True)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report

print("Carregando dados...")
tr   = pd.read_csv(f'{DATA_DIR}/treino.csv',  low_memory=False)
t1df = pd.read_csv(f'{DATA_DIR}/teste1.csv',  low_memory=False)
print(f"treino: {tr.shape} | teste1: {t1df.shape}")

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

tr   = preparar(tr)
t1df = preparar(t1df)

# Features conforme seleção LASSO da sessão anterior
S_NUM = ['reingresso','POP_RUA','AGRAVAIDS','POP_LIBER','AGRAVDROGA','idade_anos',
         'AGRAVALCOO','NU_COMU_EX','AGRAVDIABE','AGRAVTABAC','AGRAVDOENC',
         'POP_IMIG','BENEF_GOV','NU_CONTATO','TRAT_SUPER','dias_notif_trat']
S_NOM = ['CS_SEXO','HIV','TESTE_TUBE','CULTURA_ES','BACILOSC_E',
         'RAIOX_TORA','TRATAMENTO','CS_RACA','TEST_MOLEC']

# Filtrar apenas colunas existentes
S_NUM = [f for f in S_NUM if f in tr.columns]
S_NOM = [f for f in S_NOM if f in tr.columns]
SELECTED = S_NUM + S_NOM

for col in S_NOM:
    tr[col]   = tr[col].astype(str)
    t1df[col] = t1df[col].astype(str).fillna('nan')

print(f"Features numericas/binarias: {S_NUM}")
print(f"Features nominais          : {S_NOM}")

num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', RobustScaler())])
nom_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=True, drop='first', max_categories=15))
])
pre = ColumnTransformer([('num', num_pipe, S_NUM), ('nom', nom_pipe, S_NOM)], remainder='drop')

neg   = int((tr['ltfu'] == 0).sum())
pos   = int((tr['ltfu'] == 1).sum())
ratio = round(neg / pos, 1)
print(f"scale_pos_weight: {ratio} (neg={neg:,} / pos={pos:,})")

pipe = Pipeline([
    ('pre', pre),
    ('clf', LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=ratio, subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42, verbose=-1
    ))
])

print("Treinando LightGBM final (562k registros)...")
pipe.fit(tr[SELECTED], tr['ltfu'])
print("Treino concluido!")

y_prob = pipe.predict_proba(t1df[SELECTED])[:, 1]
y_pred = pipe.predict(t1df[SELECTED])
auc = roc_auc_score(t1df['ltfu'], y_prob)
f1  = f1_score(t1df['ltfu'], y_pred)

print(f"\nROC-AUC (teste1): {auc:.4f}")
print(f"F1      (teste1): {f1:.4f}")
print()
print(classification_report(t1df['ltfu'], y_pred, target_names=['Cura', 'Abandono']))

# Salvar pipeline
with open(f'{OUT_DIR}/pipeline_final.pkl', 'wb') as f:
    pickle.dump(pipe, f)

# Metadados para a API
meta = {
    'modelo': 'LightGBM',
    'features_num': S_NUM,
    'features_nom': S_NOM,
    'features_all': SELECTED,
    'roc_auc_teste1': round(float(auc), 4),
    'f1_teste1': round(float(f1), 4),
    'n_treino': int(len(tr)),
    'prop_abandono_treino': round(float(tr['ltfu'].mean()), 4),
    'scale_pos_weight': ratio
}
with open(f'{OUT_DIR}/model_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

# Tabela comparativa (resultados da sessao anterior)
res = pd.DataFrame({
    'LightGBM':     {'ROC-AUC': 0.8265, 'F1': 0.7262},
    'XGBoost':      {'ROC-AUC': 0.8253, 'F1': 0.7262},
    'LogReg_LASSO': {'ROC-AUC': 0.8118, 'F1': 0.7070},
    'RandomForest': {'ROC-AUC': 0.8116, 'F1': 0.7337}
}).T.sort_values('ROC-AUC', ascending=False)
res.to_csv(f'{OUT_DIR}/resultados_modelos.csv')

print("Arquivos salvos:")
print(f"  {OUT_DIR}/pipeline_final.pkl")
print(f"  {OUT_DIR}/model_metadata.json")
print(f"  {OUT_DIR}/resultados_modelos.csv")
print("Concluido!")

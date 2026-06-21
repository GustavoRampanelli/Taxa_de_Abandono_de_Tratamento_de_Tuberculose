"""
Script de Treino — Projeto LTFU-TB (versão corrigida)
Features explícitas por tipo — sem estouro de memória
"""
import pandas as pd
import numpy as np
import pickle
import warnings
import time
warnings.filterwarnings('ignore')

DATA_DIR = 'tuberculosis-ltgu-prediction'
OUT_DIR  = 'tuberculosis-ltfu-prediction-main/data/processed'
import os; os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. CARREGAR ────────────────────────────────────────────
print("=" * 55)
print("1. CARREGANDO DADOS")
print("=" * 55)
t0 = time.time()
tr   = pd.read_csv(f'{DATA_DIR}/treino.csv',  low_memory=False)
t1df = pd.read_csv(f'{DATA_DIR}/teste1.csv',  low_memory=False)
print(f"treino : {tr.shape[0]:,} x {tr.shape[1]}  |  ltfu=1: {tr['ltfu'].mean()*100:.1f}%")
print(f"teste1 : {t1df.shape[0]:,} x {t1df.shape[1]} |  ltfu=1: {t1df['ltfu'].mean()*100:.1f}%")
print(f"Lido em {time.time()-t0:.1f}s")

TARGET = 'ltfu'

# ── 2. FEATURE ENGINEERING ─────────────────────────────────
print("\n" + "=" * 55)
print("2. FEATURE ENGINEERING")
print("=" * 55)

for df in [tr, t1df]:
    # dias entre notificação e início do tratamento
    for col in ['DT_NOTIFIC', 'DT_INIC_TR']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col].astype(str), format='%Y%m%d', errors='coerce')
    if 'DT_NOTIFIC' in df.columns and 'DT_INIC_TR' in df.columns:
        df['dias_notif_trat'] = (df['DT_INIC_TR'] - df['DT_NOTIFIC']).dt.days.clip(0, 365)

    # reingresso (já pode existir no CSV)
    if 'reingresso' not in df.columns:
        df['reingresso'] = (df['TRATAMENTO'].astype(str) == '3').astype(int)

    # valor 9 = ignorado → NaN
    for col in ['AGRAVAIDS','AGRAVALCOO','AGRAVDIABE','AGRAVDOENC',
                'AGRAVDROGA','AGRAVTABAC','POP_RUA','POP_LIBER',
                'POP_IMIG','POP_SAUDE','BENEF_GOV','TRAT_SUPER',
                'TRATSUP_AT','CS_GESTANT']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').replace({9: np.nan, 9.0: np.nan})

print("Feature engineering concluído")

# ── 3. DEFINIR FEATURES EXPLICITAMENTE ────────────────────
# Numéricas contínuas
FEAT_NUM = ['idade_anos', 'NU_CONTATO', 'dias_notif_trat', 'NU_COMU_EX']

# Binárias SINAN (1=sim, 2=não) → tratar como numéricas após NaN
FEAT_BIN = ['AGRAVAIDS','AGRAVALCOO','AGRAVDIABE','AGRAVDOENC',
            'AGRAVDROGA','AGRAVTABAC','POP_RUA','POP_LIBER',
            'POP_IMIG','BENEF_GOV','TRAT_SUPER','reingresso']

# Nominais de baixa cardinalidade (< 10 valores únicos)
FEAT_NOM = ['CS_SEXO','CS_RACA','HIV','BACILOSC_E','CULTURA_ES',
            'FORMA','TRATAMENTO','TEST_MOLEC','RAIOX_TORA','TESTE_TUBE']

# Filtrar apenas colunas existentes
FEAT_NUM = [f for f in FEAT_NUM if f in tr.columns]
FEAT_BIN = [f for f in FEAT_BIN if f in tr.columns]
FEAT_NOM = [f for f in FEAT_NOM if f in tr.columns]
ALL_FEAT = FEAT_NUM + FEAT_BIN + FEAT_NOM

print(f"Numéricas  : {FEAT_NUM}")
print(f"Binárias   : {FEAT_BIN}")
print(f"Nominais   : {FEAT_NOM}")
print(f"Total      : {len(ALL_FEAT)} features")

# ── 4. CORRELAÇÃO DE PEARSON ───────────────────────────────
print("\n" + "=" * 55)
print("3. CORRELAÇÃO DE PEARSON")
print("=" * 55)

X_enc = tr[ALL_FEAT].copy()
for col in FEAT_NOM:
    codes = pd.Categorical(X_enc[col].astype(str)).codes.astype(float)
    codes[codes == -1] = np.nan
    X_enc[col] = codes

corr = X_enc.corrwith(tr[TARGET]).dropna().sort_values(key=abs, ascending=False)
print(corr.round(3).to_string())

# ── 5. LASSO PARA SELEÇÃO ──────────────────────────────────
print("\n" + "=" * 55)
print("4. SELEÇÃO LASSO")
print("=" * 55)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

imp_l = SimpleImputer(strategy='most_frequent')
X_imp = imp_l.fit_transform(X_enc[corr.index])
X_sc  = RobustScaler().fit_transform(X_imp)

# Subsample 150k para velocidade
rng = np.random.default_rng(42)
idx = rng.choice(len(X_sc), size=min(150_000, len(X_sc)), replace=False)

lasso = LogisticRegression(
    penalty='l1', solver='liblinear', C=0.05,
    class_weight='balanced', max_iter=1000, random_state=42
)
lasso.fit(X_sc[idx], tr[TARGET].iloc[idx])

coef = pd.DataFrame({'feature': list(corr.index), 'coef': lasso.coef_[0]})
coef = coef[coef['coef'] != 0].sort_values('coef', key=abs, ascending=False)
SELECTED = coef['feature'].tolist()
print(f"Features selecionadas: {len(SELECTED)}")
print(coef.round(3).to_string(index=False))

# Separar selecionadas por tipo
S_NUM = [f for f in SELECTED if f in FEAT_NUM + FEAT_BIN]
S_NOM = [f for f in SELECTED if f in FEAT_NOM]
print(f"\nNuméricas/binárias selecionadas: {S_NUM}")
print(f"Nominais selecionadas: {S_NOM}")

# ── 6. MONTAR PIPELINE ─────────────────────────────────────
print("\n" + "=" * 55)
print("5. PIPELINE SKLEARN")
print("=" * 55)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Converter nominais para string
for col in S_NOM:
    tr[col]   = tr[col].astype(str)
    t1df[col] = t1df[col].astype(str).fillna('nan')

num_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('sc',  RobustScaler())
])
nom_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('enc', OneHotEncoder(
        handle_unknown='ignore', sparse_output=True,
        drop='first', max_categories=15
    ))
])

transformers = []
if S_NUM: transformers.append(('num', num_pipe, S_NUM))
if S_NOM: transformers.append(('nom', nom_pipe, S_NOM))
preprocessor = ColumnTransformer(transformers, remainder='drop')

X_train = tr[SELECTED].copy()
y_train = tr[TARGET].copy()
X_test1 = t1df[SELECTED].copy()
y_test1 = t1df[TARGET].copy()

neg = int((y_train == 0).sum())
pos = int((y_train == 1).sum())
ratio = round(neg / pos, 1)
print(f"scale_pos_weight: {ratio}")

# ── 7. TREINAR 4 MODELOS ───────────────────────────────────
print("\n" + "=" * 55)
print("6. TREINANDO MODELOS")
print("=" * 55)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

modelos = {
    'LogReg_LASSO': Pipeline([('pre', preprocessor), ('clf', LogisticRegression(
        penalty='l1', solver='liblinear', C=0.1,
        class_weight='balanced', max_iter=1000, random_state=42
    ))]),
    'RandomForest': Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=50,
        class_weight='balanced', n_jobs=-1, random_state=42
    ))]),
    'XGBoost': Pipeline([('pre', preprocessor), ('clf', XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=ratio, subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, verbosity=0
    ))]),
    'LightGBM': Pipeline([('pre', preprocessor), ('clf', LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=ratio, subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42, verbose=-1
    ))]),
}

resultados = {}
melhor_auc = 0; melhor_nome = None; melhor_pipe = None

for nome, pipe in modelos.items():
    print(f"\n--- {nome} ---")
    t_start = time.time()
    pipe.fit(X_train, y_train)
    print(f"  Treino: {time.time()-t_start:.0f}s")

    y_prob = pipe.predict_proba(X_test1)[:, 1]
    y_pred = pipe.predict(X_test1)
    auc = roc_auc_score(y_test1, y_prob)
    f1  = f1_score(y_test1, y_pred)
    resultados[nome] = {'ROC-AUC': round(auc, 4), 'F1': round(f1, 4)}
    print(f"  ROC-AUC: {auc:.4f}  |  F1: {f1:.4f}")

    if auc > melhor_auc:
        melhor_auc = auc; melhor_nome = nome; melhor_pipe = pipe

# ── 8. RESULTADOS E SALVAR ─────────────────────────────────
print("\n" + "=" * 55)
print("7. RESULTADOS FINAIS")
print("=" * 55)

res_df = pd.DataFrame(resultados).T.sort_values('ROC-AUC', ascending=False)
print(res_df.to_string())
print(f"\n🏆 Melhor: {melhor_nome} (ROC-AUC={melhor_auc:.4f})")
print(f"Meta ≥0.80: {'✅' if melhor_auc>=0.80 else '❌'}  |  Meta F1≥0.70: {'✅' if resultados[melhor_nome]['F1']>=0.70 else '❌'}")

print("\n" + classification_report(y_test1, melhor_pipe.predict(X_test1), target_names=['Cura','Abandono']))

# Salvar
with open(f'{OUT_DIR}/pipeline_final.pkl', 'wb') as f:
    pickle.dump(melhor_pipe, f)
res_df.to_csv(f'{OUT_DIR}/resultados_modelos.csv')
with open(f'{OUT_DIR}/features_selecionadas.txt', 'w') as f:
    f.write('\n'.join(SELECTED))

# Salvar metadados para a API
import json
meta = {
    'modelo': melhor_nome,
    'features': SELECTED,
    'feat_num': S_NUM,
    'feat_nom': S_NOM,
    'roc_auc_teste1': resultados[melhor_nome]['ROC-AUC'],
    'f1_teste1': resultados[melhor_nome]['F1'],
    'n_treino': int(len(y_train)),
    'prop_abandono_treino': round(float(y_train.mean()), 4)
}
with open(f'{OUT_DIR}/model_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\nSalvo em {OUT_DIR}/")
print("  pipeline_final.pkl")
print("  resultados_modelos.csv")
print("  features_selecionadas.txt")
print("  model_metadata.json")
print("\nConcluído!")

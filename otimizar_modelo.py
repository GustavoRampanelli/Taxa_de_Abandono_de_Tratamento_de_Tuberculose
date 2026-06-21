import pandas as pd
import numpy as np
import warnings
import time
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import RandomizedSearchCV

DATA_DIR = 'tuberculosis-ltgu-prediction'

print("Iniciando processo de Otimização (Tuning) do Modelo LightGBM...")
print("Carregando bases de dados...")
tr   = pd.read_csv(f'{DATA_DIR}/treino.csv',  low_memory=False)
t1df = pd.read_csv(f'{DATA_DIR}/teste1.csv',  low_memory=False)
t2df = pd.read_csv(f'{DATA_DIR}/teste2.csv',  low_memory=False)

# Vamos treinar a busca APENAS no `treino.csv` 
# Mas como o RandomizedSearchCV usa CrossValidation interno, podemos até fornecer o treino+teste1 para ele, 
# mas vamos manter conservador ou usar um PredefinedSplit. 
# Para manter simples e robusto, vamos concatenar tr e t1df e usar o CV padrão de 3-folds.
tr_full = pd.concat([tr, t1df], ignore_index=True)
print(f"Base de busca de hiperparâmetros (Treino + Teste1): {tr_full.shape}")
print(f"Base Holdout inviolada (Teste 2): {t2df.shape}")

def preparar(df):
    df = df.copy()
    
    # Feature 1: Reingresso
    df['reingresso'] = (df['TRATAMENTO'].astype(str) == '3').astype(int)
    
    # Feature 2: Atraso Notificação
    for col in ['DT_NOTIFIC', 'DT_INIC_TR']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col].astype(str), format='%Y%m%d', errors='coerce')
    if 'DT_NOTIFIC' in df.columns and 'DT_INIC_TR' in df.columns:
        df['dias_notif_trat'] = (df['DT_INIC_TR'] - df['DT_NOTIFIC']).dt.days.clip(0, 365)
    else:
        df['dias_notif_trat'] = np.nan
        
    # Limpeza de ignorados
    comorbidades = ['AGRAVAIDS','AGRAVALCOO','AGRAVDIABE','AGRAVDOENC','AGRAVDROGA','AGRAVTABAC']
    vulnerabilidades = ['POP_RUA','POP_LIBER','POP_IMIG','BENEF_GOV']
    outros = ['TRAT_SUPER']
    
    for col in comorbidades + vulnerabilidades + outros:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').replace({9: np.nan, 9.0: np.nan})
            
    # FEATURE ENGINEERING AVANÇADA
    # Soma de Comorbidades
    df['total_comorbidades'] = df[comorbidades].apply(lambda x: (x == 1).sum(), axis=1)
    
    # Soma de Vulnerabilidades Sociais
    df['total_vulnerabilidades'] = df[vulnerabilidades].apply(lambda x: (x == 1).sum(), axis=1)
    
    # Flag Alta Vulnerabilidade (Se tem rua, prisão ou drogas/alcool associado)
    df['flag_alta_vulnerabilidade'] = ((df['POP_RUA'] == 1) | (df['POP_LIBER'] == 1) | (df['AGRAVDROGA'] == 1)).astype(int)
    
    return df

print("\nAplicando Engenharia de Features...")
tr_full = preparar(tr_full)
t2df    = preparar(t2df)

S_NUM = ['reingresso','POP_RUA','AGRAVAIDS','POP_LIBER','AGRAVDROGA','idade_anos',
         'AGRAVALCOO','NU_COMU_EX','AGRAVDIABE','AGRAVTABAC','AGRAVDOENC',
         'POP_IMIG','BENEF_GOV','NU_CONTATO','TRAT_SUPER','dias_notif_trat',
         'total_comorbidades', 'total_vulnerabilidades', 'flag_alta_vulnerabilidade']
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
    ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=True, drop='first'))
])
pre = ColumnTransformer([('num', num_pipe, S_NUM), ('nom', nom_pipe, S_NOM)], remainder='drop')

neg   = int((tr_full['ltfu'] == 0).sum())
pos   = int((tr_full['ltfu'] == 1).sum())
ratio = round(neg / pos, 1)

# Baseline Pipeline Model
clf = LGBMClassifier(n_jobs=-1, random_state=42, verbose=-1, scale_pos_weight=ratio)
pipe = Pipeline([('pre', pre), ('clf', clf)])

# Parâmetros para buscar
param_dist = {
    'clf__n_estimators': [100, 300, 500, 800],
    'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'clf__max_depth': [4, 6, 8, -1],
    'clf__num_leaves': [15, 31, 63, 127],
    'clf__min_child_samples': [20, 50, 100, 300],
    'clf__subsample': [0.6, 0.8, 1.0],
    'clf__colsample_bytree': [0.6, 0.8, 1.0],
    'clf__reg_alpha': [0.0, 0.1, 1.0],
    'clf__reg_lambda': [0.0, 0.1, 1.0]
}

print(f"\nIniciando RandomizedSearchCV com 20 iterações e CV=3 (Total: 60 fits)...")
print("Isso pode levar alguns minutos. Aguarde...")
start_time = time.time()

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=20, # Reduzido para ser rápido e eficiente
    scoring='roc_auc',
    cv=3,
    random_state=42,
    n_jobs=1, # Colocamos 1 no job_lib para evitar crashes de memoria com pipelines gigantes; o lightgbm já usa n_jobs=-1 internamente
    verbose=2
)

search.fit(tr_full[SELECTED], tr_full['ltfu'])

end_time = time.time()
print(f"\nBusca concluída em {(end_time - start_time)/60:.2f} minutos.")
print("Melhores hiperparâmetros encontrados:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")
    
best_model = search.best_estimator_

print("\nExecutando inferência FINAL no Teste 2 (Holdout inviolado)...")
y_prob = best_model.predict_proba(t2df[SELECTED])[:, 1]
y_pred = best_model.predict(t2df[SELECTED])

auc = roc_auc_score(t2df['ltfu'], y_prob)
f1  = f1_score(t2df['ltfu'], y_pred)

print("-" * 50)
print(f"=== RESULTADOS APÓS OTIMIZAÇÃO (Teste 2) ===")
print(f"ROC-AUC : {auc:.5f} (Anterior: ~0.8594)")
print(f"F1-Score: {f1:.5f}")
print("-" * 50)

if auc > 0.8594:
    print(">> SUCESSO! Encontramos um modelo melhor.")
else:
    print(">> O modelo original ainda era melhor ou equivalente. As novas features não geraram ganhos no Holdout.")

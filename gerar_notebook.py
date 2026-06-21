import json

cells = [
    {"cell_type":"markdown","metadata":{},"source":[
        "# Pipeline de Preparação de Dados — Macro 3\n",
        "Passos: seleção estatística de preditores, encoding, scaling, imputação, classes raras, data leakage."
    ]},

    # 1. Carregar e filtrar
    {"cell_type":"markdown","metadata":{},"source":["## 1. Carregar e Filtrar Dados"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "import pandas as pd\nimport numpy as np\nimport warnings\nwarnings.filterwarnings('ignore')\n\n",
        "# Tenta treino.csv; fallback para amostra\n",
        "import os\n",
        "DATA_PATH = 'treino.csv' if os.path.exists('treino.csv') else 'amostra_2025.xlsx'\n",
        "if DATA_PATH.endswith('.csv'):\n",
        "    df_raw = pd.read_csv(DATA_PATH)\n",
        "else:\n",
        "    df_raw = pd.read_excel(DATA_PATH)\n",
        "print(f'Fonte: {DATA_PATH} | Shape: {df_raw.shape}')\n"
    ]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "# Decodificar idade\n",
        "df_raw['nu_str'] = df_raw['NU_IDADE_N'].astype(str).str.zfill(4)\n",
        "df_raw['idade_unid'] = pd.to_numeric(df_raw['nu_str'].str[0], errors='coerce')\n",
        "df_raw['idade_val']  = pd.to_numeric(df_raw['nu_str'].str[1:], errors='coerce')\n",
        "df_raw['idade_anos'] = np.where(df_raw['idade_unid']==4, df_raw['idade_val'],\n",
        "                        np.where(df_raw['idade_unid']==3, (df_raw['idade_val']/12).round(),\n",
        "                        np.where(df_raw['idade_unid']==2, (df_raw['idade_val']/365).round(), np.nan)))\n\n",
        "# Filtros alinhados com data-prep.py\n",
        "df = df_raw.copy()\n",
        "df = df[df['idade_anos'] >= 18]\n",
        "df = df[df['FORMA'].astype(str) == '1']\n",
        "df = df[df['TRATAMENTO'].astype(str) != '6']\n",
        "df = df[df['SITUA_ENCE'].isin([1.0, 2.0])]\n\n",
        "# Variável-alvo\n",
        "df['ltfu'] = df['SITUA_ENCE'].map({1.0: 0, 2.0: 1}).astype(int)\n",
        "print(f'Após filtros: {df.shape}')\n",
        "print(f'ltfu=1: {df[\"ltfu\"].sum()} | ltfu=0: {(df[\"ltfu\"]==0).sum()}')\n"
    ]},

    # 2. Drop leakage e colunas inúteis
    {"cell_type":"markdown","metadata":{},"source":["## 2. Remover Colunas com Data Leakage e 100% Nulas"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "# Drop 100% nulos\n",
        "null_pct = df.isnull().mean()\n",
        "drop_null = null_pct[null_pct == 1.0].index.tolist()\n",
        "print(f'Drop 100% nulos ({len(drop_null)}): {drop_null}')\n\n",
        "# Data leakage: revelam o desfecho\n",
        "drop_leakage = ['SITUA_ENCE','DT_ENCERRA','SITUA_9_M','SITUA_12_M',\n",
        "                 'BACILOSC_1','BACILOSC_2','BACILOSC_3','BACILOSC_4','BACILOSC_5','BACILOSC_6',\n",
        "                 'BAC_APOS_6','TRANSF','UF_TRANSF','MUN_TRANSF']\n\n",
        "# Campos administrativos sem valor preditivo\n",
        "drop_admin = ['TP_NOT','ID_AGRAVO','NU_ANO','DT_DIGITA','DT_TRANSUS','DT_TRANSDM',\n",
        "               'DT_TRANSSM','DT_TRANSRM','DT_TRANSRS','DT_TRANSSE','NDUPLIC_N',\n",
        "               'IN_VINCULA','FLXRECEBI','MIGRADO_W','CS_FLXRET','NU_IDADE_N',\n",
        "               'nu_str','idade_unid','idade_val']\n\n",
        "drop_all = list(set(drop_null + drop_leakage + drop_admin))\n",
        "drop_all = [c for c in drop_all if c in df.columns]\n",
        "df = df.drop(columns=drop_all)\n",
        "print(f'Colunas restantes: {df.shape[1]} (removidas: {len(drop_all)})')\n"
    ]},

    # 3. Feature engineering
    {"cell_type":"markdown","metadata":{},"source":["## 3. Feature Engineering"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "# Feature: reingresso apos abandono anterior\n",
        "df['reingresso'] = (df['TRATAMENTO'].astype(str) == '3').astype(int)\n",
        "print('reingresso=1:', df['reingresso'].sum())\n\n",
        "# Feature: atraso entre notificacao e inicio do tratamento\n",
        "for col in ['DT_NOTIFIC','DT_INIC_TR','DT_DIAG']:\n",
        "    if col in df.columns:\n",
        "        df[col] = pd.to_datetime(df[col].astype(str), format='%Y%m%d', errors='coerce')\n\n",
        "if 'DT_INIC_TR' in df.columns and 'DT_NOTIFIC' in df.columns:\n",
        "    df['dias_notif_trat'] = (df['DT_INIC_TR'] - df['DT_NOTIFIC']).dt.days\n",
        "    df['dias_notif_trat'] = df['dias_notif_trat'].clip(0, 365)\n",
        "    print('dias_notif_trat: media=', round(df['dias_notif_trat'].mean(), 1))\n\n",
        "# Tratar colunas de data como numericas (ano, mes) ou remover\n",
        "date_cols = [c for c in df.columns if df[c].dtype == 'datetime64[ns]']\n",
        "df = df.drop(columns=date_cols)\n",
        "print('Colunas de data removidas:', date_cols)\n"
    ]},

    # 4. Definir grupos de colunas
    {"cell_type":"markdown","metadata":{},"source":["## 4. Classificar Colunas por Tipo de Encoding"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "TARGET = 'ltfu'\n",
        "EXCLUDE = [TARGET]\n\n",
        "# Numericas continuas\n",
        "NUM_COLS = ['idade_anos','NU_CONTATO','dias_notif_trat']\n",
        "NUM_COLS = [c for c in NUM_COLS if c in df.columns]\n\n",
        "# Ordinal: escolaridade\n",
        "ORD_COLS = ['CS_ESCOL_N']\n",
        "ORD_COLS = [c for c in ORD_COLS if c in df.columns]\n\n",
        "# Binarias com 9=ignorado -> tratar 9 como NaN e depois OneHot\n",
        "BIN_COLS = ['CS_SEXO','AGRAVAIDS','AGRAVALCOO','AGRAVDIABE','AGRAVDOENC',\n",
        "             'AGRAVDROGA','AGRAVTABAC','POP_RUA','POP_LIBER','POP_IMIG',\n",
        "             'POP_SAUDE','BENEF_GOV','reingresso']\n",
        "BIN_COLS = [c for c in BIN_COLS if c in df.columns]\n\n",
        "# Nominais multiclasse\n",
        "NOM_COLS = ['CS_RACA','HIV','BACILOSC_E','CULTURA_ES','FORMA','TRATAMENTO',\n",
        "             'TEST_MOLEC','RAIO_TORA','TESTE_TUBE']\n",
        "NOM_COLS = [c for c in NOM_COLS if c in df.columns]\n\n",
        "print(f'Numericas: {NUM_COLS}')\n",
        "print(f'Ordinais: {ORD_COLS}')\n",
        "print(f'Binarias: {BIN_COLS}')\n",
        "print(f'Nominais: {NOM_COLS}')\n",
        "FEATURES = NUM_COLS + ORD_COLS + BIN_COLS + NOM_COLS\n",
        "print(f'Total features: {len(FEATURES)}')\n"
    ]},

    # 5. Tratamento de 9=ignorado e classes raras
    {"cell_type":"markdown","metadata":{},"source":["## 5. Tratar Valor 9 (Ignorado no SINAN) e Classes Raras"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "# Valor 9 em variaveis binarias SINAN = ignorado/nao informado\n",
        "# Substituir por NaN para tratamento pelo imputer\n",
        "cols_com_9 = ['AGRAVAIDS','AGRAVALCOO','AGRAVDIABE','AGRAVDOENC','AGRAVDROGA',\n",
        "               'AGRAVTABAC','POP_RUA','POP_LIBER','POP_IMIG','POP_SAUDE','BENEF_GOV']\n",
        "for col in cols_com_9:\n",
        "    if col in df.columns:\n",
        "        df[col] = df[col].replace(9.0, np.nan)\n",
        "        df[col] = df[col].replace(9, np.nan)\n\n",
        "# Classes raras: agrupar valores com freq < 5% em categoria 'OUTRO'\n",
        "RARE_THRESHOLD = 0.05\n",
        "X = df[FEATURES].copy()\n",
        "cat_cols = ORD_COLS + BIN_COLS + NOM_COLS\n",
        "for col in cat_cols:\n",
        "    if col not in X.columns:\n",
        "        continue\n",
        "    freq = X[col].value_counts(normalize=True)\n",
        "    rare = freq[freq < RARE_THRESHOLD].index\n",
        "    if len(rare) > 0:\n",
        "        X[col] = X[col].replace(rare, 'OUTRO')\n",
        "        print(f'{col}: {len(rare)} classe(s) rara(s) agrupadas em OUTRO -> {list(rare)}')\n",
        "print('Tratamento de classes raras concluido.')\n"
    ]},

    # 6. Correlacao Pearson
    {"cell_type":"markdown","metadata":{},"source":["## 6. Seleção Estatística — Correlação de Pearson com ltfu"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "import matplotlib.pyplot as plt\n\n",
        "plt.rcParams.update({'figure.facecolor':'#0f0f0f','axes.facecolor':'#1a1a1a',\n",
        "    'axes.edgecolor':'#444','axes.labelcolor':'#ccc','xtick.color':'#aaa',\n",
        "    'ytick.color':'#aaa','text.color':'#eee','grid.color':'#333','grid.alpha':0.5})\n\n",
        "# Codificar categoricas para correlacao\n",
        "X_enc = X.copy()\n",
        "y = df[TARGET]\n",
        "for col in cat_cols:\n",
        "    if col in X_enc.columns:\n",
        "        X_enc[col] = pd.Categorical(X_enc[col].astype(str)).codes.replace(-1, np.nan)\n\n",
        "corr = X_enc.corrwith(y).dropna().sort_values(key=abs, ascending=False)\n",
        "print('Top 15 correlacoes com ltfu:')\n",
        "print(corr.head(15).round(3).to_string())\n\n",
        "fig, ax = plt.subplots(figsize=(10,7))\n",
        "top = corr.head(20)\n",
        "colors = ['#e74c3c' if v > 0 else '#3498db' for v in top.values]\n",
        "ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])\n",
        "ax.axvline(0, color='#aaa', linewidth=0.8)\n",
        "ax.set_title('Correlacao de Pearson com ltfu (top 20)')\n",
        "ax.set_xlabel('Correlacao')\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]},

    # 7. LASSO
    {"cell_type":"markdown","metadata":{},"source":["## 7. Seleção Estatística — LASSO (Regularizacao L1)"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.preprocessing import RobustScaler\n",
        "from sklearn.impute import SimpleImputer\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder\n\n",
        "# Preparar X_lasso com imputacao simples para rodar LASSO\n",
        "X_lasso = X_enc.copy()\n",
        "imp = SimpleImputer(strategy='most_frequent')\n",
        "X_lasso_imp = imp.fit_transform(X_lasso[corr.index])\n",
        "scaler = RobustScaler()\n",
        "X_lasso_sc = scaler.fit_transform(X_lasso_imp)\n\n",
        "lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, max_iter=1000, random_state=42)\n",
        "lasso.fit(X_lasso_sc, y)\n\n",
        "coef_df = pd.DataFrame({'feature': corr.index, 'coef': lasso.coef_[0]})\n",
        "coef_df = coef_df[coef_df['coef'] != 0].sort_values('coef', key=abs, ascending=False)\n",
        "print(f'Features selecionadas pelo LASSO: {len(coef_df)}')\n",
        "print(coef_df.to_string(index=False))\n\n",
        "SELECTED_FEATURES = coef_df['feature'].tolist()\n"
    ]},

    # 8. Pipeline completo
    {"cell_type":"markdown","metadata":{},"source":["## 8. Pipeline Sklearn — RobustScaler + Encoders + IterativeImputer"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "from sklearn.experimental import enable_iterative_imputer\n",
        "from sklearn.impute import IterativeImputer\n",
        "from sklearn.model_selection import cross_val_score\n",
        "from sklearn.ensemble import RandomForestClassifier\n\n",
        "# Filtrar apenas features selecionadas pelo LASSO que existem\n",
        "feat_num = [f for f in SELECTED_FEATURES if f in NUM_COLS]\n",
        "feat_ord = [f for f in SELECTED_FEATURES if f in ORD_COLS]\n",
        "feat_nom = [f for f in SELECTED_FEATURES if f in (BIN_COLS + NOM_COLS)]\n\n",
        "print('Features numericas no pipeline:', feat_num)\n",
        "print('Features ordinais no pipeline:', feat_ord)\n",
        "print('Features nominais no pipeline:', feat_nom)\n\n",
        "# Sub-pipelines\n",
        "num_pipe = Pipeline([\n",
        "    ('imputer', IterativeImputer(max_iter=10, random_state=42)),\n",
        "    ('scaler', RobustScaler())\n",
        "])\n\n",
        "ord_pipe = Pipeline([\n",
        "    ('imputer', SimpleImputer(strategy='most_frequent')),\n",
        "    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))\n",
        "])\n\n",
        "nom_pipe = Pipeline([\n",
        "    ('imputer', SimpleImputer(strategy='most_frequent')),\n",
        "    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))\n",
        "])\n\n",
        "transformers = []\n",
        "if feat_num: transformers.append(('num', num_pipe, feat_num))\n",
        "if feat_ord: transformers.append(('ord', ord_pipe, feat_ord))\n",
        "if feat_nom: transformers.append(('nom', nom_pipe, feat_nom))\n\n",
        "preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')\n\n",
        "full_pipeline = Pipeline([\n",
        "    ('preprocess', preprocessor),\n",
        "    ('clf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))\n",
        "])\n\n",
        "print('Pipeline construido com sucesso!')\n",
        "print(full_pipeline)\n"
    ]},

    # 9. Validacao cruzada
    {"cell_type":"markdown","metadata":{},"source":["## 9. Validação Cruzada — Baseline do Modelo"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "from sklearn.model_selection import StratifiedKFold\n",
        "from sklearn.metrics import roc_auc_score\n\n",
        "X_pipe = X[SELECTED_FEATURES].copy()\n",
        "# Substituir 9 por NaN nas colunas nominais\n",
        "for col in feat_nom:\n",
        "    if col in X_pipe.columns:\n",
        "        X_pipe[col] = X_pipe[col].astype(str)\n\n",
        "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
        "scores = cross_val_score(full_pipeline, X_pipe, y, cv=cv,\n",
        "                          scoring='roc_auc', error_score='raise')\n\n",
        "print(f'ROC-AUC por fold: {scores.round(3)}')\n",
        "print(f'Media: {scores.mean():.3f} +/- {scores.std():.3f}')\n",
        "print()\n",
        "scores_f1 = cross_val_score(full_pipeline, X_pipe, y, cv=cv,\n",
        "                              scoring='f1', error_score='raise')\n",
        "print(f'F1 por fold:      {scores_f1.round(3)}')\n",
        "print(f'Media: {scores_f1.mean():.3f} +/- {scores_f1.std():.3f}')\n"
    ]},

    # 10. Salvar
    {"cell_type":"markdown","metadata":{},"source":["## 10. Salvar Dataset Processado e Pipeline"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "import pickle, os\n\n",
        "os.makedirs('data/processed', exist_ok=True)\n\n",
        "# Salvar pipeline\n",
        "with open('data/processed/pipeline.pkl', 'wb') as f:\n",
        "    pickle.dump(full_pipeline, f)\n",
        "print('Pipeline salvo em data/processed/pipeline.pkl')\n\n",
        "# Salvar dataset pre-processado (antes do pipeline sklearn)\n",
        "X_save = X[SELECTED_FEATURES].copy()\n",
        "X_save['ltfu'] = y.values\n",
        "X_save.to_csv('data/processed/features_selecionadas.csv', index=False)\n",
        "print(f'Dataset salvo em data/processed/features_selecionadas.csv')\n",
        "print(f'Shape final: {X_save.shape}')\n",
        "print(f'Features: {SELECTED_FEATURES}')\n"
    ]},

    # Resumo
    {"cell_type":"markdown","metadata":{},"source":[
        "## Resumo\n\n",
        "| Etapa | Resultado |\n",
        "|-------|-----------|\n",
        "| Filtros aplicados | idade>=18, FORMA=pulmonar, SITUA_ENCE in {1,2} |\n",
        "| Colunas removidas | 100% nulas + leakage + admin |\n",
        "| Features criadas | `reingresso`, `dias_notif_trat`, `idade_anos` |\n",
        "| Valor 9 (ignorado SINAN) | Substituido por NaN |\n",
        "| Classes raras | Agrupadas em 'OUTRO' (threshold 5%) |\n",
        "| Selecao Pearson | Top correlacoes identificadas |\n",
        "| Selecao LASSO | Features com coef != 0 mantidas |\n",
        "| Encoding | RobustScaler + OrdinalEncoder + OneHotEncoder |\n",
        "| Imputacao | IterativeImputer (num) + most_frequent (cat) |\n",
        "| Baseline CV | Ver celula 9 |\n"
    ]}
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

out = "tuberculosis-ltfu-prediction-main/feature_engineering.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Notebook criado: {out} | Celulas: {len(cells)}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

# Configuração visual
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'sans-serif'

DATA_DIR = 'tuberculosis-ltgu-prediction'
OUT_DIR = 'graficos'
os.makedirs(OUT_DIR, exist_ok=True)

print("Carregando modelo e dados para gerar gráficos visuais...")
with open('tuberculosis-ltfu-prediction-main/data/processed/pipeline_final.pkl', 'rb') as f:
    pipe = pickle.load(f)

t2df = pd.read_csv(f'{DATA_DIR}/teste2.csv', low_memory=False)

def preparar(df):
    df = df.copy()
    df['reingresso'] = (df['TRATAMENTO'].astype(str) == '3').astype(int)
    for col in ['DT_NOTIFIC', 'DT_INIC_TR']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col].astype(str), format='%Y%m%d', errors='coerce')
    if 'DT_NOTIFIC' in df.columns and 'DT_INIC_TR' in df.columns:
        df['dias_notif_trat'] = (df['DT_INIC_TR'] - df['DT_NOTIFIC']).dt.days.clip(0, 365)
    else:
        df['dias_notif_trat'] = np.nan
    for col in ['AGRAVAIDS','AGRAVALCOO','AGRAVDIABE','AGRAVDOENC','AGRAVDROGA',
                'AGRAVTABAC','POP_RUA','POP_LIBER','POP_IMIG','BENEF_GOV','TRAT_SUPER']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').replace({9: np.nan, 9.0: np.nan})
    return df

t2df = preparar(t2df)
nom_cols = ['CS_SEXO','HIV','TESTE_TUBE','CULTURA_ES','BACILOSC_E','RAIOX_TORA','TRATAMENTO','CS_RACA','TEST_MOLEC']
for col in nom_cols:
    t2df[col] = t2df[col].astype(str).fillna('nan')

print("Calculando predições...")
# Pega as colunas na mesma ordem que o modelo exige (já embutido no pipeline, mas passamos todas as disponíveis)
# O scikit-learn pipeline ignora o que sobra via remainder='drop', mas por garantia, mandamos o t2df inteiro se as colunas baterem.
# Para evitar erro de ordem, extraímos a ordem correta usando feature_names_in_ se possível, mas o Pipeline lida bem com df Pandas.
y_prob = pipe.predict_proba(t2df)[:, 1]
y_pred = pipe.predict(t2df)
y_true = t2df['ltfu']

# ========================================================
# 1. Matriz de Confusão
# ========================================================
print("Gerando Matriz de Confusão...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Cura (0)', 'Abandono (1)'],
            yticklabels=['Cura (0)', 'Abandono (1)'], annot_kws={"size": 18})
plt.title('Matriz de Confusão (Teste 2)', fontsize=20, pad=20)
plt.xlabel('Predição do Modelo', fontsize=16)
plt.ylabel('Desfecho Real', fontsize=16)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/1_matriz_confusao.png', dpi=300)
plt.close()

# ========================================================
# 2. Curva ROC
# ========================================================
print("Gerando Curva ROC...")
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#1f77b4', lw=3, label=f'LightGBM (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Aleatório (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=14)
plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=14)
plt.title('Curva ROC - Capacidade de Separação', fontsize=18, pad=20)
plt.legend(loc="lower right", fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/2_curva_roc.png', dpi=300)
plt.close()

# ========================================================
# 3. Distribuição de Probabilidades (KDE Plot)
# ========================================================
print("Gerando Distribuição de Probabilidades...")
plt.figure(figsize=(10, 6))
sns.kdeplot(y_prob[y_true == 0], color="green", fill=True, label='Pacientes Curados', alpha=0.5, bw_adjust=0.8)
sns.kdeplot(y_prob[y_true == 1], color="red", fill=True, label='Pacientes que Abandonaram', alpha=0.5, bw_adjust=0.8)

plt.axvline(x=0.5, color='black', linestyle='--', lw=2, label='Limiar de Decisão (0.5)')
plt.xlabel('Probabilidade Prevista de Abandono', fontsize=14)
plt.ylabel('Densidade', fontsize=14)
plt.title('Separação das Classes pelas Probabilidades do Modelo', fontsize=18, pad=20)
plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/3_distribuicao_probs.png', dpi=300)
plt.close()

# ========================================================
# 4. Feature Importance Simples (Do LightGBM interno)
# ========================================================
print("Gerando Gráfico de Importância de Variáveis...")
try:
    lgbm_model = pipe.named_steps['clf']
    col_transformer = pipe.named_steps['pre']
    
    # Extrair os nomes após o OneHotEncoder e Imputer
    num_cols = col_transformer.transformers_[0][2]
    # O OHE pode criar multiplas colunas
    ohe = col_transformer.transformers_[1][1].named_steps['enc']
    nom_cols_original = col_transformer.transformers_[1][2]
    nom_cols_encoded = ohe.get_feature_names_out(nom_cols_original)
    
    all_feature_names = list(num_cols) + list(nom_cols_encoded)
    importances = lgbm_model.feature_importances_
    
    # Criar DF e pegar as Top 10
    feat_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)
    
    # Simplificar nomes para o gráfico ficar bonito
    feat_df['Feature'] = feat_df['Feature'].str.replace('_nan', ': Não Infor.')
    feat_df['Feature'] = feat_df['Feature'].str.replace('TRATAMENTO_3', 'Reingresso Após Abandono')
    feat_df['Feature'] = feat_df['Feature'].str.replace('TRAT_SUPER_2.0', 'TDO Não Realizado')
    feat_df['Feature'] = feat_df['Feature'].str.replace('POP_RUA_1.0', 'Situação de Rua')
    feat_df['Feature'] = feat_df['Feature'].str.replace('idade_anos', 'Idade (Anos)')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis')
    plt.title('Top 10 Fatores de Risco (Importância do LightGBM)', fontsize=18, pad=20)
    plt.xlabel('Importância Relativa (Ganho)', fontsize=14)
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/4_feature_importance.png', dpi=300)
    plt.close()
except Exception as e:
    print(f"Não foi possível gerar Feature Importance diretamente do Pipeline: {e}")


print(f"\n✅ Tudo pronto! Gráficos salvos com sucesso na pasta '{OUT_DIR}/'")
print("Você pode copiar essas imagens (.png) para o seu Word ou para os Slides da Apresentação!")

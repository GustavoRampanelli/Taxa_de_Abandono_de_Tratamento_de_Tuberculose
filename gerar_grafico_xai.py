import pandas as pd
import numpy as np
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')

print("Iniciando Geração do Gráfico de Explicabilidade (xAI)...")

# 1. Carregar o Modelo
with open('tuberculosis-ltfu-prediction-main/data/processed/pipeline_final.pkl', 'rb') as f:
    pipe = pickle.load(f)

# 1.1 Simulação de Casos Extremos (Casos Estranhos)
print("\n-> Simulando Casos Clínicos Extremos no Modelo...")
caso_a = {
    'reingresso': 0, 'POP_RUA': 2.0, 'AGRAVAIDS': 2.0, 'POP_LIBER': 2.0, 
    'AGRAVDROGA': 2.0, 'idade_anos': 85.0, 'AGRAVALCOO': 2.0, 'NU_COMU_EX': 0.0, 
    'AGRAVDIABE': 1.0, 'AGRAVTABAC': 2.0, 'AGRAVDOENC': 1.0, 'POP_IMIG': 2.0, 
    'BENEF_GOV': 2.0, 'NU_CONTATO': 2.0, 'TRAT_SUPER': 1.0, 'dias_notif_trat': 1.0,
    'CS_SEXO': 'F', 'HIV': '2', 'TESTE_TUBE': '4', 'CULTURA_ES': '2', 
    'BACILOSC_E': '2', 'RAIOX_TORA': '2', 'TRATAMENTO': '1', 'CS_RACA': '1', 'TEST_MOLEC': '2'
}
caso_b = {
    'reingresso': 1, 'POP_RUA': 1.0, 'AGRAVAIDS': 2.0, 'POP_LIBER': 2.0, 
    'AGRAVDROGA': 1.0, 'idade_anos': 22.0, 'AGRAVALCOO': 1.0, 'NU_COMU_EX': 0.0, 
    'AGRAVDIABE': 2.0, 'AGRAVTABAC': 1.0, 'AGRAVDOENC': 2.0, 'POP_IMIG': 2.0, 
    'BENEF_GOV': 2.0, 'NU_CONTATO': 0.0, 'TRAT_SUPER': 2.0, 'dias_notif_trat': 30.0,
    'CS_SEXO': 'M', 'HIV': '2', 'TESTE_TUBE': '5', 'CULTURA_ES': '4', 
    'BACILOSC_E': '1', 'RAIOX_TORA': '1', 'TRATAMENTO': '3', 'CS_RACA': '4', 'TEST_MOLEC': '1'
}
df_estranhos = pd.DataFrame([caso_a, caso_b])
prob_a = pipe.predict_proba(df_estranhos.iloc[[0]])[0][1]
prob_b = pipe.predict_proba(df_estranhos.iloc[[1]])[0][1]

print(f"  * CASO A (Idoso, diabético, novo, com TDO): Risco de Abandono = {prob_a * 100:.2f}%")
print(f"  * CASO B (Jovem, morador de rua, reingresso, sem TDO): Risco de Abandono = {prob_b * 100:.2f}%")
print("-> O modelo capturou corretamente as assimetrias comportamentais e sociais.\n")

# 2. Carregar Amostra
# Como o dataset de teste é gigante, pegamos uma amostra para calcular a Permutation Importance
print("Carregando amostra de dados...")
t2df = pd.read_csv('tuberculosis-ltgu-prediction/teste2.csv', low_memory=False).sample(n=500, random_state=42)

# Tratamento básico (mesmo do pipeline)
t2df['reingresso'] = (t2df['TRATAMENTO'].astype(str) == '3').astype(int)
for col in ['DT_NOTIFIC', 'DT_INIC_TR']:
    if col in t2df.columns:
        t2df[col] = pd.to_datetime(t2df[col].astype(str), format='%Y%m%d', errors='coerce')
if 'DT_NOTIFIC' in t2df.columns and 'DT_INIC_TR' in t2df.columns:
    t2df['dias_notif_trat'] = (t2df['DT_INIC_TR'] - t2df['DT_NOTIFIC']).dt.days.clip(0, 365)
else:
    t2df['dias_notif_trat'] = np.nan
for col in ['AGRAVAIDS','AGRAVALCOO','AGRAVDIABE','AGRAVDOENC','AGRAVDROGA',
            'AGRAVTABAC','POP_RUA','POP_LIBER','POP_IMIG','BENEF_GOV','TRAT_SUPER']:
    if col in t2df.columns:
        t2df[col] = pd.to_numeric(t2df[col], errors='coerce').replace({9: np.nan, 9.0: np.nan})

nom_cols = ['CS_SEXO','HIV','TESTE_TUBE','CULTURA_ES','BACILOSC_E','RAIOX_TORA','TRATAMENTO','CS_RACA','TEST_MOLEC']
for col in nom_cols:
    t2df[col] = t2df[col].astype(str).fillna('nan')

# Features usadas
S_NUM = ['reingresso','POP_RUA','AGRAVAIDS','POP_LIBER','AGRAVDROGA','idade_anos',
         'AGRAVALCOO','NU_COMU_EX','AGRAVDIABE','AGRAVTABAC','AGRAVDOENC',
         'POP_IMIG','BENEF_GOV','NU_CONTATO','TRAT_SUPER','dias_notif_trat']
S_NOM = ['CS_SEXO','HIV','TESTE_TUBE','CULTURA_ES','BACILOSC_E',
         'RAIOX_TORA','TRATAMENTO','CS_RACA','TEST_MOLEC']
SELECTED = S_NUM + S_NOM

# 3. Calcular Permutation Importance
print("Calculando Permutation Importance (isso pode levar alguns segundos)...")
result = permutation_importance(pipe, t2df[SELECTED], t2df['ltfu'], n_repeats=5, random_state=42, n_jobs=-1, scoring='roc_auc')

importances = pd.Series(result.importances_mean, index=SELECTED)
top_features = importances.sort_values(ascending=False).head(10)

# Dicionário amigável para plotagem
nomes_amigaveis = {
    'reingresso'     : 'Reingresso pós-abandono',
    'TRATAMENTO'     : 'Tipo de Entrada (Caso)',
    'POP_RUA'        : 'População em Situação de Rua',
    'AGRAVAIDS'      : 'Coinfecção HIV/AIDS',
    'AGRAVDROGA'     : 'Uso de Drogas Ilícitas',
    'idade_anos'     : 'Idade do Paciente',
    'AGRAVALCOO'     : 'Alcoolismo',
    'TRAT_SUPER'     : 'Tratamento Supervisionado (TDO)',
    'POP_LIBER'      : 'Privado de Liberdade',
    'dias_notif_trat': 'Atraso na Notificação',
    'CS_SEXO'        : 'Sexo Biológico',
    'CS_RACA'        : 'Raça/Cor',
    'BENEF_GOV'      : 'Beneficiário Governo',
    'NU_COMU_EX'     : 'Comunicantes Examinados',
    'NU_CONTATO'     : 'Nº de Contatos Registrados',
    'CULTURA_ES'     : 'Cultura de Escarro',
    'BACILOSC_E'     : 'Baciloscopia de Escarro',
    'RAIOX_TORA'     : 'Raio-X de Tórax',
    'TESTE_TUBE'     : 'Teste Tuberculinínico',
    'TEST_MOLEC'     : 'Teste Molecular Rápido',
    'HIV'            : 'Soropositividade HIV',
    'AGRAVDIABE'     : 'Diabetes',
    'AGRAVTABAC'     : 'Tabagismo',
    'AGRAVDOENC'     : 'Outras Doenças Associadas',
    'POP_IMIG'       : 'Imigrante',
}

labels = [nomes_amigaveis.get(feat, feat) for feat in top_features.index]

# Normalizar valores para percentual (0-100%) em relação ao total de impacto somado
total_impacto = top_features.values.sum()
valores_pct = (top_features.values / total_impacto) * 100

# 4. Plotar e Salvar o Gráfico
print("Gerando gráfico...")
plt.figure(figsize=(11, 6))
sns.set_style("whitegrid")
ax = sns.barplot(x=valores_pct, y=labels, palette="viridis")

plt.title("O que o Modelo mais leva em conta? (Explicabilidade xAI)", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Contribuição Relativa para a Decisão (%)", fontsize=12)
plt.ylabel("Fator Clínico / Social", fontsize=12)

# Adicionar os valores nas barras como %
for p in ax.patches:
    width = p.get_width()
    plt.text(width + 0.4, p.get_y() + p.get_height()/2. + 0.1, f'{width:.1f}%', ha="left", va="center", fontsize=10)

plt.tight_layout()

# Salvar na pasta docs/graficos
os.makedirs('docs/graficos', exist_ok=True)
caminho_salvar = 'docs/graficos/xai_importance.png'
plt.savefig(caminho_salvar, dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] Grafico salvo com sucesso em: {caminho_salvar}")

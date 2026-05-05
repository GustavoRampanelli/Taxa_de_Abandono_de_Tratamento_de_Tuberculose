import pandas as pd
import os

print("Carregando o arquivo feather recente...")
df = pd.read_feather('data/tuberculose_feather/TUBEBR24.dbc')

print("Filtrando os dados de 2025...")
df_2025 = df[df['DT_NOTIFIC'].astype(str).str.startswith('2025')]

# Limitar a uma amostra de 500 linhas para o Excel não ficar pesado
amostra = df_2025.sample(n=min(500, len(df_2025)), random_state=42)

caminho_saida = "amostra_2025.xlsx"
print(f"Exportando para {caminho_saida}...")
amostra.to_excel(caminho_saida, index=False)

print(f"✅ Arquivo {caminho_saida} gerado com sucesso com {len(amostra)} linhas e {len(amostra.columns)} colunas!")

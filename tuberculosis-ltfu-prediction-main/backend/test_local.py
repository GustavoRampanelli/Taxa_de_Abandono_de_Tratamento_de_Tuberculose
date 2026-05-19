import sys, os, pickle, json
import pandas as pd, numpy as np
sys.path.insert(0, '.')

# Carregar modelo
with open('pipeline_final.pkl', 'rb') as f:
    pipe = pickle.load(f)
print('Modelo carregado OK:', type(pipe.named_steps['clf']).__name__)

# Simular predicao — paciente de alto risco
row = {
    'reingresso': 1, 'POP_RUA': 1, 'AGRAVAIDS': 2, 'POP_LIBER': 2,
    'AGRAVDROGA': 1, 'idade_anos': 35, 'AGRAVALCOO': 1, 'NU_COMU_EX': 2,
    'AGRAVDIABE': 2, 'AGRAVTABAC': 1, 'AGRAVDOENC': 2, 'POP_IMIG': 2,
    'BENEF_GOV': 1, 'NU_CONTATO': 3, 'TRAT_SUPER': 2, 'dias_notif_trat': 5,
    'CS_SEXO': 'M', 'HIV': '2', 'TESTE_TUBE': '1', 'CULTURA_ES': '1',
    'BACILOSC_E': '1', 'RAIOX_TORA': '2', 'TRATAMENTO': '3',
    'CS_RACA': '4', 'TEST_MOLEC': '1'
}
df = pd.DataFrame([row])
prob = float(pipe.predict_proba(df)[0][1])
pred = int(pipe.predict(df)[0])
classe = 'Abandono' if pred == 1 else 'Cura'
nivel = 'ALTO' if prob >= 0.70 else 'MODERADO' if prob >= 0.40 else 'BAIXO'

print('Probabilidade de abandono:', round(prob, 4))
print('Classe predita:', classe)
print('Nivel de risco:', nivel)

# Simular predicao — paciente de baixo risco
row2 = dict(row)
row2['reingresso'] = 0
row2['TRATAMENTO'] = '1'
row2['POP_RUA'] = 2
row2['AGRAVDROGA'] = 2
row2['AGRAVALCOO'] = 2
row2['TRAT_SUPER'] = 1
df2 = pd.DataFrame([row2])
prob2 = float(pipe.predict_proba(df2)[0][1])
pred2 = int(pipe.predict(df2)[0])
nivel2 = 'ALTO' if prob2 >= 0.70 else 'MODERADO' if prob2 >= 0.40 else 'BAIXO'
print()
print('Paciente baixo risco:')
print('Probabilidade de abandono:', round(prob2, 4))
print('Nivel de risco:', nivel2)

# FastAPI imports
from fastapi import FastAPI
from pydantic import BaseModel
print()
print('FastAPI imports OK')
print('Tudo pronto para deploy!')

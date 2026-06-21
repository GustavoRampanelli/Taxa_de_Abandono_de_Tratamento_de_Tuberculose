import pandas as pd

base = 'tuberculosis-ltgu-prediction'

tr = pd.read_csv(f'{base}/treino.csv')
t1 = pd.read_csv(f'{base}/teste1.csv')
t2 = pd.read_csv(f'{base}/teste2.csv')

print('=== treino.csv ===')
print(f'Shape: {tr.shape}')
print(f'Colunas: {list(tr.columns)}')
if 'ltfu' in tr.columns:
    print(f'ltfu=1: {int(tr["ltfu"].sum())} ({tr["ltfu"].mean()*100:.1f}%)')
    print(f'ltfu=0: {int((tr["ltfu"]==0).sum())} ({(tr["ltfu"]==0).mean()*100:.1f}%)')

print()
print('=== teste1.csv ===')
print(f'Shape: {t1.shape}')
if 'ltfu' in t1.columns:
    print(f'ltfu=1: {int(t1["ltfu"].sum())} ({t1["ltfu"].mean()*100:.1f}%)')

print()
print('=== teste2.csv ===')
print(f'Shape: {t2.shape}')
if 'ltfu' in t2.columns:
    print(f'ltfu=1: {int(t2["ltfu"].sum())} ({t2["ltfu"].mean()*100:.1f}%)')

print()
print(f'Total registros: {len(tr)+len(t1)+len(t2):,}')
print(f'Feather: 746 MB')

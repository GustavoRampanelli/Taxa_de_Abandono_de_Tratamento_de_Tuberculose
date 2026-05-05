# Fontes e Estrutura dos Dados — Projeto LTFU-TB

## Origem dos Dados

| Campo | Detalhe |
|-------|---------|
| **Sistema** | SINAN — Sistema de Informação de Agravos de Notificação |
| **Agravo** | Tuberculose (TB) — código SINAN: `TUBEN` |
| **Gestora** | SVS — Secretaria de Vigilância em Saúde / Ministério da Saúde |
| **Acesso** | Biblioteca `pysus` (Python) via DataSUS FTP público |
| **Formato bruto** | `.dbc` (formato TabWin, compressão DBC) → convertido para `.feather` |
| **Script de download** | `baixar_tuberculose_feather.py` |
| **Script de preparo** | `data-prep.py` |
| **Arquivo consolidado** | `data/tuberculose_unificado.feather` |
| **Cobertura temporal** | Múltiplos anos disponíveis no SINAN; treino: anos < 2025; teste: ano 2025 |
| **Data de extração** | Maio / 2026 |

## Arquivos de Dados

```
tuberculosis-ltfu-prediction-main/
├── data/
│   ├── tuberculose_unificado.feather    # Dataset completo consolidado (todos os anos)
│   ├── tuberculose_feather/            # Arquivos .feather individuais por ano
│   └── treino.csv                      # Gerado por data-prep.py (anos < 2025)
│   └── teste1.csv                      # Gerado por data-prep.py (1ª metade de 2025)
│   └── teste2.csv                      # Gerado por data-prep.py (2ª metade de 2025)
├── amostra_2025.xlsx                   # Amostra aleatória de 500 registros de 2025
└── docs-sinan/
    ├── TUBEN_DIC_DADOS.pdf             # Dicionário oficial das variáveis
    ├── TUBEN_CADERNO_ANALISE.pdf       # Caderno de análise SINAN-TB
    ├── TUBEN_FICHA.pdf                 # Ficha de notificação
    ├── TUBEN_FICHA_ACOMP.pdf          # Ficha de acompanhamento
    └── TUBEN_INSTRUCIONAL.pdf          # Instrucional de preenchimento
```

> **Nota:** Os arquivos `treino.csv`, `teste1.csv` e `teste2.csv` são gerados executando
> `data-prep.py` dentro do container Docker (ver README.md). Também disponíveis em:
> https://drive.google.com/drive/folders/13BOVwEUAK8QolcCXbvhkNtaSci3sECEd?usp=sharing

## Dimensões da Amostra (amostra_2025.xlsx)

| Métrica | Valor |
|---------|-------|
| Registros | 500 |
| Variáveis | 94 |
| Colunas 100% nulas | 10 (drop imediato) |
| Colunas > 80% nulas | 8 (avaliar) |
| Período de notificação | 2024–2025 |

## Divisão Treino / Teste

A divisão é **temporal** (sem data leakage):

```
Dataset completo (tuberculose_unificado.feather)
│
├── DT_NOTIFIC.year < 2025  →  treino.csv
│
└── DT_NOTIFIC.year == 2025  →  ordenado por data
    ├── primeira metade  →  teste1.csv
    └── segunda metade   →  teste2.csv
```

**Justificativa:** A divisão temporal evita contaminação futura no treino e simula
o cenário real de produção, onde o modelo é treinado com dados históricos e aplicado
a notificações novas.

## Variável-Alvo

| Coluna SINAN | Valor | Significado | Label ML |
|-------------|-------|-------------|----------|
| `SITUA_ENCE` | 1 | Cura | `ltfu = 0` |
| `SITUA_ENCE` | 2 | Abandono | `ltfu = 1` |
| `SITUA_ENCE` | 3 | Transferência | Excluído |
| `SITUA_ENCE` | 4 | Óbito por outras causas | Excluído |
| `SITUA_ENCE` | 5 | Óbito por TB | Excluído |
| `SITUA_ENCE` | 7 | TB-DR (resistente) | Excluído |
| `SITUA_ENCE` | 8 | Mudança de esquema | Excluído |
| `SITUA_ENCE` | 10 | Falência | Excluído |
| `SITUA_ENCE` | NaN | Sem encerramento | Excluído |

> O modelo prediz a probabilidade de abandono (`ltfu=1`) **antes** do encerramento do caso,
> com base apenas em variáveis disponíveis no momento da notificação.

## Filtros Aplicados pelo data-prep.py

| Filtro | Justificativa |
|--------|--------------|
| `idade_anos >= 18` | Foco em adultos; pediatria tem protocolo distinto |
| `FORMA == "1"` | Somente TB pulmonar; extrapulmonar tem perfil diferente |
| `TRATAMENTO != "6"` | Excluir casos pós-óbito (dado incoerente) |
| `POP_RUA` ∈ {1,2,null} | Remover valores incoerentes (ex: 9=ignorado como filtro) |
| `POP_LIBER` ∈ {1,2,null} | Idem |
| `POP_IMIG` ∈ {1,2,null} | Idem |
| `CS_GESTANT` ∈ {5,6,9,null} | Manter apenas não-gestantes e ignorados |
| `TEST_MOLEC != "2"` | Excluir resultados inválidos de GeneXpert |
| `TEST_SENSI` ∉ {1,2,3,4} | Excluir casos com resistência confirmada (perfil diferente) |
| `SITUA_ENCE` ∈ {1,2} | Manter apenas cura e abandono para classificação binária |

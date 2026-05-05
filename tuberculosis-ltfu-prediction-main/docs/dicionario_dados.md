# Dicionário de Dados — SINAN Tuberculose (TUBEN)

**Fonte oficial:** `docs-sinan/TUBEN_DIC_DADOS.pdf`  
**Amostra de referência:** `amostra_2025.xlsx` — 500 registros, 94 variáveis  
**Responsável:** Membro 3  
**Versão:** 1.0 — Macro 2

---

## Legenda de Status

| Status | Significado |
|--------|-------------|
| ✅ Relevante | Manter no pipeline de ML |
| ❌ Drop | Remover: 100% nulo ou data leakage |
| ⚠️ Avaliar | Alta missingness ou uso condicional |
| 🔑 Alvo | Variável-alvo (gerada a partir dela) |
| 🔒 Admin | Campo administrativo/identificador, sem valor preditivo |
| 📅 Data | Campo de data — pode gerar features derivadas |

---

## Grupo 1 — Identificação e Notificação

| Coluna | Descrição | Tipo | Valores | % Nulo | Status |
|--------|-----------|------|---------|--------|--------|
| `TP_NOT` | Tipo de notificação (2=individual) | Categórica | 2 | 0% | 🔒 Admin |
| `ID_AGRAVO` | Código do agravo (sempre A15–A19 TB) | Texto | A150 | 0% | 🔒 Admin |
| `DT_NOTIFIC` | Data de notificação | Data | YYYYMMDD | 0% | 📅 Data |
| `NU_ANO` | Ano de notificação | Numérica | 2024, 2025 | 0% | 🔒 Admin |
| `SG_UF_NOT` | UF da unidade notificadora | Categórica | Siglas UF | 0% | ✅ Relevante |
| `ID_MUNICIP` | Município da unidade notificadora (IBGE 6d) | Categórica | Cód. 6 dígitos | 0% | ✅ Relevante |
| `ID_REGIONA` | Regional de saúde da unidade | Categórica | Cód. regional | 38,8% | ⚠️ Avaliar |
| `DT_DIAG` | Data do diagnóstico | Data | YYYYMMDD | 0% | 📅 Data |
| `NDUPLIC_N` | Indicador de duplicidade | Binária | 1=sim, 2=não | 0% | 🔒 Admin |
| `IN_VINCULA` | Vinculação a outro agravo | Binária | 1=sim, 2=não | 0% | 🔒 Admin |
| `DT_DIGITA` | Data de digitação no SINAN | Data | YYYYMMDD | 0% | 🔒 Admin |
| `DT_TRANSUS` | Data de transferência US→SM | Data | YYYYMMDD | 0% | 🔒 Admin |
| `DT_TRANSDM` | Data de transferência DM→SM | Data | YYYYMMDD | 0% | 🔒 Admin |
| `DT_TRANSSM` | Data de transferência SM→SR | Data | YYYYMMDD | 0% | 🔒 Admin |
| `DT_TRANSRM` | Data de transferência RM→SVS | Data | YYYYMMDD | 100% | ❌ Drop |
| `DT_TRANSRS` | Data de transferência RS | Data | YYYYMMDD | 0% | 🔒 Admin |
| `DT_TRANSSE` | Data de transferência SE | Data | YYYYMMDD | 0% | 🔒 Admin |
| `CS_FLXRET` | Fluxo retorno | Categórica | — | 100% | ❌ Drop |
| `FLXRECEBI` | Fluxo recebido | Categórica | — | 0% | 🔒 Admin |
| `MIGRADO_W` | Migrado de outro sistema | Binária | 1/2 | 0% | 🔒 Admin |

---

## Grupo 2 — Dados Demográficos do Paciente

| Coluna | Descrição | Tipo | Valores | % Nulo | Status |
|--------|-----------|------|---------|--------|--------|
| `ANO_NASC` | Ano de nascimento | Numérica | YYYY | 0% | ✅ Relevante (derivar idade) |
| `NU_IDADE_N` | Idade codificada SINAN | Texto | Prefixo+valor (ex: 4060=60a) | 0% | ✅ Relevante (→ `idade_anos`) |
| `CS_SEXO` | Sexo | Binária | M / F | 0% | ✅ Relevante |
| `CS_GESTANT` | Situação gestacional | Categórica | 1=1ºtri, 2=2ºtri, 3=3ºtri, 4=IG, 5=não, 6=não se aplica, 9=ignorado | 0% | ⚠️ Avaliar (filtrado: mantém 5,6,9) |
| `CS_RACA` | Raça/cor | Categórica | 1=branca, 2=preta, 3=amarela, 4=parda, 5=indígena, 9=ignorado | 1,6% | ✅ Relevante |
| `CS_ESCOL_N` | Escolaridade | Ordinal | 0=analfabeto, 1=1ª–4ª, 2=5ª–8ª, 3=médio, 4=superior, 5=n/a, 9=ignorado | 8% | ✅ Relevante |
| `ID_OCUPA_N` | Ocupação (CBO) | Texto | Código CBO | 0% | ⚠️ Avaliar (alta cardinalidade) |

---

## Grupo 3 — Residência

| Coluna | Descrição | Tipo | Valores | % Nulo | Status |
|--------|-----------|------|---------|--------|--------|
| `SG_UF` | UF de residência do paciente | Categórica | Siglas UF | 0% | ✅ Relevante |
| `ID_MN_RESI` | Município de residência (IBGE 6d) | Categórica | Cód. 6 dígitos | 0% | ✅ Relevante |
| `ID_RG_RESI` | Regional de saúde de residência | Categórica | Cód. regional | 0% | ⚠️ Avaliar (redundante com ID_REGIONA) |
| `ID_PAIS` | País de residência | Categórica | Cód. país | 0% | ⚠️ Avaliar (maioria Brasil) |

---

## Grupo 4 — Tipo de Tratamento e Institucionalização

| Coluna | Descrição | Tipo | Valores | % Nulo | Status |
|--------|-----------|------|---------|--------|--------|
| `TRATAMENTO` | Tipo de entrada no tratamento | Categórica | 1=caso novo, 2=recidiva, 3=reingresso após abandono, 4=não sabe, 5=transferência, 6=pós-óbito | 0% | ✅ Relevante |
| `INSTITUCIO` | Tipo de instituição | Categórica | — | 0% | ⚠️ Avaliar |

> ⚠️ **Atenção:** `TRATAMENTO == 3` (reingresso após abandono anterior) é preditor fortíssimo de novo abandono. Manter sem imputação.

---

## Grupo 5 — Dados Clínicos / Diagnóstico

| Coluna | Descrição | Tipo | Valores | % Nulo | Status |
|--------|-----------|------|---------|--------|--------|
| `RAIO_TORA` | Raio-X de tórax | Categórica | 1=normal, 2=suspeito, 3=outra doença, 4=n/r | 0% | ✅ Relevante |
| `TESTE_TUBE` | Teste tuberculínico (PPD) | Categórica | 1=não reator, 2=reator fraco, 3=reator forte, 4=n/r | 0% | ✅ Relevante |
| `FORMA` | Forma clínica | Categórica | 1=pulmonar, 2=extrapulmonar, 3=pulmonar+extrapulmonar | 0% | ✅ Relevante (filtrado: apenas 1) |
| `EXTRAPU1_N` | 1ª localização extrapulmonar | Categórica | Cód. localização | 0% | ⚠️ Avaliar (irrelevante após filtro FORMA=1) |
| `EXTRAPU2_N` | 2ª localização extrapulmonar | Categórica | Cód. localização | >99% | ❌ Drop |

---

## Grupo 6 — Comorbidades

| Coluna | Descrição | Tipo | Valores | % Nulo | Status |
|--------|-----------|------|---------|--------|--------|
| `AGRAVAIDS` | Agravo AIDS | Binária | 1=sim, 2=não, 9=ignorado | 2,8% | ✅ Relevante |
| `AGRAVALCOO` | Agravo alcoolismo | Binária | 1=sim, 2=não, 9=ignorado | 3,0% | ✅ Relevante |
| `AGRAVDIABE` | Agravo diabetes | Binária | 1=sim, 2=não, 9=ignorado | 3,2% | ✅ Relevante |
| `AGRAVDOENC` | Agravo doença mental | Binária | 1=sim, 2=não, 9=ignorado | 3,6% | ✅ Relevante |
| `AGRAVOUTRA` | Outros agravos | Binária | 1=sim, 2=não, 9=ignorado | 35,4% | ⚠️ Avaliar |
| `AGRAVDROGA` | Agravo uso de drogas | Binária | 1=sim, 2=não, 9=ignorado | 3,2% | ✅ Relevante |
| `AGRAVTABAC` | Agravo tabagismo | Binária | 1=sim, 2=não, 9=ignorado | 3,0% | ✅ Relevante |
| `HIV` | Resultado do teste HIV | Categórica | 1=positivo, 2=negativo, 3=em andamento, 4=não realizado | 0% | ✅ Relevante |
| `ANT_RETRO` | Uso de antirretroviral | Binária | 1=sim, 2=não, 9=ignorado | 70,8% | ⚠️ Avaliar (alta missingness) |

> ⚠️ **Atenção:** Valor `9` em todas as binárias significa **"ignorado/não informado"** no SINAN — tratar como categoria própria, **não** como valor numérico.

---

## Grupo 7 — Exames Laboratoriais

| Coluna | Descrição | Tipo | Valores | % Nulo | Status |
|--------|-----------|------|---------|--------|--------|
| `BACILOSC_E` | Baciloscopia de escarro (diagnóstico) | Categórica | 1=positivo, 2=negativo, 3=não realizado, 4=em andamento | 0% | ✅ Relevante |
| `BACILOS_E2` | Baciloscopia escarro 2 | Categórica | — | 100% | ❌ Drop |
| `BACILOSC_O` | Baciloscopia de outro material | Categórica | 1=positivo, 2=negativo, 3=não realizado | 79% | ⚠️ Avaliar |
| `CULTURA_ES` | Cultura de escarro | Categórica | 1=positivo, 2=negativo, 3=não realizado, 4=em andamento | 0% | ✅ Relevante |
| `CULTURA_OU` | Cultura de outro material | Categórica | — | 84% | ⚠️ Avaliar |
| `HISTOPATOL` | Histopatológico | Categórica | 1=baar+, 2=compatível TB, 3=não compatível, 4=n/r, 5=n/a | 9,4% | ⚠️ Avaliar |
| `TEST_MOLEC` | Teste molecular (GeneXpert) | Categórica | 1=detectado sensível, 2=detectado resistente, 3=não detectado, 4=inconclusivo, 5=n/r | 5,8% | ✅ Relevante (filtrado: ≠2) |
| `TEST_SENSI` | Teste de sensibilidade | Categórica | 1–4=resistências, 5=sensível, 6=em andamento, 7=n/r | 44,6% | ⚠️ Avaliar (filtrado: exclui 1–4) |

---

## Grupo 8 — Medicamentos (Todos 100% Nulos → DROP)

| Coluna | Descrição | % Nulo | Status |
|--------|-----------|--------|--------|
| `RIFAMPICIN` | Rifampicina | 100% | ❌ Drop |
| `ISONIAZIDA` | Isoniazida | 100% | ❌ Drop |
| `ETAMBUTOL` | Etambutol | 100% | ❌ Drop |
| `ESTREPTOMI` | Estreptomicina | 100% | ❌ Drop |
| `PIRAZINAMI` | Pirazinamida | 100% | ❌ Drop |
| `ETIONAMIDA` | Etionamida | 100% | ❌ Drop |
| `OUTRAS` | Outros medicamentos | 100% | ❌ Drop |
| `DOENCA_TRA` | Doença que transmite | 100% | ❌ Drop |
| `DT_MUDANCA` | Data de mudança de esquema | 100% | ❌ Drop |
| `SITUA_9_M` | Situação aos 9 meses | 100% | ❌ Drop |
| `SITUA_12_M` | Situação aos 12 meses | 100% | ❌ Drop |

---

## Grupo 9 — Tratamento e Acompanhamento

| Coluna | Descrição | Tipo | Valores | % Nulo | Status |
|--------|-----------|------|---------|--------|--------|
| `DT_INIC_TR` | Data de início do tratamento | Data | YYYYMMDD | 6,4% | 📅 Data (derivar dias até início) |
| `TRAT_SUPER` | Tratamento supervisionado — TDO | Binária | 1=sim, 2=não, 9=ignorado | 79% | ⚠️ Avaliar (alta missingness) |
| `NU_CONTATO` | Nº de contatos registrados | Numérica | 0–N | 5,6% | ✅ Relevante |
| `BACILOSC_1` | Baciloscopia controle mês 1 | Categórica | 1=positivo, 2=negativo, 3=n/r, 4=n/a | 39,6% | ⚠️ Avaliar (dado de acompanhamento — leakage potencial) |
| `BACILOSC_2` | Baciloscopia controle mês 2 | Categórica | idem | 46,4% | ⚠️ Avaliar |
| `BACILOSC_3` | Baciloscopia controle mês 3 | Categórica | idem | 52,2% | ⚠️ Avaliar |
| `BACILOSC_4` | Baciloscopia controle mês 4 | Categórica | idem | 59,2% | ⚠️ Avaliar |
| `BACILOSC_5` | Baciloscopia controle mês 5 | Categórica | idem | 63% | ⚠️ Avaliar |
| `BACILOSC_6` | Baciloscopia controle mês 6 | Categórica | idem | 65,4% | ⚠️ Avaliar |
| `TRATSUP_AT` | TDO da unidade de acompanhamento | Binária | 1=sim, 2=não, 9=ignorado | 34,4% | ⚠️ Avaliar |
| `BAC_APOS_6` | Baciloscopia após 6 meses | Categórica | — | 81,2% | ❌ Drop (pós-tratamento) |
| `NU_COMU_EX` | Nº de comunicantes examinados | Numérica | 0–N | 31,6% | ⚠️ Avaliar |

> ⚠️ **Atenção ao data leakage:** `BACILOSC_1` a `BACILOSC_6` são exames realizados **durante** o tratamento. Usá-los como features implica que o modelo só pode ser aplicado após os respectivos meses — isso restringe o uso clínico (predição na admissão). Excluir na versão principal; avaliar em modelo de predição tardia.

---

## Grupo 10 — Unidade de Acompanhamento

| Coluna | Descrição | Tipo | Valores | % Nulo | Status |
|--------|-----------|------|---------|--------|--------|
| `SG_UF_AT` | UF da unidade de acompanhamento | Categórica | Siglas UF | 0,8% | ✅ Relevante |
| `ID_MUNIC_A` | Município da unidade de acompanhamento | Categórica | Cód. 6 dígitos | 0,8% | ✅ Relevante |
| `DT_NOTI_AT` | Data de notificação na unidade de acompanhamento | Data | YYYYMMDD | 21,8% | 📅 Data |
| `SG_UF_2` | UF da 2ª unidade | Categórica | Siglas UF | 22% | ⚠️ Avaliar |
| `ID_MUNIC_2` | Município da 2ª unidade | Categórica | Cód. 6 dígitos | 22,2% | ⚠️ Avaliar |
| `TPUNINOT` | Tipo de unidade notificadora | Categórica | Vários tipos de UBS/hospital | 26,8% | ⚠️ Avaliar |

---

## Grupo 11 — Populações Vulneráveis

| Coluna | Descrição | Tipo | Valores | % Nulo | Status |
|--------|-----------|------|---------|--------|--------|
| `POP_LIBER` | População privada de liberdade | Binária | 1=sim, 2=não, 9=ignorado | 2,8% | ✅ Relevante |
| `POP_RUA` | População em situação de rua | Binária | 1=sim, 2=não, 9=ignorado | 3,4% | ✅ Relevante |
| `POP_SAUDE` | Profissional de saúde | Categórica | 1=sim, 2=não, 3=n/a, 9=ignorado | 3,2% | ⚠️ Avaliar |
| `POP_IMIG` | Imigrante | Categórica | 1=sim, 2=não, 3=refugiado, 9=ignorado | 3,4% | ✅ Relevante |
| `BENEF_GOV` | Beneficiário de programa social (Bolsa Família, etc.) | Binária | 1=sim, 2=não, 9=ignorado | 27,4% | ✅ Relevante |

---

## Grupo 12 — Transferência

| Coluna | Descrição | Tipo | Valores | % Nulo | Status |
|--------|-----------|------|---------|--------|--------|
| `TRANSF` | Transferência para outra UF | Categórica | 1=sim, 2=não | 91% | ⚠️ Avaliar |
| `UF_TRANSF` | UF de destino da transferência | Categórica | Cód. UF | 93,6% | ❌ Drop (quase vazio) |
| `MUN_TRANSF` | Município de destino | Categórica | Cód. município | 94% | ❌ Drop (quase vazio) |

---

## Grupo 13 — Encerramento (**🔑 Alvo**)

| Coluna | Descrição | Tipo | Valores | % Nulo | Status |
|--------|-----------|------|---------|--------|--------|
| `SITUA_ENCE` | Situação de encerramento | Categórica | 1=cura, 2=abandono, 3=transferência, 4=óbito outras causas, 5=óbito TB, 7=TB-DR, 8=mudança esquema, 10=falência, NaN=sem encerramento | 36,8% | 🔑 Alvo |
| `DT_ENCERRA` | Data de encerramento | Data | YYYYMMDD | 41,8% | ❌ Drop (leakage — ocorre após desfecho) |

---

## Resumo Executivo

| Status | Quantidade | Exemplos |
|--------|-----------|---------|
| ✅ Relevante (manter) | ~30 | CS_SEXO, AGRAVAIDS, POP_RUA, TRATAMENTO, HIV |
| ❌ Drop imediato | ~15 | RIFAMPICIN, ISONIAZIDA, BACILOS_E2, DT_ENCERRA |
| ⚠️ Avaliar | ~30 | TRAT_SUPER, BACILOSC_1–6, ID_REGIONA, ANT_RETRO |
| 🔒 Admin | ~15 | TP_NOT, ID_AGRAVO, DT_DIGITA, NDUPLIC_N |
| 🔑 Alvo | 1 | SITUA_ENCE → `ltfu` |
| 📅 Data | 5 | DT_NOTIFIC, DT_DIAG, DT_INIC_TR |

### Features de Alta Prioridade para o Modelo

Com base na literatura e nos dados disponíveis, as variáveis com maior poder discriminativo esperado:

1. `TRATAMENTO` = 3 (reingresso pós-abandono) — preditor direto de recidiva
2. `POP_RUA` = 1 — associação forte com abandono na literatura
3. `AGRAVAIDS` / `HIV` positivo — comorbidade crítica
4. `AGRAVALCOO` / `AGRAVDROGA` — determinantes sociais do abandono
5. `CS_ESCOL_N` — escolaridade como proxy de vulnerabilidade
6. `BENEF_GOV` — acesso a benefícios sociais
7. `TRAT_SUPER` — TDO é o principal fator protetor do abandono
8. `NU_CONTATO` — engajamento da rede de apoio
9. `CS_SEXO` — homens têm maior taxa de abandono
10. `idade_anos` (derivada de `NU_IDADE_N`) — jovens adultos com maior risco

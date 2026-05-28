# Dicionário de Dados - SINAN Tuberculose

Este documento descreve as principais variáveis utilizadas na modelagem de predição de abandono do tratamento de tuberculose, derivadas dos microdados originais do Sistema de Informação de Agravos de Notificação (SINAN/DataSUS).

## Variável Alvo (Target)
A variável alvo original do SINAN é `SITUA_ENCE` (Situação de Encerramento).
*   **1** - Cura (utilizada como classe Negativa `0` na predição)
*   **2** - Abandono (utilizada como classe Positiva `1` na predição)
*   *(Os demais valores como Óbito, Transferência e Mudança de Esquema foram filtrados da base de treinamento para preservar o escopo preditivo).*

## Características Demográficas
*   **`CS_SEXO`**: Sexo do paciente (`M` = Masculino, `F` = Feminino).
*   **`NU_IDADE_N`**: Idade do paciente (Formato SINAN: o prefixo `4` indica que o valor está em anos. Exemplo: `4025` = 25 anos). No pipeline, esta coluna é processada para `idade_anos`.
*   **`CS_RACA`**: Raça/Cor (`1`=Branca, `2`=Preta, `3`=Amarela, `4`=Parda, `5`=Indígena, `9`=Ignorado).
*   **`CS_ESCOL_N`**: Nível de escolaridade.

## Dados Clínicos e Laboratoriais
*   **`FORMA`**: Forma clínica da Tuberculose (`1`=Pulmonar, `2`=Extrapulmonar, `3`=Mista).
*   **`HIV`**: Resultado do teste anti-HIV (`1`=Positivo, `2`=Negativo, `3`=Em andamento, `4`=Não realizado).
*   **`TESTE_TUBE`**: Teste Tuberculínico (`1` a `3` = Faixas em mm, `4`=Não reator, `5`=Não realizado).
*   **`CULTURA_ES`**: Cultura de escarro.
*   **`BACILOSC_E`**: Baciloscopia de escarro de acompanhamento.
*   **`TEST_MOLEC`**: Teste Rápido Molecular (TRM-TB).
*   **`RAIOX_TORA`**: Radiografia do Tórax (`1`=Suspeito, `2`=Normal, `3`=Outra patologia, `4`=Não realizado).

## Vulnerabilidades Sociais (Binárias: 1=Sim, 2=Não, 9=Ignorado)
*   **`POP_RUA`**: População em situação de rua.
*   **`POP_LIBER`**: População privada de liberdade.
*   **`POP_IMIG`**: Imigrantes.
*   **`BENEF_GOV`**: Beneficiário de programa de transferência de renda do governo.

## Comorbidades e Agravos (Binárias: 1=Sim, 2=Não, 9=Ignorado)
*   **`AGRAVAIDS`**: Coinfecção por AIDS.
*   **`AGRAVALCOO`**: Alcoolismo.
*   **`AGRAVDROGA`**: Uso de drogas ilícitas.
*   **`AGRAVTABAC`**: Tabagismo.
*   **`AGRAVDIABE`**: Diabetes mellitus.
*   **`AGRAVDOENC`**: Outras doenças agravantes crônicas.

## Dados do Tratamento e Acompanhamento
*   **`TRATAMENTO`**: Tipo de entrada.
    *   `1`: Caso Novo
    *   `2`: Recidiva
    *   `3`: Reingresso após abandono (Fator de altíssimo risco).
    *   `4`: Não sabe
    *   `5`: Transferência
*   **`TRAT_SUPER`**: Tratamento Diretamente Observado (TDO) realizado? (`1`=Sim, `2`=Não, `9`=Ignorado).
*   **`NU_CONTATO`**: Número de contatos identificados.
*   **`NU_COMU_EX`**: Número de comunicantes examinados.
*   **`DT_NOTIFIC` e `DT_INIC_TR`**: Datas de notificação e de início do tratamento, utilizadas para calcular o atraso de notificação (`dias_notif_trat`).

---
> **Nota de Imputação e Limpeza**: As variáveis marcadas com o valor `9` (Ignorado) no SINAN foram tratadas via pipeline. Em algoritmos de árvore ou lineares, valores `9` não representam magnitude numérica, sendo substituídos por `NaN` no pré-processamento numérico, ou convertidos apropriadamente no OneHotEncoder de variáveis nominais.

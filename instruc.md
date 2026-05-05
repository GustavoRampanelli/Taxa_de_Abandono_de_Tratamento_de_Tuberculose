Passos do Trabalho
Preparar os dados
Pré-selecionar preditores relevantes (critério de relevância aparente)
Verificar se os preditores são relevantes para o alvo (análise estatística): coeficiente de correlação de Pearson (dataframe.corr()) ou treinar uma regressão (com LASSO).
Preparar os dados: 
Variáveis numéricas: RobustScaler
Variáveis categóricas: OneHotEncoder, OrdinalEncoder ou TargetEncoder
Missings: depende muito do significado dos valores ausentes. Algumas boas estratégias são: (1) deletar linhas se forem poucos missings, (2) usar IterativeImputer
Classes raras: cuidar com as variáveis preditoras categóricas com valores muito infrequentes. 
Data leakage: cuidar para não incluir nos preditores variáveis que só estão presentes na hora do alvo ou posteriormente. 
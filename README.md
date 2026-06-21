# 🩺 PrevTB: Predição de Abandono do Tratamento de Tuberculose (LTFU-Predict)

**Sistema Inteligente de Suporte à Decisão Clínica — Projeto Acadêmico (Atitus Educação)**

![PrevTB Overview](docs/images/bg_tuberculose.jpg)

## 📌 Contexto de Saúde Pública
A Tuberculose (TB) continua sendo um grave problema de saúde pública no Brasil. O sucesso no combate à doença depende diretamente da adesão ao tratamento, que dura em média seis meses. A interrupção prematura — fenômeno conhecido como **Loss to Follow-Up (LTFU)** ou **Abandono** — além de agravar a saúde do paciente, induz resistência bacteriana aos fármacos de primeira linha (MDR-TB).

O **PrevTB** foi desenvolvido para analisar os registros iniciais dos pacientes no SINAN (Sistema de Informação de Agravos de Notificação) e prever o risco de abandono com altíssima precisão. Isso permite que a vigilância epidemiológica e os postos de saúde direcionem recursos escassos (como o Tratamento Diretamente Observado - TDO e apoio socioeconômico) aos perfis mais vulneráveis logo no diagnóstico.

---

## 📊 Métricas e Resultados do Modelo

A base de dados foi treinada utilizando mais de 500 mil registros oficiais do SINAN. A modelagem utilizou **LightGBM** balanceado, focado em minimizar o erro do sistema público ao negligenciar pacientes de alto risco.

*   **ROC-AUC (Capacidade Global de Acerto):** `0.859` (85.9%)
*   **F1-Score (Equilíbrio e Precisão):** `0.865` (86.5%)
*   **Sensibilidade / Recall (Captura de Risco):** `0.880` (88.0%)

> **Nota:** O alto Recall garante que, de cada 100 pacientes propensos ao abandono, o sistema emite alertas para 88. A lógica interna não é uma "Caixa Preta": o sistema incorpora um módulo visual **xAI (Explicabilidade)** via *Permutation Importance* para expor o peso de fatores como Situação de Rua, HIV, Idade Jovem e Reingresso por abandono anterior.

---

## 💻 Arquitetura do Sistema

O projeto adota uma arquitetura modular moderna (Machine Learning Serverless):

1. **Frontend (SPA - Single Page Application)**
   - Hospedado no GitHub Pages (pasta `docs/`).
   - Escrito em Vanilla JS, HTML5 e CSS3 (Design Responsivo).
   - Comunica-se via FETCH API com o servidor na nuvem.
2. **Backend (API de Inferência - FastAPI)**
   - Servidor Python construído com **FastAPI** (`tuberculosis-ltfu-prediction-main/backend/app.py`).
   - Carrega o pipeline serializado (`pipeline_final.pkl`).
   - Aceita payload JSON e retorna classe, risco, probabilidade e fatores contribuintes.
3. **Pipeline de Dados & Treinamento**
   - Scripts modulares independentes (`treinar_modelos.py`, `otimizar_modelo.py`, `gerar_graficos_avaliacao.py`).

---

## 🚀 Como Executar Localmente

### 1. Iniciar a API (Backend)
Certifique-se de que o Python 3.10+ está instalado e instale as dependências.
```bash
# Navegue até a pasta do backend
cd tuberculosis-ltfu-prediction-main/backend

# Instale os requisitos
pip install fastapi uvicorn pandas lightgbm scikit-learn

# Execute o servidor localmente
python app.py
```
A API rodará em `http://localhost:8000`. A documentação nativa do Swagger pode ser vista em `http://localhost:8000/docs`.

### 2. Iniciar a Interface (Frontend)
Você não precisa de um servidor Node/NPM. Basta abrir o arquivo `docs/index.html` em qualquer navegador.
*(Dica: Ao rodar localmente, você pode precisar alterar a URL base do FETCH no arquivo `app.js` de `https://...` para `http://localhost:8000`).*

---

## 📡 Documentação da API

O servidor expõe o endpoint de predição via **POST**.

**Endpoint:** `POST /predict`

**Payload de Exemplo (JSON):**
```json
{
  "idade_anos": 35,
  "CS_SEXO": "M",
  "CS_RACA": "4",
  "TRATAMENTO": "3",
  "BACILOSC_E": "1",
  "CULTURA_ES": "1",
  "RAIOX_TORA": "2",
  "TESTE_TUBE": "1",
  "TEST_MOLEC": "1",
  "HIV": "2",
  "AGRAVAIDS": 2,
  "AGRAVALCOO": 1,
  "AGRAVDIABE": 2,
  "AGRAVDOENC": 2,
  "AGRAVDROGA": 2,
  "AGRAVTABAC": 1,
  "POP_RUA": 1,
  "POP_LIBER": 2,
  "POP_IMIG": 2,
  "BENEF_GOV": 1,
  "TRAT_SUPER": 2,
  "NU_CONTATO": 3,
  "NU_COMU_EX": 2,
  "dias_notif_trat": 5
}
```

**Resposta de Sucesso:**
```json
{
  "probabilidade_abandono": 0.8412,
  "classe": "Abandono",
  "nivel_risco": "ALTO",
  "fatores_risco": [
    "Situação de rua — vulnerabilidade social crítica",
    "Sem TDO (tratamento supervisionado) — importante fator protetor ausente"
  ],
  "recomendacao": "AÇÃO URGENTE: Iniciar ou intensificar TDO imediatamente...",
  "aviso": "Esta predição é uma ferramenta de apoio clínico baseada em dados populacionais..."
}
```

---

## 👥 Equipe de Desenvolvimento

Projeto desenvolvido para conclusão do módulo de especialização da **Atitus Educação**.

- **Gustavo Rampanelli** — Modelagem de ML, Infraestrutura Cloud e API Backend
- **Vinicius Gehring Capellari** — Líder do Projeto, Documentação Científica e Métricas
- **Victor Quadri** — Validação e Engenharia de Features Clínicas
- **João Vitor Burati** — Estatística Multivariada & Validação SINAN
- **Laura Cemin Iora** — Arquitetura de Produção e Frontend Web

---
*© 2026 PrevTB. Licença de Uso Acadêmico.*

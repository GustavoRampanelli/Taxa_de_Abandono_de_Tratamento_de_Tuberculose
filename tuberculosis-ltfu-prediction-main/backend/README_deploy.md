# Deploy no Render — Backend FastAPI LTFU-TB

## Pré-requisitos
- Conta gratuita em [render.com](https://render.com)
- Repositório no GitHub com a pasta `backend/` incluindo o `pipeline_final.pkl`

## Passo a Passo

### 1. Garantir que o pipeline_final.pkl está no repositório
```bash
# Na raiz do projeto
git add tuberculosis-ltfu-prediction-main/backend/pipeline_final.pkl
git add tuberculosis-ltfu-prediction-main/backend/app.py
git add tuberculosis-ltfu-prediction-main/backend/requirements.txt
git commit -m "feat: adicionar backend FastAPI com modelo LightGBM"
git push
```
> ⚠️ O arquivo `.pkl` tem ~5MB — verifique se não está no `.gitignore`

### 2. Criar Web Service no Render
1. Acesse [dashboard.render.com](https://dashboard.render.com)
2. Clique em **"New +"** → **"Web Service"**
3. Conecte seu repositório GitHub
4. Configure:

| Campo | Valor |
|-------|-------|
| **Name** | `ltfu-tb-api` |
| **Region** | Oregon (US West) ou qualquer |
| **Branch** | `main` |
| **Root Directory** | `tuberculosis-ltfu-prediction-main/backend` |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | `Free` |

5. Clique em **"Create Web Service"**

### 3. URL gerada pelo Render
Após o deploy (3–5 minutos), a URL será:
```
https://ltfu-tb-api.onrender.com
```

### 4. Testar a API
```bash
# Health check
curl https://ltfu-tb-api.onrender.com/health

# Predição de exemplo
curl -X POST https://ltfu-tb-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "idade_anos": 35,
    "CS_SEXO": "M",
    "TRATAMENTO": "3",
    "POP_RUA": 1,
    "AGRAVDROGA": 1,
    "NU_CONTATO": 3
  }'
```

### 5. Documentação interativa
Acesse: `https://ltfu-tb-api.onrender.com/docs`

## CORS
O CORS está configurado com `allow_origins=["*"]`.
Após confirmar a URL do GitHub Pages, restrinja para:
```python
allow_origins=["https://SEU_USUARIO.github.io"]
```

## Atenção — Cold Start
No plano gratuito do Render, a API dorme após 15 min sem uso.
A primeira requisição pode levar **~30 segundos**.
Implemente no frontend uma mensagem de "Aguardando servidor acordar...".

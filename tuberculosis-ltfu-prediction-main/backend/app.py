"""
Backend FastAPI — Predição de Abandono do Tratamento de TB (LTFU)
Universidade / Nanodegree — 2026

Endpoints:
  POST /predict        → predição de risco de abandono
  GET  /health         → status da API
  GET  /modelo/info    → metadados do modelo
"""
import pickle
import json
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

# ── Inicializar app ─────────────────────────────────────────
app = FastAPI(
    title="API LTFU-TB",
    description="Predição de abandono do tratamento de tuberculose com LightGBM treinado no SINAN.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — permite chamadas do GitHub Pages e local ─────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Em produção, substituir pelo domínio exato do GitHub Pages
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ── Carregar modelo ─────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pipeline_final.pkl")
META_PATH  = os.path.join(os.path.dirname(__file__), "model_metadata.json")

try:
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
    print("Modelo carregado com sucesso.")
except Exception as e:
    print(f"ERRO ao carregar modelo: {e}")
    pipeline = None

meta = {}
if os.path.exists(META_PATH):
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)


# ── Schema de entrada (Pydantic) ────────────────────────────
class PacienteInput(BaseModel):
    # ── Dados demográficos ──
    idade_anos: float = Field(..., ge=0, le=120, description="Idade em anos")
    CS_SEXO: str = Field(..., description="Sexo: M ou F")
    CS_RACA: Optional[str] = Field(None, description="Raça/cor: 1=Branca, 2=Preta, 3=Amarela, 4=Parda, 5=Indígena")

    # ── Tipo de caso ──
    TRATAMENTO: str = Field(..., description="Tipo de entrada: 1=Caso novo, 2=Recidiva, 3=Reingresso pós-abandono, 4=Não sabe, 5=Transferência")
    reingresso: Optional[int] = Field(None, ge=0, le=1, description="1 se reingresso após abandono (derivado de TRATAMENTO=3)")

    # ── Clínico ──
    FORMA: Optional[str] = Field(None, description="Forma clínica: 1=Pulmonar, 2=Extrapulmonar, 3=Ambas")
    BACILOSC_E: Optional[str] = Field(None, description="Baciloscopia de escarro: 1=Positivo, 2=Negativo, 3=Não realizado")
    CULTURA_ES: Optional[str] = Field(None, description="Cultura de escarro: 1=Positivo, 2=Negativo, 3=Em andamento, 4=Não realizado")
    RAIOX_TORA: Optional[str] = Field(None, description="Raio-X de tórax: 1=Normal, 2=Suspeito, 3=Outra patologia, 4=Não realizado")
    TESTE_TUBE: Optional[str] = Field(None, description="Teste tuberculínico: 1=Reator forte, 2=Reator fraco, 3=Não reator, 4=Não realizado")
    TEST_MOLEC: Optional[str] = Field(None, description="Teste molecular: 1=Detectado, 2=Não detectado, 3=Inconclusivo, 4=Não realizado")
    HIV: Optional[str] = Field(None, description="Situação HIV: 1=Positivo, 2=Negativo, 3=Em andamento, 4=Não realizado")

    # ── Comorbidades ──
    AGRAVAIDS:   Optional[float] = Field(None, description="Agravo AIDS: 1=Sim, 2=Não")
    AGRAVALCOO:  Optional[float] = Field(None, description="Alcoolismo: 1=Sim, 2=Não")
    AGRAVDIABE:  Optional[float] = Field(None, description="Diabetes: 1=Sim, 2=Não")
    AGRAVDOENC:  Optional[float] = Field(None, description="Doença mental/emoc.: 1=Sim, 2=Não")
    AGRAVDROGA:  Optional[float] = Field(None, description="Uso de drogas: 1=Sim, 2=Não")
    AGRAVTABAC:  Optional[float] = Field(None, description="Tabagismo: 1=Sim, 2=Não")

    # ── Vulnerabilidade social ──
    POP_RUA:   Optional[float] = Field(None, description="Pop. em situação de rua: 1=Sim, 2=Não")
    POP_LIBER: Optional[float] = Field(None, description="Pop. privada de liberdade: 1=Sim, 2=Não")
    POP_IMIG:  Optional[float] = Field(None, description="Imigrante: 1=Sim, 2=Não")
    BENEF_GOV: Optional[float] = Field(None, description="Benefício governamental: 1=Sim, 2=Não")

    # ── Cuidado ──
    TRAT_SUPER:      Optional[float] = Field(None, description="TDO (tratamento supervisionado): 1=Sim, 2=Não")
    NU_CONTATO:      Optional[float] = Field(None, ge=0, description="Nº de contatos registrados")
    NU_COMU_EX:      Optional[float] = Field(None, ge=0, description="Nº comunicantes examinados")
    dias_notif_trat: Optional[float] = Field(None, ge=0, description="Dias entre notificação e início do tratamento")

    class Config:
        json_schema_extra = {
            "example": {
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
                "AGRAVAIDS": 2, "AGRAVALCOO": 1, "AGRAVDIABE": 2,
                "AGRAVDOENC": 2, "AGRAVDROGA": 2, "AGRAVTABAC": 1,
                "POP_RUA": 1, "POP_LIBER": 2, "POP_IMIG": 2, "BENEF_GOV": 1,
                "TRAT_SUPER": 2,
                "NU_CONTATO": 3, "NU_COMU_EX": 2, "dias_notif_trat": 5
            }
        }


# ── Schema de saída ─────────────────────────────────────────
class PredicaoOutput(BaseModel):
    probabilidade_abandono: float
    classe:  str
    nivel_risco: str
    fatores_risco: list[str]
    recomendacao: str
    aviso: str


# ── Helpers ─────────────────────────────────────────────────
def calcular_reingresso(tratamento: str) -> int:
    return 1 if str(tratamento).strip() == "3" else 0


def nivel_risco(prob: float) -> str:
    if prob >= 0.70:
        return "ALTO"
    elif prob >= 0.40:
        return "MODERADO"
    else:
        return "BAIXO"


def fatores_principais(paciente: PacienteInput, prob: float) -> list[str]:
    """Gera lista de fatores de risco identificados no paciente."""
    fatores = []
    if paciente.reingresso == 1 or str(paciente.TRATAMENTO) == "3":
        fatores.append("Reingresso após abandono anterior — maior predictor de novo abandono")
    if paciente.POP_RUA == 1:
        fatores.append("Situação de rua — vulnerabilidade social crítica")
    if paciente.AGRAVDROGA == 1:
        fatores.append("Uso de drogas — impacta adesão ao tratamento")
    if paciente.AGRAVALCOO == 1:
        fatores.append("Alcoolismo — fator de risco clássico para abandono")
    if paciente.AGRAVAIDS == 1:
        fatores.append("Coinfecção HIV/AIDS — aumenta complexidade do tratamento")
    if paciente.TRAT_SUPER == 2:
        fatores.append("Sem TDO (tratamento supervisionado) — importante fator protetor ausente")
    if paciente.BENEF_GOV == 1 and prob >= 0.50:
        fatores.append("Beneficiário governamental — possível vulnerabilidade socioeconômica")
    if paciente.POP_LIBER == 1:
        fatores.append("Privado de liberdade — ambiente institucional com desafios de adesão")
    if paciente.idade_anos < 25:
        fatores.append("Idade jovem — menor adesão historicamente documentada")
    return fatores if fatores else ["Sem fatores de risco individualmente dominantes identificados"]


def recomendacao(nivel: str) -> str:
    if nivel == "ALTO":
        return ("AÇÃO URGENTE: Iniciar ou intensificar TDO imediatamente. "
                "Agendar visita domiciliar. Acionar assistência social se necessário.")
    elif nivel == "MODERADO":
        return ("ATENÇÃO: Reforçar orientações sobre importância da adesão. "
                "Avaliar necessidade de TDO. Manter contato ativo com o paciente.")
    else:
        return ("Acompanhamento padrão. Manter monitoramento periódico e reforço educativo.")


# ── Endpoints ───────────────────────────────────────────────
@app.get("/health", tags=["Infraestrutura"])
def health_check():
    """Verifica se a API está online e o modelo carregado."""
    return {
        "status": "online",
        "modelo_carregado": pipeline is not None,
        "versao": "1.0.0"
    }


@app.get("/modelo/info", tags=["Infraestrutura"])
def info_modelo():
    """Retorna metadados do modelo treinado."""
    return {
        "algoritmo": meta.get("modelo", "LightGBM"),
        "n_treino": meta.get("n_treino"),
        "roc_auc_teste1": meta.get("roc_auc_teste1"),
        "f1_teste1": meta.get("f1_teste1"),
        "features": meta.get("features_all", []),
        "proporcao_abandono_treino": meta.get("prop_abandono_treino"),
        "fonte_dados": "SINAN-TB — DataSUS (todos os anos disponíveis)",
        "aviso": "Ferramenta de apoio à decisão clínica. Não substitui avaliação médica."
    }


@app.post("/predict", response_model=PredicaoOutput, tags=["Predição"])
def predict(paciente: PacienteInput):
    """
    Recebe dados do paciente e retorna probabilidade de abandono do tratamento.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado. Tente novamente em instantes.")

    # Derivar reingresso se não informado
    reingresso_val = paciente.reingresso
    if reingresso_val is None:
        reingresso_val = calcular_reingresso(paciente.TRATAMENTO)

    # Montar DataFrame com as features na ordem exata do treino
    row = {
        # numéricas/binárias
        "reingresso":      reingresso_val,
        "POP_RUA":         paciente.POP_RUA,
        "AGRAVAIDS":       paciente.AGRAVAIDS,
        "POP_LIBER":       paciente.POP_LIBER,
        "AGRAVDROGA":      paciente.AGRAVDROGA,
        "idade_anos":      paciente.idade_anos,
        "AGRAVALCOO":      paciente.AGRAVALCOO,
        "NU_COMU_EX":      paciente.NU_COMU_EX,
        "AGRAVDIABE":      paciente.AGRAVDIABE,
        "AGRAVTABAC":      paciente.AGRAVTABAC,
        "AGRAVDOENC":      paciente.AGRAVDOENC,
        "POP_IMIG":        paciente.POP_IMIG,
        "BENEF_GOV":       paciente.BENEF_GOV,
        "NU_CONTATO":      paciente.NU_CONTATO,
        "TRAT_SUPER":      paciente.TRAT_SUPER,
        "dias_notif_trat": paciente.dias_notif_trat,
        # nominais
        "CS_SEXO":    str(paciente.CS_SEXO) if paciente.CS_SEXO else "nan",
        "HIV":        str(paciente.HIV)      if paciente.HIV      else "nan",
        "TESTE_TUBE": str(paciente.TESTE_TUBE) if paciente.TESTE_TUBE else "nan",
        "CULTURA_ES": str(paciente.CULTURA_ES) if paciente.CULTURA_ES else "nan",
        "BACILOSC_E": str(paciente.BACILOSC_E) if paciente.BACILOSC_E else "nan",
        "RAIOX_TORA": str(paciente.RAIOX_TORA) if paciente.RAIOX_TORA else "nan",
        "TRATAMENTO": str(paciente.TRATAMENTO),
        "CS_RACA":    str(paciente.CS_RACA)    if paciente.CS_RACA    else "nan",
        "TEST_MOLEC": str(paciente.TEST_MOLEC) if paciente.TEST_MOLEC else "nan",
    }

    df_input = pd.DataFrame([row])

    # Predição
    prob = float(pipeline.predict_proba(df_input)[0][1])
    nivel = nivel_risco(prob)
    fatores = fatores_principais(paciente, prob)
    rec = recomendacao(nivel)

    return PredicaoOutput(
        probabilidade_abandono=round(prob, 4),
        classe="Abandono" if prob >= 0.5 else "Cura",
        nivel_risco=nivel,
        fatores_risco=fatores,
        recomendacao=rec,
        aviso="Esta predição é uma ferramenta de apoio clínico baseada em dados populacionais. Não substitui avaliação médica individualizada."
    )


# ── Execução local ──────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

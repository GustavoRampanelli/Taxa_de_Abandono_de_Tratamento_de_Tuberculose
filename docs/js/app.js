/**
 * prevTB — Frontend Application Logic
 * Integrates with FastAPI Render backend for real-time ML inference
 */

// ── CONFIGURAÇÃO DA API ───────────────────────────────────────
// Substitua pela sua URL real do Render após o deploy
const API_BASE_URL = "https://taxa-de-abandono-de-tratamento-de.onrender.com"; 
// Fallback local se estiver rodando localmente
const LOCAL_API_URL = "http://localhost:8000";

let ACTIVE_API_URL = API_BASE_URL;

// ── ELEMENTOS DO DOM ──────────────────────────────────────────
const elements = {
    // Abas / Navegação
    btnDashboard: document.getElementById("btnDashboard"),
    btnModelo: document.getElementById("btnModelo"),
    btnSobre: document.getElementById("btnSobre"),
    
    sections: {
        Dashboard: document.getElementById("sectionDashboard"),
        Modelo: document.getElementById("sectionModelo"),
        Sobre: document.getElementById("sectionSobre")
    },

    // Formulário
    form: document.getElementById("ltfuForm"),
    btnReset: document.getElementById("btnReset"),
    btnSubmit: document.getElementById("btnSubmit"),

    // Estados de Resultado
    resultCard: document.getElementById("resultCard"),
    resultPlaceholder: document.getElementById("resultPlaceholder"),
    resultLoading: document.getElementById("resultLoading"),
    resultSuccess: document.getElementById("resultSuccess"),
    
    // Indicadores do Loading
    loadingStatusText: document.getElementById("loadingStatusText"),
    loadingProgress: document.getElementById("loadingProgress"),
    coldStartWarning: document.getElementById("coldStartWarning"),

    // Renderização dos Resultados
    riskIndicatorContainer: document.getElementById("riskIndicatorContainer"),
    riskLevel: document.getElementById("riskLevel"),
    riskPercent: document.getElementById("riskPercent"),
    riskMeterFill: document.getElementById("riskMeterFill"),
    factorsList: document.getElementById("factorsList"),
    recommendationBox: document.getElementById("recommendationBox"),

    // Indicadores de status globais
    topProgressBar: document.getElementById("topProgressBar"),
    apiStatusIndicator: document.getElementById("apiStatusIndicator")
};

// ── 1. DIRECIONAMENTO E INICIALIZAÇÃO DE NAVEGAÇÃO ─────────────
function initNavigation() {
    const tabs = ["Dashboard", "Modelo", "Sobre"];

    tabs.forEach(tab => {
        elements[`btn${tab}`].addEventListener("click", () => {
            // Remover active de todos os botões
            tabs.forEach(t => elements[`btn${t}`].classList.remove("active"));
            // Ocultar todas as seções
            tabs.forEach(t => {
                elements.sections[t].classList.add("hidden");
                elements.sections[t].classList.remove("active");
            });

            // Ativar a aba clicada
            elements[`btn${tab}`].classList.add("active");
            elements.sections[tab].classList.remove("hidden");
            elements.sections[tab].classList.add("active");

            // Scroll suave para o topo
            window.scrollTo({ top: 0, behavior: "smooth" });
        });
    });
}

// ── 2. MONITORAMENTO DE STATUS DA API ─────────────────────────
async function checkApiStatus() {
    try {
        // Tenta conectar no Render
        let response = await fetch(`${ACTIVE_API_URL}/health`, { method: 'GET', timeout: 5000 });
        if (response.ok) {
            setApiOnline(true);
            return;
        }
    } catch (e) {
        console.warn("API principal fora do ar. Tentando conexão local fallback...");
    }

    try {
        // Se falhar, tenta localhost
        let responseLocal = await fetch(`${LOCAL_API_URL}/health`, { method: 'GET' });
        if (responseLocal.ok) {
            ACTIVE_API_URL = LOCAL_API_URL;
            setApiOnline(true);
            console.log("Conectado à API local com sucesso:", ACTIVE_API_URL);
            return;
        }
    } catch (e) {
        console.error("Ambos os servidores (Render e Local) estão offline.");
    }
    
    setApiOnline(false);
}

function setApiOnline(isOnline) {
    if (isOnline) {
        elements.apiStatusIndicator.textContent = "API ONLINE";
        elements.apiStatusIndicator.className = "api-status-indicator online";
    } else {
        elements.apiStatusIndicator.textContent = "API OFFLINE";
        elements.apiStatusIndicator.className = "api-status-indicator offline";
    }
}

// ── 3. ANIMAÇÃO DE CARREGAMENTO / INFERÊNCIA ───────────────────
let progressInterval = null;

function startLoadingAnimation() {
    // Reset de estados visuais anteriores
    elements.resultPlaceholder.classList.add("hidden");
    elements.resultSuccess.classList.add("hidden");
    elements.resultLoading.classList.remove("hidden");
    elements.resultCard.className = "card result-card placeholder-state";
    
    elements.topProgressBar.className = "top-progress-bar active";

    let progress = 0;
    elements.loadingProgress.style.width = "0%";
    elements.coldStartWarning.classList.add("hidden");

    const statusSteps = [
        { limit: 15, text: "Conectando ao servidor..." },
        { limit: 40, text: "Buscando pesos no LightGBM..." },
        { limit: 65, text: "Analisando comorbidades..." },
        { limit: 85, text: "Processando vulnerabilidade social..." },
        { limit: 95, text: "Finalizando cálculo de probabilidade..." }
    ];

    let stepIndex = 0;
    let timerCount = 0;

    progressInterval = setInterval(() => {
        timerCount += 100;
        // Progresso suave
        if (progress < 95) {
            progress += 0.5;
            elements.loadingProgress.style.width = `${progress}%`;
        }

        // Atualizar textos baseados no progresso
        if (stepIndex < statusSteps.length && progress >= statusSteps[stepIndex].limit) {
            elements.loadingStatusText.textContent = statusSteps[stepIndex].text;
            stepIndex++;
        }

        // Se passar de 6 segundos, exibe aviso sobre Cold Start do Render
        if (timerCount > 6000) {
            elements.coldStartWarning.classList.remove("hidden");
        }
    }, 100);
}

function stopLoadingAnimation(isSuccess) {
    clearInterval(progressInterval);
    elements.loadingProgress.style.width = "100%";
    
    if (isSuccess) {
        elements.topProgressBar.className = "top-progress-bar done";
        setTimeout(() => {
            elements.topProgressBar.className = "top-progress-bar";
        }, 500);
    } else {
        elements.topProgressBar.className = "top-progress-bar";
    }

    setTimeout(() => {
        elements.resultLoading.classList.add("hidden");
    }, 200);
}

// ── 4. SUBMISSÃO E CONSUMO DO MODELO ML ───────────────────────
async function handleFormSubmit(e) {
    e.preventDefault();

    // Iniciar animações de loading
    startLoadingAnimation();

    // Extrair dados do formulário
    const formData = new FormData(elements.form);
    const rawData = {};
    
    formData.forEach((value, key) => {
        rawData[key] = value;
    });

    // Mapeamento e conversão de tipos requeridos pelo FastAPI
    const payload = {
        idade_anos: parseFloat(rawData.idade_anos),
        CS_SEXO: rawData.CS_SEXO,
        CS_RACA: rawData.CS_RACA === "nan" ? null : rawData.CS_RACA,
        TRATAMENTO: rawData.TRATAMENTO,
        
        // Conversão de opcionais
        FORMA: rawData.FORMA === "nan" ? null : rawData.FORMA,
        BACILOSC_E: rawData.BACILOSC_E === "nan" ? null : rawData.BACILOSC_E,
        CULTURA_ES: rawData.CULTURA_ES === "nan" ? null : rawData.CULTURA_ES,
        RAIOX_TORA: rawData.RAIOX_TORA === "nan" ? null : rawData.RAIOX_TORA,
        TESTE_TUBE: rawData.TESTE_TUBE === "nan" ? null : rawData.TESTE_TUBE,
        TEST_MOLEC: rawData.TEST_MOLEC === "nan" ? null : rawData.TEST_MOLEC,
        HIV: rawData.HIV === "nan" ? null : rawData.HIV,

        // Comorbidades e Vulnerabilidades
        AGRAVAIDS:   rawData.AGRAVAIDS === "nan" ? null : parseFloat(rawData.AGRAVAIDS),
        AGRAVALCOO:  rawData.AGRAVALCOO === "nan" ? null : parseFloat(rawData.AGRAVALCOO),
        AGRAVDIABE:  rawData.AGRAVDIABE === "nan" ? null : parseFloat(rawData.AGRAVDIABE),
        AGRAVDOENC:  rawData.AGRAVDOENC === "nan" ? null : parseFloat(rawData.AGRAVDOENC),
        AGRAVDROGA:  rawData.AGRAVDROGA === "nan" ? null : parseFloat(rawData.AGRAVDROGA),
        AGRAVTABAC:  rawData.AGRAVTABAC === "nan" ? null : parseFloat(rawData.AGRAVTABAC),
        
        POP_RUA:   rawData.POP_RUA === "nan" ? null : parseFloat(rawData.POP_RUA),
        POP_LIBER: rawData.POP_LIBER === "nan" ? null : parseFloat(rawData.POP_LIBER),
        POP_IMIG:  rawData.POP_IMIG === "nan" ? null : parseFloat(rawData.POP_IMIG),
        BENEF_GOV: rawData.BENEF_GOV === "nan" ? null : parseFloat(rawData.BENEF_GOV),

        // Cuidados
        TRAT_SUPER:      rawData.TRAT_SUPER === "nan" ? null : parseFloat(rawData.TRAT_SUPER),
        NU_CONTATO:      rawData.NU_CONTATO ? parseFloat(rawData.NU_CONTATO) : 0,
        NU_COMU_EX:      rawData.NU_COMU_EX ? parseFloat(rawData.NU_COMU_EX) : 0,
        dias_notif_trat: rawData.dias_notif_trat ? parseFloat(rawData.dias_notif_trat) : 0
    };

    try {
        const response = await fetch(`${ACTIVE_API_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`Falha no cálculo: ${response.statusText}`);
        }

        const data = await response.json();
        
        // Finalizar animação e renderizar
        stopLoadingAnimation(true);
        renderResults(data);

    } catch (error) {
        console.error(error);
        stopLoadingAnimation(false);
        showErrorState(error.message);
    }
}

// ── 5. RENDERIZAÇÃO DE RESULTADOS NA TELA ───────────────────────
function renderResults(data) {
    elements.resultCard.className = "card result-card";
    elements.resultPlaceholder.classList.add("hidden");
    elements.resultSuccess.classList.remove("hidden");

    // Risco e Probabilidade
    const probPct = Math.round(data.probabilidade_abandono * 100);
    elements.riskPercent.textContent = `${probPct}%`;
    elements.riskLevel.textContent = `${data.nivel_risco} RISCO`;

    // Resetar classes de risco
    elements.riskIndicatorContainer.className = "risk-indicator";
    elements.recommendationBox.className = "recommendation-box";

    // Adicionar estilos específicos da faixa de risco
    let riskClass = "";
    if (data.nivel_risco === "ALTO") {
        riskClass = "risk-state-high";
        elements.recommendationBox.classList.add("high");
    } else if (data.nivel_risco === "MODERADO") {
        riskClass = "risk-state-mod";
        elements.recommendationBox.classList.add("mod");
    } else {
        riskClass = "risk-state-low";
        elements.recommendationBox.classList.add("low");
    }

    elements.riskIndicatorContainer.classList.add(riskClass);

    // Animando a barra de risco
    setTimeout(() => {
        elements.riskMeterFill.style.width = `${probPct}%`;
    }, 100);

    // Listar fatores contribuintes
    elements.factorsList.innerHTML = "";
    if (data.fatores_risco && data.fatores_risco.length > 0) {
        data.fatores_risco.forEach(fator => {
            const li = document.createElement("li");
            li.textContent = fator;
            elements.factorsList.appendChild(li);
        });
    } else {
        const li = document.createElement("li");
        li.textContent = "Sem fatores determinantes detectados.";
        elements.factorsList.appendChild(li);
    }

    // Diretriz de acompanhamento
    elements.recommendationBox.textContent = data.recomendacao;
}

// Exibir painel com erro amigável se a requisição falhar
function showErrorState(errorMsg) {
    elements.resultCard.className = "card result-card placeholder-state";
    elements.resultPlaceholder.classList.remove("hidden");
    elements.resultSuccess.classList.add("hidden");

    const placeholderTitle = elements.resultPlaceholder.querySelector("h3");
    const placeholderText = elements.resultPlaceholder.querySelector("p");

    placeholderTitle.textContent = "Serviço Indisponível";
    placeholderTitle.style.color = "var(--risk-high)";
    placeholderText.innerHTML = `Ocorreu um erro ao conectar com o modelo preditivo: <br><strong style="font-size:11px; color:var(--text-main);">${errorMsg}</strong>.<br><br>Verifique se o backend está ativo no Render ou se a API local está rodando em <code>localhost:8000</code>.`;
}

// Limpar Ficha do Paciente
function resetForm() {
    elements.form.reset();
    
    // Voltar painel lateral para o estado inicial
    elements.resultCard.className = "card result-card placeholder-state";
    elements.resultPlaceholder.classList.remove("hidden");
    elements.resultSuccess.classList.add("hidden");
    elements.resultLoading.classList.add("hidden");

    const placeholderTitle = elements.resultPlaceholder.querySelector("h3");
    const placeholderText = elements.resultPlaceholder.querySelector("p");
    
    placeholderTitle.textContent = "Aguardando Avaliação";
    placeholderTitle.style.color = "var(--text-main)";
    placeholderText.innerHTML = `Preencha os campos obrigatórios da Ficha do Paciente e clique no botão <strong>"Calcular Probabilidade de Risco"</strong> para iniciar a inferência do modelo.`;
    
    elements.riskMeterFill.style.width = "0%";
}

// ── EVENTS ────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    checkApiStatus();

    // Eventos do formulário
    elements.form.addEventListener("submit", handleFormSubmit);
    elements.btnReset.addEventListener("click", resetForm);
});

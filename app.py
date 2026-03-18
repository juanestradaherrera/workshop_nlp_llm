"""
Parcial – NLP & LLMs Avanzado
Maestría en Ciencia de Datos | EAFIT 2026-1
Docente: Jorge Iván Padilla-Buriticá

Aplicación Streamlit con 4 secciones:
1. Quiz Teórico (Parte 01)
2. Laboratorio de Parámetros (Parte 02)
3. Métricas de Similitud (Parte 03)
4. Agente Conversacional (Parte 04)
"""

import streamlit as st
import time
import json
import re
import math
from collections import Counter

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NLP & LLMs Avanzado – EAFIT",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# ESTILOS GLOBALES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
}
.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #00FFB3;
    border-bottom: 2px solid #00FFB3;
    padding-bottom: 0.4rem;
    margin-bottom: 1.2rem;
}
.section-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.metric-pill {
    display: inline-block;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: #58a6ff;
    margin: 2px;
}
.answer-box {
    background: #0d2137;
    border-left: 4px solid #00FFB3;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-top: 0.5rem;
    font-size: 0.92rem;
    line-height: 1.65;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR – API KEY & MODELO
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 Configuración API")

    api_provider = st.selectbox("Proveedor", ["Groq", "OpenAI", "Gemini"])

    # Intentar leer desde secrets, si no, pedir input
    secret_key_map = {
        "Groq": "GROQ_API_KEY",
        "OpenAI": "OPENAI_API_KEY",
        "Gemini": "GEMINI_API_KEY",
    }
    secret_name = secret_key_map[api_provider]
    default_key = st.secrets.get(secret_name, "") if hasattr(st, "secrets") else ""

    api_key = st.text_input(
        f"API Key ({api_provider})",
        value=default_key,
        type="password",
        help="Nunca se almacena en código. Usa st.secrets o ingresa aquí.",
    )

    model_options = {
        "Groq": ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
        "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "Gemini": ["gemini-1.5-flash", "gemini-1.5-pro"],
    }
    selected_model = st.selectbox("Modelo", model_options[api_provider])

    st.markdown("---")
    st.markdown("**EAFIT · NLP & LLMs Avanzado**  \nPeriodo 2026-1")


# ─────────────────────────────────────────────
# FUNCIÓN CENTRAL DE LLAMADA A LA API
# ─────────────────────────────────────────────
def call_llm(
    messages: list,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 512,
    top_k: int = None,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> tuple[str, object]:
    """
    Llama al LLM seleccionado y retorna (texto_respuesta, objeto_usage).
    Soporta Groq, OpenAI y Gemini según la selección del sidebar.
    """
    if not api_key:
        st.warning("⚠️ Ingresa una API Key en el panel lateral.")
        return "", None

    if api_provider == "Groq":
        from groq import Groq
        client = Groq(api_key=api_key)
        extra = {}
        if top_k is not None:
            # Groq no expone top_k directamente; se omite sin error
            pass
        resp = client.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        return resp.choices[0].message.content, resp.usage

    elif api_provider == "OpenAI":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        return resp.choices[0].message.content, resp.usage

    elif api_provider == "Gemini":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(selected_model)
        # Gemini usa GenerationConfig
        from google.generativeai.types import GenerationConfig
        cfg = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k else 40,
            max_output_tokens=max_tokens,
        )
        # Convertir messages al formato de Gemini
        prompt = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )
        resp = model.generate_content(prompt, generation_config=cfg)
        # Simular objeto usage compatible
        class FakeUsage:
            prompt_tokens = 0
            completion_tokens = len(resp.text.split())
            total_tokens = completion_tokens
        return resp.text, FakeUsage()

    return "", None


# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────
def simple_tokenize(text: str) -> list[str]:
    """Tokenización simple por palabras (sin dependencias externas)."""
    return re.findall(r"\b\w+\b", text.lower())


def type_token_ratio(text: str) -> float:
    """
    Calcula Type-Token Ratio (TTR): tipos únicos / total tokens.
    Valor entre 0 y 1; más alto = mayor diversidad léxica.
    """
    tokens = simple_tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def cosine_similarity_vectors(v1: list, v2: list) -> float:
    """Similitud coseno entre dos vectores numéricos."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a ** 2 for a in v1))
    norm2 = math.sqrt(sum(b ** 2 for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Obtiene embeddings usando sentence-transformers (all-MiniLM-L6-v2).
    Retorna lista de vectores.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(texts).tolist()
    except Exception as e:
        st.error(f"Error obteniendo embeddings: {e}")
        return [[], []]


def compute_bleu(reference: str, hypothesis: str) -> float:
    """
    Calcula BLEU score usando nltk con smoothing.
    Compara n-gramas entre referencia e hipótesis.
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        nltk.download("punkt", quiet=True)
        ref_tokens = simple_tokenize(reference)
        hyp_tokens = simple_tokenize(hypothesis)
        sf = SmoothingFunction().method1
        return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=sf)
    except Exception as e:
        st.error(f"Error BLEU: {e}")
        return 0.0


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """
    Calcula ROUGE-L (F1) usando rouge-score.
    Mide la subsecuencia común más larga entre referencia e hipótesis.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return scores["rougeL"].fmeasure
    except Exception as e:
        st.error(f"Error ROUGE-L: {e}")
        return 0.0


def compute_bertscore(reference: str, hypothesis: str) -> float:
    """
    Calcula BERTScore F1 usando bert-score.
    Mide similitud semántica token-a-token con BERT.
    """
    try:
        from bert_score import score as bert_score
        P, R, F1 = bert_score([hypothesis], [reference], lang="es", verbose=False)
        return F1[0].item()
    except Exception as e:
        st.error(f"Error BERTScore: {e}")
        return 0.0


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """
    Estima costo en USD según pricing aproximado de cada proveedor/modelo.
    Precios por 1M tokens (input / output).
    """
    pricing = {
        "Groq": {
            "llama-3.3-70b-versatile": (0.59, 0.79),
            "mixtral-8x7b-32768": (0.24, 0.24),
            "gemma2-9b-it": (0.20, 0.20),
        },
        "OpenAI": {
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4o": (5.00, 15.00),
            "gpt-3.5-turbo": (0.50, 1.50),
        },
        "Gemini": {
            "gemini-1.5-flash": (0.075, 0.30),
            "gemini-1.5-pro": (3.50, 10.50),
        },
    }
    inp_price, out_price = pricing.get(api_provider, {}).get(selected_model, (1.0, 1.0))
    return (prompt_tokens * inp_price + completion_tokens * out_price) / 1_000_000


# ─────────────────────────────────────────────
# PESTAÑAS PRINCIPALES
# ─────────────────────────────────────────────
tab2, tab3, tab4 = st.tabs([
    "⚗️ Laboratorio de Parámetros",
    "📊 Métricas de Similitud",
    "🤖 Agente Conversacional",
])


# ══════════════════════════════════════════════
# PARTE 02 – LABORATORIO DE PARÁMETROS
# ══════════════════════════════════════════════
with tab2:
    import plotly.graph_objects as go
    import plotly.express as px

    st.markdown('<div class="main-title">PARTE 02 · Laboratorio de Sintonización de Parámetros (30 pts)</div>', unsafe_allow_html=True)

    # ── Panel de control ──
    st.subheader("🎛️ Panel de Control Interactivo")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        temperature = st.slider("🌡️ Temperatura", 0.0, 2.0, 0.7, 0.05,
                                 help="Creatividad vs. determinismo de la respuesta")
        top_p = st.slider("🎯 Top-p (nucleus)", 0.0, 1.0, 0.9, 0.05,
                           help="Diversidad controlando masa de probabilidad acumulada")
    with col_p2:
        top_k = st.slider("🔢 Top-k", 1, 100, 40,
                           help="Vocabulario efectivo en cada paso de generación")
        max_tokens_lab = st.slider("📏 Max tokens", 50, 2048, 512, 50,
                                    help="Longitud máxima de la respuesta generada")
    with col_p3:
        freq_penalty = st.slider("🔁 Frequency penalty", 0.0, 2.0, 0.0, 0.1,
                                  help="Penalización por repetición de tokens frecuentes")
        pres_penalty = st.slider("💡 Presence penalty", 0.0, 2.0, 0.0, 0.1,
                                  help="Penalización por aparición previa de tokens")

    prompt_lab = st.text_area(
        "✏️ Prompt personalizado",
        value="Explica el concepto de atención (attention) en transformers.",
        height=80,
    )

    if st.button("🚀 Generar respuesta con configuración actual"):
        with st.spinner("Generando..."):
            t0 = time.time()
            text, usage = call_llm(
                [{"role": "user", "content": prompt_lab}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens_lab,
                frequency_penalty=freq_penalty,
                presence_penalty=pres_penalty,
            )
            latency = time.time() - t0
        if text:
            st.markdown("**Respuesta generada:**")
            st.info(text)
            if usage:
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱️ Latencia", f"{latency:.2f}s")
                c2.metric("📥 Tokens entrada", usage.prompt_tokens)
                c3.metric("📤 Tokens salida", usage.completion_tokens)

    st.markdown("---")

    # ── Experimento comparativo ──
    st.subheader("🔬 Experimento Comparativo (4 configuraciones)")

    FIXED_PROMPT = "Explica el concepto de atención (attention) en transformers."
    configs = [
        {"label": "T=0.1 p=0.9\n(Determinista+amplio)", "temp": 0.1, "top_p": 0.9},
        {"label": "T=1.5 p=0.9\n(Creativo+amplio)",     "temp": 1.5, "top_p": 0.9},
        {"label": "T=0.1 p=0.3\n(Determinista+estricto)", "temp": 0.1, "top_p": 0.3},
        {"label": "T=1.5 p=0.3\n(Creativo+estricto)",   "temp": 1.5, "top_p": 0.3},
    ]

    if "exp_results" not in st.session_state:
        st.session_state["exp_results"] = None

    if st.button("▶️ Ejecutar experimento comparativo (4 llamadas API)"):
        results = []
        progress = st.progress(0)
        for i, cfg in enumerate(configs):
            with st.spinner(f"Config {i+1}/4: temp={cfg['temp']}, top_p={cfg['top_p']}..."):
                t0 = time.time()
                text, usage = call_llm(
                    [{"role": "user", "content": FIXED_PROMPT}],
                    temperature=cfg["temp"],
                    top_p=cfg["top_p"],
                    max_tokens=300,
                )
                latency = time.time() - t0
                if text:
                    results.append({
                        "label": cfg["label"],
                        "text": text,
                        "tokens": len(simple_tokenize(text)),
                        "ttr": type_token_ratio(text),
                        "latency": latency,
                    })
            progress.progress((i + 1) / 4)
        st.session_state["exp_results"] = results
        st.success("✅ Experimento completado.")

    if st.session_state["exp_results"]:
        results = st.session_state["exp_results"]
        cols = st.columns(4)
        for i, (col, r) in enumerate(zip(cols, results)):
            with col:
                st.markdown(f"**Config {i+1}**")
                st.caption(r["label"].replace("\n", " · "))
                st.text_area(f"resp_{i}", r["text"], height=200, key=f"resp_text_{i}")

        # Gráficas Plotly
        labels = [f"C{i+1}" for i in range(len(results))]
        tokens_vals = [r["tokens"] for r in results]
        ttr_vals = [r["ttr"] for r in results]

        fig_tokens = go.Figure(go.Bar(
            x=labels, y=tokens_vals,
            marker_color=["#00FFB3", "#FF6B6B", "#4ECDC4", "#FFE66D"],
            text=tokens_vals, textposition="outside",
        ))
        fig_tokens.update_layout(
            title="Longitud en Tokens por Configuración",
            yaxis_title="Tokens",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cdd9e5",
        )

        fig_ttr = go.Figure(go.Bar(
            x=labels, y=[round(v, 3) for v in ttr_vals],
            marker_color=["#00FFB3", "#FF6B6B", "#4ECDC4", "#FFE66D"],
            text=[f"{v:.3f}" for v in ttr_vals], textposition="outside",
        ))
        fig_ttr.update_layout(
            title="Diversidad Léxica (Type-Token Ratio)",
            yaxis_title="TTR",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cdd9e5",
        )

        col_g1, col_g2 = st.columns(2)
        col_g1.plotly_chart(fig_tokens, use_container_width=True)
        col_g2.plotly_chart(fig_ttr, use_container_width=True)

        # ── Campo de observaciones guardables ──
        st.markdown("**📝 Documentación de Observaciones**")

        # Inicializar historial de observaciones guardadas
        if "saved_observations" not in st.session_state:
            st.session_state["saved_observations"] = []

        obs_input = st.text_area(
            "Anota tus observaciones sobre el efecto de los parámetros:",
            placeholder=(
                "Ejemplo: Con temperatura alta (1.5) y top_p=0.9 la respuesta fue más creativa "
                "pero menos coherente. Con temperatura baja (0.1) la respuesta fue más concisa y técnica. "
                "El TTR más alto se observó en configuraciones con mayor temperatura..."
            ),
            height=120,
            key="obs_input_field",
        )

        col_obs1, col_obs2 = st.columns([1, 4])
        with col_obs1:
            if st.button("💾 Guardar observación"):
                if obs_input.strip():
                    import datetime
                    st.session_state["saved_observations"].append({
                        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                        "text": obs_input.strip(),
                    })
                    st.success("✅ Observación guardada.")
                else:
                    st.warning("Escribe algo antes de guardar.")
        with col_obs2:
            if st.button("🗑️ Limpiar todas las observaciones") and st.session_state["saved_observations"]:
                st.session_state["saved_observations"] = []
                st.rerun()

        # Mostrar observaciones guardadas
        if st.session_state["saved_observations"]:
            st.markdown("**📋 Observaciones guardadas:**")
            for i, entry in enumerate(st.session_state["saved_observations"], 1):
                st.markdown(
                    f'<div class="answer-box">'
                    f'<span style="color:#00FFB3;font-size:0.75rem;">#{i} · {entry["timestamp"]}</span><br>'
                    f'{entry["text"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════
# PARTE 03 – MÉTRICAS DE SIMILITUD
# ══════════════════════════════════════════════
with tab3:
    import plotly.graph_objects as go

    st.markdown('<div class="main-title">PARTE 03 · Métricas de Similitud y Evaluación Automática (30 pts)</div>', unsafe_allow_html=True)

    col_ref, col_gen = st.columns(2)
    with col_ref:
        reference_text = st.text_area(
            "📄 Texto de Referencia (Ground Truth)",
            value=(
                "La atención en transformers permite al modelo ponderar la importancia relativa "
                "de cada token en la secuencia de entrada al generar cada token de salida. "
                "Utiliza tres matrices: Query, Key y Value, calculando puntuaciones de atención "
                "mediante el producto escalar escalado de Q y K, seguido de softmax."
            ),
            height=160,
        )

    with col_gen:
        gen_prompt = st.text_area(
            "🔤 Prompt para generar respuesta candidata",
            value="Explica brevemente cómo funciona el mecanismo de atención en los transformers.",
            height=80,
        )
        gen_temp = st.slider("Temperatura (generación)", 0.0, 1.5, 0.5, 0.1, key="gen_temp")

    if st.button("⚙️ Generar texto y calcular todas las métricas"):
        with st.spinner("Generando respuesta con LLM..."):
            generated_text, _ = call_llm(
                [{"role": "user", "content": gen_prompt}],
                temperature=gen_temp,
                max_tokens=200,
            )

        if generated_text:
            st.markdown("**🤖 Texto Generado:**")
            st.info(generated_text)

            st.markdown("---")
            st.subheader("📐 Cálculo de Métricas")

            scores = {}

            # 1. Similitud Coseno
            with st.spinner("Calculando similitud coseno (embeddings)..."):
                embeddings = get_embeddings([reference_text, generated_text])
                if embeddings[0] and embeddings[1]:
                    cos_sim = cosine_similarity_vectors(embeddings[0], embeddings[1])
                    scores["Coseno"] = round(cos_sim, 4)
                else:
                    scores["Coseno"] = 0.0

            # 2. BLEU
            with st.spinner("Calculando BLEU..."):
                bleu = compute_bleu(reference_text, generated_text)
                scores["BLEU"] = round(bleu, 4)

            # 3. ROUGE-L
            with st.spinner("Calculando ROUGE-L..."):
                rouge = compute_rouge_l(reference_text, generated_text)
                scores["ROUGE-L"] = round(rouge, 4)

            # 4. BERTScore
            with st.spinner("Calculando BERTScore (puede tardar ~15s la primera vez)..."):
                bert_f1 = compute_bertscore(reference_text, generated_text)
                scores["BERTScore"] = round(bert_f1, 4)

            # Mostrar métricas
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🔵 Coseno", scores["Coseno"])
            m2.metric("🟠 BLEU", scores["BLEU"])
            m3.metric("🟢 ROUGE-L", scores["ROUGE-L"])
            m4.metric("🟣 BERTScore", scores["BERTScore"])

            # 5. LLM-as-Judge
            st.markdown("---")
            st.subheader("⚖️ LLM-as-Judge")

            judge_system = """Eres un evaluador experto en NLP. Evalúa la respuesta generada
comparándola con la referencia. Responde ÚNICAMENTE en JSON con este esquema exacto (sin markdown):
{
  "score": <número 1-10>,
  "veracidad": <número 1-10>,
  "coherencia": <número 1-10>,
  "relevancia": <número 1-10>,
  "fortalezas": "<texto>",
  "debilidades": "<texto>"
}"""

            judge_user = f"""REFERENCIA: {reference_text}

RESPUESTA GENERADA: {generated_text}

PROMPT ORIGINAL: {gen_prompt}"""

            with st.spinner("LLM evaluando respuesta..."):
                judge_raw, _ = call_llm(
                    [
                        {"role": "system", "content": judge_system},
                        {"role": "user", "content": judge_user},
                    ],
                    temperature=0.1,
                    max_tokens=400,
                )

            judge_data = None
            if judge_raw:
                try:
                    clean = re.sub(r"```json|```", "", judge_raw).strip()
                    judge_data = json.loads(clean)
                except Exception:
                    st.warning("⚠️ No se pudo parsear el JSON del juez. Respuesta bruta:")
                    st.code(judge_raw)

            if judge_data:
                j1, j2, j3, j4 = st.columns(4)
                j1.metric("🏆 Score Global", f"{judge_data.get('score', '?')}/10")
                j2.metric("✅ Veracidad", f"{judge_data.get('veracidad', '?')}/10")
                j3.metric("🔗 Coherencia", f"{judge_data.get('coherencia', '?')}/10")
                j4.metric("🎯 Relevancia", f"{judge_data.get('relevancia', '?')}/10")

                col_f, col_d = st.columns(2)
                with col_f:
                    st.success(f"**Fortalezas:** {judge_data.get('fortalezas', '')}")
                with col_d:
                    st.error(f"**Debilidades:** {judge_data.get('debilidades', '')}")

                # Radar chart
                judge_score_norm = judge_data.get("score", 5) / 10
                categories = ["Coseno", "BLEU", "ROUGE-L", "BERTScore", "LLM-Judge"]
                values = [
                    scores["Coseno"],
                    scores["BLEU"],
                    scores["ROUGE-L"],
                    scores["BERTScore"],
                    judge_score_norm,
                ]
                values_closed = values + [values[0]]
                cats_closed = categories + [categories[0]]

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values_closed,
                    theta=cats_closed,
                    fill="toself",
                    fillcolor="rgba(0, 255, 179, 0.2)",
                    line=dict(color="#00FFB3", width=2),
                    name="Puntuaciones",
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1], tickfont_color="#cdd9e5"),
                        angularaxis=dict(tickfont_color="#cdd9e5"),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#cdd9e5",
                    title="Radar Chart – Métricas Normalizadas",
                    showlegend=False,
                )
                st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════
# PARTE 04 – AGENTE CONVERSACIONAL
# ══════════════════════════════════════════════
with tab4:
    import plotly.graph_objects as go

    st.markdown('<div class="main-title">PARTE 04 · Agente Conversacional con Métricas de Producción (15 pts)</div>', unsafe_allow_html=True)

    # ── Personalidades predefinidas ──
    PRESET_PERSONALITIES = {
        "🎓 TutorML (Machine Learning)": (
            "TutorML",
            """Eres TutorML, un tutor experto en Machine Learning y Deep Learning.
Tu personalidad es amigable, precisa y pedagógica. Explicas conceptos complejos con analogías claras.
Dominio: ML, DL, NLP, LLMs, matemáticas para IA, frameworks (PyTorch, TensorFlow, scikit-learn).
Restricciones: No respondes preguntas fuera de tu dominio de ML/IA. Si te preguntan algo fuera de tu área, lo indicas con amabilidad y rediriges la conversación.
Siempre estructura tus respuestas con claridad. Usa ejemplos concretos cuando sea posible.""",
        ),
        "⚖️ Asistente Jurídico": (
            "LexBot",
            """Eres LexBot, un asistente jurídico experto en derecho colombiano e internacional.
Tu personalidad es formal, precisa y cautelosa. Siempre aclaras que tus respuestas son orientativas y no reemplazan asesoría legal profesional.
Dominio: derecho civil, penal, laboral, comercial, constitucional colombiano.
Restricciones: No emites conceptos definitivos sobre casos reales. Siempre recomiendas consultar un abogado.""",
        ),
        "⚽ Experto en Deportes": (
            "SportsPro",
            """Eres SportsPro, un analista deportivo apasionado y conocedor de múltiples disciplinas.
Tu personalidad es entusiasta, dinámica y accesible. Usas analogías deportivas para explicar conceptos.
Dominio: fútbol, baloncesto, tenis, atletismo, ciclismo, estadísticas deportivas, historia del deporte.
Restricciones: No haces predicciones de apuestas ni pronósticos de resultados futuros.""",
        ),
        "🍳 Chef Culinario": (
            "ChefBot",
            """Eres ChefBot, un chef profesional con formación en cocina francesa, italiana y latinoamericana.
Tu personalidad es cálida, creativa y detallista. Amas compartir trucos de cocina y el origen cultural de cada plato.
Dominio: recetas, técnicas culinarias, maridaje, nutrición básica, historia gastronómica.
Restricciones: No recomendas dietas médicas ni tratamientos nutricionales especializados.""",
        ),
        "✏️ Personalidad personalizada": ("Mi Agente", ""),
    }

    # ── Sidebar del agente ──
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🤖 Config Agente")
        agent_temp = st.slider("Temperatura agente", 0.0, 1.5, 0.7, 0.05, key="agent_temp")
        agent_max_tokens = st.slider("Max tokens agente", 100, 1024, 400, 50, key="agent_max")

        if st.button("🗑️ Limpiar conversación"):
            st.session_state["agent_messages"] = []
            st.session_state["agent_metrics_history"] = []
            st.rerun()

    # ── Selector de personalidad ──
    st.subheader("🎭 Personalidad del Agente")
    preset_choice = st.selectbox(
        "Elige una personalidad predefinida o crea la tuya:",
        list(PRESET_PERSONALITIES.keys()),
        key="preset_choice",
    )
    default_name, default_prompt = PRESET_PERSONALITIES[preset_choice]

    col_name, col_reset = st.columns([3, 1])
    with col_name:
        agent_name = st.text_input(
            "Nombre del agente",
            value=default_name,
            key="agent_name_input",
        )
    with col_reset:
        st.markdown("<br>", unsafe_allow_html=True)

    agent_system_prompt = st.text_area(
        "System prompt (define personalidad, dominio y restricciones)",
        value=default_prompt,
        height=160,
        key="agent_system_prompt",
        help="Este prompt define quién es el agente. Edítalo libremente.",
    )

    if st.button("✅ Aplicar nueva personalidad y reiniciar conversación"):
        st.session_state["agent_messages"] = []
        st.session_state["agent_metrics_history"] = []
        st.success(f"✅ Agente **{agent_name}** listo. Conversación reiniciada.")
        st.rerun()

    st.markdown("---")

    # ── Inicializar estado ──
    if "agent_messages" not in st.session_state:
        st.session_state["agent_messages"] = []
    if "agent_metrics_history" not in st.session_state:
        st.session_state["agent_metrics_history"] = []

    # ── Renderizar historial ──
    current_name = st.session_state.get("agent_name_input", "Agente")
    current_prompt = st.session_state.get("agent_system_prompt", "")
    st.markdown(f"**🤖 {current_name}**")
    st.caption(f"System prompt activo: _{current_prompt[:120]}..._" if len(current_prompt) > 120 else f"System prompt activo: _{current_prompt}_")

    for msg in st.session_state["agent_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Input del usuario ──
    user_input = st.chat_input(f"Escribe tu mensaje para {current_name}...")

    if user_input:
        # Agregar mensaje del usuario
        st.session_state["agent_messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Preparar mensajes con system prompt dinámico
        full_messages = [{"role": "system", "content": current_prompt}] + \
                         st.session_state["agent_messages"]

        with st.chat_message("assistant"):
            with st.spinner(f"{current_name} está pensando..."):
                t0 = time.time()
                response_text, usage = call_llm(
                    full_messages,
                    temperature=agent_temp,
                    max_tokens=agent_max_tokens,
                )
                latency = time.time() - t0

            if response_text:
                st.markdown(response_text)
                st.session_state["agent_messages"].append(
                    {"role": "assistant", "content": response_text}
                )

                # ── Métricas de producción ──
                prompt_tok = usage.prompt_tokens if usage else 0
                comp_tok = usage.completion_tokens if usage else len(simple_tokenize(response_text))
                tps = comp_tok / latency if latency > 0 else 0
                cost = estimate_cost(prompt_tok, comp_tok)

                # LLM-Judge automático de la última respuesta
                judge_score = None
                if len(st.session_state["agent_messages"]) >= 2:
                    judge_prompt = f"""Evalúa esta respuesta de un agente conversacional del 1 al 10.
Criterios: precisión, claridad y utilidad para el usuario.
Responde SOLO un JSON: {{"score": <1-10>, "razon": "<breve>"}}

PREGUNTA: {user_input}
RESPUESTA: {response_text}"""
                    try:
                        j_text, _ = call_llm(
                            [{"role": "user", "content": judge_prompt}],
                            temperature=0.1,
                            max_tokens=100,
                        )
                        j_clean = re.sub(r"```json|```", "", j_text).strip()
                        j_data = json.loads(j_clean)
                        judge_score = j_data.get("score", None)
                    except Exception:
                        judge_score = None

                # Mostrar métricas inline
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                mc1.metric("⏱️ Latencia", f"{latency:.2f}s")
                mc2.metric("⚡ TPS", f"{tps:.1f}")
                mc3.metric("📥 In tokens", prompt_tok)
                mc4.metric("📤 Out tokens", comp_tok)
                mc5.metric("💰 Costo USD", f"${cost:.6f}")
                if judge_score:
                    st.metric("⚖️ LLM-Judge", f"{judge_score}/10")

                # Guardar en historial
                st.session_state["agent_metrics_history"].append({
                    "turn": len(st.session_state["agent_metrics_history"]) + 1,
                    "latency": round(latency, 3),
                    "tps": round(tps, 1),
                    "prompt_tokens": prompt_tok,
                    "comp_tokens": comp_tok,
                    "cost": round(cost, 7),
                    "judge_score": judge_score if judge_score else 0,
                })

    # ── Gráfica de historial de métricas ──
    if st.session_state["agent_metrics_history"]:
        st.markdown("---")
        st.subheader("📈 Historial de Métricas por Turno")
        hist = st.session_state["agent_metrics_history"]
        turns = [h["turn"] for h in hist]

        metric_choice = st.selectbox(
            "Métrica a visualizar",
            ["latency", "tps", "comp_tokens", "cost", "judge_score"],
            format_func=lambda x: {
                "latency": "Latencia (s)",
                "tps": "Tokens/segundo",
                "comp_tokens": "Tokens de salida",
                "cost": "Costo USD",
                "judge_score": "Puntuación LLM-Judge",
            }[x],
        )

        color_map = {
            "latency": "#FF6B6B",
            "tps": "#4ECDC4",
            "comp_tokens": "#FFE66D",
            "cost": "#F7DC6F",
            "judge_score": "#00FFB3",
        }

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=turns,
            y=[h[metric_choice] for h in hist],
            mode="lines+markers",
            line=dict(color=color_map[metric_choice], width=2),
            marker=dict(size=8),
        ))
        fig_hist.update_layout(
            xaxis_title="Turno de conversación",
            yaxis_title=metric_choice,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cdd9e5",
            xaxis=dict(dtick=1),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

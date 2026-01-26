import os
import re
import time
import difflib
from dataclasses import dataclass
from typing import Dict, List, Optional
import streamlit as st
from streamlit_option_menu import option_menu

# =========================
# IMPORTAÇÕES DE SEGURANÇA
# =========================
try:
    from veritas_utils import (
        extract_text_from_txt_bytes,
        extract_text_from_docx_bytes,
        extract_text_from_pdf_bytes,
        compute_matches,
        highlight_text,
    )
except ImportError:
    # Fallbacks para não quebrar o app se faltar arquivo
    def extract_text_from_txt_bytes(b): return b.decode("utf-8", errors="ignore")
    def extract_text_from_docx_bytes(b): return "Erro: veritas_utils não encontrado."
    def extract_text_from_pdf_bytes(b): return "Erro: veritas_utils não encontrado."
    def compute_matches(*args): return 0.0, []
    def highlight_text(t, m): return t

try:
    from veritas_report import generate_pdf_report, generate_web_pdf_report, generate_ai_pdf_report, generate_ai_docx_report
except ImportError:
    generate_pdf_report = None
    generate_web_pdf_report = None
    generate_ai_pdf_report = None
    generate_ai_docx_report = None

# =========================
# CONSTANTES
# =========================
APP_TITLE = "Veritas"
APP_SUBTITLE = "Análise de Similaridade e Integridade Acadêmica"
DISCL = "O Veritas aponta similaridade, não necessariamente plágio. A interpretação depende do contexto."
INTERNET_PRIVACY_NOTE = "🔒 **Privacidade**: Buscamos apenas fragmentos aleatórios na web, nunca o texto inteiro."
AI_HEURISTIC_NOTE = "🤖 Este detector busca padrões estatísticos. Use como indício, não como prova absoluta."

PROFILES = {
    "Padrão (Equilibrado)": {"chunk_words": 60, "stride_words": 25, "threshold": 0.75, "top_k_per_chunk": 1},
    "Rigoroso (Cópia Literal)": {"chunk_words": 80, "stride_words": 35, "threshold": 0.85, "top_k_per_chunk": 1},
    "Sensível (Paráfrase)": {"chunk_words": 40, "stride_words": 15, "threshold": 0.60, "top_k_per_chunk": 1},
}

# =========================
# FUNÇÕES UTILITÁRIAS
# =========================
def _read_any(uploaded_file) -> str:
    if not uploaded_file: return ""
    try:
        if uploaded_file.name.endswith(".txt"): return extract_text_from_txt_bytes(uploaded_file.getvalue())
        if uploaded_file.name.endswith(".docx"): return extract_text_from_docx_bytes(uploaded_file.getvalue())
        if uploaded_file.name.endswith(".pdf"): return extract_text_from_pdf_bytes(uploaded_file.getvalue())
    except Exception:
        return ""
    return ""

def _get_serpapi_key() -> Optional[str]:
    return st.secrets.get("SERPAPI_KEY") or os.getenv("SERPAPI_KEY")

def _split_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-zÀ-ÿ0-9]+", (text or "").lower())

def build_chunks(text: str, chunk_words: int, stride_words: int, max_chunks: int = 12) -> List[str]:
    words = _split_words(text)
    if not words: return []
    chunks = []
    i = 0
    while i < len(words) and len(chunks) < max_chunks:
        chunk = words[i : i + chunk_words]
        if len(chunk) >= max(12, chunk_words // 2):
            chunks.append(" ".join(chunk))
        i += stride_words
    return list(dict.fromkeys(chunks))

def seq_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()

@dataclass
class WebHit:
    title: str
    link: str
    snippet: str
    score: float
    chunk: str

def serpapi_search_chunk(chunk: str, serpapi_key: str, num_results: int = 5) -> List[Dict]:
    import requests
    q = f'"{chunk}"' if len(chunk) >= 80 else chunk
    try:
        r = requests.get("https://serpapi.com/search.json", params={"engine": "google", "q": q, "api_key": serpapi_key, "num": num_results, "hl": "pt", "gl": "br"}, timeout=20)
        return r.json().get("organic_results", []) or []
    except:
        return []

def web_similarity_scan(text, serpapi_key, profile_params, num_chunks, num_results):
    chunks = build_chunks(text, int(profile_params["chunk_words"]), int(profile_params["stride_words"]), num_chunks)
    raw_hits = []
    bar = st.progress(0)
    for i, c in enumerate(chunks):
        results = serpapi_search_chunk(c, serpapi_key, num_results)
        for it in results:
            sim = seq_similarity(c, it.get("snippet", ""))
            if sim > 0.1:
                raw_hits.append(WebHit(it.get("title", ""), it.get("link", ""), it.get("snippet", ""), sim, c))
        bar.progress((i + 1) / len(chunks))
    bar.empty()
    unique = {}
    for h in sorted(raw_hits, key=lambda x: x.score, reverse=True):
        if h.link not in unique: unique[h.link] = h
    return list(unique.values())[:20]

# --- NOVA FUNÇÃO DE IA DETALHADA ---
def analyze_ai_indicia(text: str) -> Dict:
    text = (text or "").strip()
    words = _split_words(text)
    if not words: return {"score": 0, "band": ("gray", "Indefinido"), "metrics": {"ttr": 0, "conn": 0}, "msg": "Texto insuficiente.", "reasons": []}
    
    # 1. Riqueza de Vocabulário (TTR)
    unique_ratio = len(set(words)) / len(words)
    
    # 2. Densidade de Conectores
    ai_conn = ["além disso", "em suma", "portanto", "todavia", "nesse sentido", "por outro lado", "vale ressaltar", "conclui-se", "resumo"]
    conn_count = sum(text.lower().count(c) for c in ai_conn)
    conn_density = (conn_count / len(words)) * 1000
    
    score = 0
    reasons = []
    
    # Penalidades explicadas
    if unique_ratio < 0.45: 
        score += 40
        reasons.append("Vocabulário muito repetitivo (típico de IA)")
    elif unique_ratio < 0.55:
        score += 15
        reasons.append("Pouca variedade de palavras")
        
    if conn_density > 12: 
        score += 40
        reasons.append("Uso excessivo de conectores lógicos (padrão de robô)")
    elif conn_density > 8:
        score += 15
        
    score = min(100, score)
    
    # Definição da Faixa
    if score < 30: 
        band = ("green", "Baixa Probabilidade de IA")
        msg = "O texto apresenta boa variedade vocabular e estrutura natural."
    elif score < 60: 
        band = ("yellow", "Indícios Moderados")
        msg = "O texto tem sinais mistos. Pode haver repetição de termos ou estrutura rígida."
    else: 
        band = ("red", "Alta Probabilidade de IA")
        msg = "O texto apresenta padrões fortes de geração automática: vocabulário pobre e conectores excessivos."

    return {
        "score": score, 
        "band": band, 
        "metrics": {"ttr": unique_ratio, "conn": conn_density}, 
        "msg": msg,
        "reasons": reasons
    }

def _inject_css():
    st.markdown("""
    <style>
    .header-container { padding: 20px; background: white; border-radius: 10px; margin-bottom: 20px; text-align: center; border: 1px solid #ddd; }
    .result-card { background: white; padding: 15px; border-radius: 10px; border: 1px solid #eee; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .disclaimer-box { background: #f8f9fa; padding: 10px; border-radius: 5px; font-size: 0.85rem; color: #666; margin-bottom: 15px; border-left: 3px solid #ccc; }
    </style>
    """, unsafe_allow_html=True)

# =========================
# APP PRINCIPAL
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🔍")

if "library" not in st.session_state: st.session_state["library"] = {}
if "last_result" not in st.session_state: st.session_state["last_result"] = None
if "internet_last" not in st.session_state: st.session_state["internet_last"] = None
if "ai_last" not in st.session_state: st.session_state["ai_last"] = None
if "profile" not in st.session_state: st.session_state["profile"] = "Padrão (Equilibrado)"

_inject_css()

st.markdown(f"""
<div class="header-container">
    <h1 style="color: #1e40af; margin:0;">🔍 {APP_TITLE}</h1>
    <p style="color: #64748b; margin:0;">{APP_SUBTITLE}</p>
</div>
""", unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=["Biblioteca", "Internet", "IA", "Relatórios", "Gerenciar"],
    icons=["folder", "globe", "robot", "file-text", "gear"],
    orientation="horizontal",
)

with st.sidebar:
    st.header("⚙️ Configurações")
    st.session_state["profile"] = st.selectbox("Perfil", list(PROFILES.keys()))
    if _get_serpapi_key(): st.success("✅ SerpAPI OK")
    else: st.warning("⚠️ SerpAPI Off")
    st.divider()
    st.caption("Allminds © 2026")

# --- 1. BIBLIOTECA (VISUAL MELHORADO) ---
if selected == "Biblioteca":
    st.subheader("📂 Comparação Local")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        tab_p, tab_u = st.tabs(["📝 Colar", "📁 Upload"])
        q_text = ""
        q_name = "Texto Inserido"
        with tab_p:
            t = st.text_area("Texto:", height=200)
            if t: q_text = t
        with tab_u:
            f = st.file_uploader("Arquivo (Biblio)", type=["docx", "pdf", "txt"], key="lib_up_input")
            if f: 
                q_text = _read_any(f)
                q_name = f.name
        
        if st.button("🔍 Comparar", type="primary", disabled=not q_text):
            if not st.session_state["library"]:
                st.error("Biblioteca vazia!")
            else:
                p = PROFILES[st.session_state["profile"]]
                with st.spinner("Analisando..."):
                    sim, matches = compute_matches(q_text, st.session_state["library"], p["chunk_words"], p["stride_words"], p["top_k_per_chunk"], p["threshold"])
                    st.session_state["last_result"] = {"sim": sim, "matches": matches, "name": q_name, "text": q_text}

    with c2:
        res = st.session_state["last_result"]
        if res:
            score = res['sim']
            if score < 0.03:
                color, status, expl = "green", "Original", "Nenhuma similaridade relevante."
            elif score < 0.25:
                color, status, expl = "yellow", "Atenção", "Pequenos trechos similares."
            else:
                color, status, expl = "red", "Alerta Alto", "Grande parte do texto já existe na biblioteca."

            st.markdown(f"""
            <div class="result-card" style="border-top: 5px solid {color}; text-align: center;">
                <h2 style="color: {color}; margin:0;">{score*100:.1f}%</h2>
                <p style="font-weight: bold; color: #555;">{status}</p>
                <hr style="margin: 10px 0;">
                <p style="font-size: 0.9rem; color: #666;">{expl}</p>
            </div>
            """, unsafe_allow_html=True)

            if res["matches"]:
                st.markdown("#### 🔍 Trechos:")
                for m in res["matches"][:5]:
                    with st.expander(f"{m.score*100:.0f}% - {m.source_doc}"):
                        st.markdown(f"**Texto:** ...{m.query_chunk}...")

# --- 2. INTERNET ---
elif selected == "Internet":
    st.subheader("🌐 Busca Web")
    st.markdown(f"<div class='disclaimer-box'>{INTERNET_PRIVACY_NOTE}</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.2])
    with c1:
        tab_web_p, tab_web_u = st.tabs(["📝 Colar", "📁 Upload"])
        w_text = ""
        with tab_web_p:
            t = st.text_area("Texto para Web:", height=200, key="web_in_paste")
            if t: w_text = t
        with tab_web_u:
            f = st.file_uploader("Arquivo (Web)", type=["docx", "pdf", "txt"], key="web_in_up")
            if f:
                w_text = _read_any(f)
                st.success(f"Lido: {f.name}")
        
        if st.button("Buscar na Web", type="primary", disabled=not w_text):
            if not _get_serpapi_key():
                st.error("Sem chave SerpAPI.")
            else:
                p = PROFILES[st.session_state["profile"]]
                with st.spinner("Buscando..."):
                    hits = web_similarity_scan(w_text, _get_serpapi_key(), p, 5, 5)
                    st.session_state["internet_last"] = {"hits": hits, "name": "Busca Web"}

    with c2:
        w_res = st.session_state["internet_last"]
        if w_res and w_res["hits"]:
            for h in w_res["hits"]:
                st.markdown(f"""<div class="result-card"><a href="{h.link}" target="_blank"><b>{h.title}</b></a><br><span style="color:red">{h.score*100:.0f}%</span> - {h.snippet}</div>""", unsafe_allow_html=True)
        elif w_res:
            st.success("

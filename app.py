import os
import re
import time
import difflib
from dataclasses import dataclass
from typing import Dict, List, Optional
import streamlit as st
from streamlit_option_menu import option_menu  # A MELHORIA: Biblioteca de menu

# =========================
# IMPORTAÇÕES DE MÓDULOS EXTERNOS
# =========================
# Tenta importar os módulos auxiliares (certifique-se de que veritas_utils.py existe na pasta)
try:
    from veritas_utils import (
        extract_text_from_txt_bytes,
        extract_text_from_docx_bytes,
        extract_text_from_pdf_bytes,
        compute_matches,
        highlight_text,
    )
except ImportError:
    # Fallbacks simples caso o arquivo não exista, para o app não quebrar
    def extract_text_from_txt_bytes(b): return b.decode("utf-8", errors="ignore")
    def extract_text_from_docx_bytes(b): return "Erro: veritas_utils não encontrado."
    def extract_text_from_pdf_bytes(b): return "Erro: veritas_utils não encontrado."
    def compute_matches(*args): return 0.0, []
    def highlight_text(t, m): return t

# Tenta importar módulos de relatório
try:
    from veritas_report import generate_pdf_report, generate_web_pdf_report, generate_ai_pdf_report, generate_ai_docx_report
except ImportError:
    generate_pdf_report = None
    generate_web_pdf_report = None
    generate_ai_pdf_report = None
    generate_ai_docx_report = None

# =========================
# CONFIGURAÇÕES E CONSTANTES
# =========================
APP_TITLE = "Veritas"
APP_SUBTITLE = "Análise de Similaridade e Integridade Acadêmica"

DISCL = (
    "O Veritas realiza análise automatizada de padrões textuais. "
    "O resultado **não é um veredito de plágio**, pois a integridade depende de citações, "
    "contexto e uso legítimo de fontes."
)

ETHICAL_NOTE = (
    "Similaridade nem sempre é plágio. Termos técnicos, bibliografia e citações diretas aumentam a taxa. "
    "Use esta ferramenta para apoiar sua revisão, não para acusar."
)

INTERNET_PRIVACY_NOTE = (
    "🔒 **Privacidade**: No modo Internet, enviamos apenas **fragmentos aleatórios** do texto para busca, "
    "nunca o documento inteiro. Ainda assim, evite usar com dados confidenciais."
)

AI_HEURISTIC_NOTE = (
    "🤖 **Análise Heurística de IA**\n\n"
    "Este módulo busca padrões estatísticos (repetição, pobreza vocabular, conectores excessivos). "
    "Ele aponta **indícios**, não provas. Textos humanos técnicos podem ser sinalizados, e IAs bem editadas podem passar. "
    "Use como guia para melhorar a naturalidade do texto."
)

PROFILES = {
    "Padrão (Equilibrado)": {"chunk_words": 60, "stride_words": 25, "threshold": 0.75, "top_k_per_chunk": 1},
    "Rigoroso (Cópia Literal)": {"chunk_words": 80, "stride_words": 35, "threshold": 0.85, "top_k_per_chunk": 1},
    "Sensível (Paráfrase)": {"chunk_words": 40, "stride_words": 15, "threshold": 0.60, "top_k_per_chunk": 1},
}

# =========================
# ESTILO CSS
# =========================
def _inject_css():
    st.markdown(
        """
        <style>
        /* Fonte e Cores Gerais */
        .main { background-color: #f8fafc; font-family: 'Segoe UI', sans-serif; }
        
        /* Cabeçalho Personalizado */
        .header-container {
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            text-align: center;
        }
        .header-title { color: #1e40af; font-size: 2.5rem; font-weight: 700; margin: 0; }
        .header-subtitle { color: #64748b; font-size: 1.1rem; margin-top: 5px; }

        /* Cards de Resultado */
        .result-card {
            background-color: white;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
            margin-bottom: 15px;
        }
        
        /* Pills e Badges */
        .pill {
            display: inline-block; padding: 4px 12px; border-radius: 99px;
            font-size: 0.85rem; font-weight: 600; margin-right: 8px;
        }
        .pill-green { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
        .pill-yellow { background: #fef9c3; color: #854d0e; border: 1px solid #fde047; }
        .pill-red { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }

        /* Avisos */
        .disclaimer-box {
            font-size: 0.85rem; color: #64748b; background: #f1f5f9;
            padding: 12px; border-radius: 8px; border-left: 4px solid #cbd5e1;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================
# UTILITÁRIOS INTERNOS
# =========================
def _read_any(uploaded_file) -> str:
    if not uploaded_file: return ""
    name = uploaded_file.name.lower()
    b = uploaded_file.getvalue()
    try:
        if name.endswith(".txt"): return extract_text_from_txt_bytes(b)
        if name.endswith(".docx"): return extract_text_from_docx_bytes(b)
        if name.endswith(".pdf"): return extract_text_from_pdf_bytes(b)
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")
        return ""
    return str(b)

def _get_band_color(score: float):
    if score < 0.15: return "green", "Baixa Similaridade", "Bom sinal de originalidade."
    if score < 0.40: return "yellow", "Atenção Moderada", "Verifique trechos comuns."
    return "red", "Alta Similaridade", "Revisão obrigatória necessária."

# =========================
# LÓGICA INTERNET (SERPAPI)
# =========================
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
    params = {
        "engine": "google", "q": q, "api_key": serpapi_key,
        "num": num_results, "hl": "pt", "gl": "br",
    }
    try:
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        return data.get("organic_results", []) or []
    except Exception:
        return []

def web_similarity_scan(text, serpapi_key, profile_params, num_chunks, num_results):
    chunks = build_chunks(text, int(profile_params["chunk_words"]), int(profile_params["stride_words"]), num_chunks)
    raw_hits = []
    
    progress_bar = st.progress(0)
    for i, c in enumerate(chunks):
        results = serpapi_search_chunk(c, serpapi_key, num_results)
        for it in results:
            title = it.get("title", "")
            link = it.get("link", "")
            snippet = it.get("snippet", "")
            sim = seq_similarity(c, snippet)
            if sim > 0.1:
                raw_hits.append(WebHit(title, link, snippet, sim, c))
        progress_bar.progress((i + 1) / len(chunks))
    progress_bar.empty()

    raw_hits.sort(key=lambda x: x.score, reverse=True)
    unique_hits = {}
    for h in raw_hits:
        if h.link not in unique_hits:
            unique_hits[h.link] = h
    
    return list(unique_hits.values())[:20]

# =========================
# LÓGICA IA (HEURÍSTICA)
# =========================
def analyze_ai_indicia(text: str) -> Dict:
    text = (text or "").strip()
    words = _split_words(text)
    if not words: return {"score": 0, "band": ("gray", "Indefinido"), "details": {}}

    unique_ratio = len(set(words)) / len(words)
    ai_connectors = ["além disso", "em suma", "portanto", "todavia", "nesse sentido", "por outro lado", "vale ressaltar"]
    conn_count = sum(text.lower().count(c) for c in ai_connectors)
    conn_density = (conn_count / len(words)) * 1000

    score = 0
    if unique_ratio < 0.4: score += 30
    if conn_density > 8: score += 30
    
    score = min(100, score)
    if score < 30: band = ("green", "Baixo Indício")
    elif score < 60: band = ("yellow", "Indício Moderado")
    else: band = ("red", "Indício Elevado")

    return {
        "score": score, "band": band,
        "metrics": {"ttr": unique_ratio, "conn": conn_density}
    }

# =========================
# INICIALIZAÇÃO DE ESTADO
# =========================
def _init_state():
    defaults = {"library": {}, "last_result": None, "profile": "Padrão (Equilibrado)", "internet_last": None, "ai_last": None}
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

# =========================
# INTERFACE PRINCIPAL
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🔍")
_init_state()
_inject_css()

# --- HEADER E NAVEGAÇÃO (AQUI ESTÁ A CORREÇÃO) ---
st.markdown(f"""
<div class="header-container">
    <div class="header-title">🔍 {APP_TITLE}</div>
    <div class="header-subtitle">{APP_SUBTITLE}</div>
</div>
""", unsafe_allow_html=True)

# Menu Horizontal Reativo
selected = option_menu(
    menu_title=None,
    options=["Biblioteca", "Internet", "IA", "Relatórios", "Gerenciar"],
    icons=["folder-symlink", "globe", "robot", "file-earmark-pdf", "gear"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#ffffff"},
        "icon": {"color": "#1e40af", "font-size": "16px"},
        "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#f1f5f9"},
        "nav-link-selected": {"background-color": "#1e40af", "color": "white"},
    }
)

# Barra Lateral (Configurações)
with st.sidebar:
    st.header("⚙️ Configurações")
    st.session_state["profile"] = st.selectbox("Perfil de Análise", list(PROFILES.keys()))
    params = PROFILES[st.session_state["profile"]]
    st.caption(f"Blocos de {params['chunk_words']} palavras | Precisão: {int(params['threshold']*100)}%")
    
    st.divider()
    st.markdown("### Status da API")
    if _get_serpapi_key():
        st.success("✅ SerpAPI Conectada")
    else:
        st.warning("⚠️ SerpAPI Off (Configure nos Secrets)")
    
    st.divider()
    st.info("Allminds © 2026")

# --- CONTEÚDO BASEADO NO MENU SELECIONADO ---

# 1. BIBLIOTECA LOCAL
if selected == "Biblioteca":
    st.subheader("📂 Comparação com Biblioteca Local")
    col_input, col_res = st.columns([1, 1.2], gap="large")
    
    with col_input:
        st.markdown("### 1. Texto para Análise")
        tab_paste, tab_upload = st.tabs(["📝 Colar Texto", "📁 Upload Arquivo"])
        
        query_text = ""
        query_name = "Texto Inserido"
        
        with tab_paste:
            text_paste = st.text_area("Cole aqui:", height=250)
            if text_paste: query_text = text_paste
        with tab_upload:
            file_upload = st.file_uploader("Word, PDF ou TXT", type=["docx", "pdf", "txt"])
            if file_upload:
                query_text = _read_any(file_upload)
                query_name = file_upload.name

        btn_analyze = st.button("🔍 Comparar", type="primary", use_container_width=True, disabled=not query_text)

    with col_res:
        st.markdown("### 2. Resultado")
        if btn_analyze:
            corpus = st.session_state["library"]
            if not corpus:
                st.error("Biblioteca vazia! Vá em 'Gerenciar' para adicionar arquivos.")
            else:
                p = PROFILES[st.session_state["profile"]]
                with st.spinner("Processando..."):
                    sim, matches = compute_matches(query_text, corpus, p["chunk_words"], p["stride_words"], p["top_k_per_chunk"], p["threshold"])
                    st.session_state["last_result"] = {"sim": sim, "matches": matches, "name": query_name, "text": query_text}

        res = st.session_state["last_result"]
        if res:
            color, label, desc = _get_band_color(res["sim"])
            st.markdown(f"""
            <div class="result-card" style="border-left: 5px solid {color};">
                <h2 style="color:{color}; margin:0;">{res['sim']*100:.1f}%</h2>
                <span class="pill pill-{color}">{label}</span>
                <p style="color:#64748b;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("Ver Detalhes dos Matches"):
                for m in res["matches"][:10]:
                    st.markdown(f"**Fonte:** {m.source_doc} | **Sim:** {m.score*100:.0f}%")
                    st.code(m.query_chunk)
        else:
            st.info("Aguardando análise.")

# 2. INTERNET
elif selected == "Internet":
    st.subheader("🌐 Busca na Web")
    st.markdown(f"<div class='disclaimer-box'>{INTERNET_PRIVACY_NOTE}</div>", unsafe_allow_html=True)
    
    col_web_in, col_web_out = st.columns([1, 1.2], gap="large")
    with col_web_in:
        web_text = st.text_area("Texto para busca na web:", height=200)
        btn_web = st.button("Buscar na Internet", type="primary", use_container_width=True, disabled=not web_text)
    
    with col_web_out:
        if btn_web:
            if not _get_serpapi_key():
                st.error("Chave SerpAPI não configurada.")
            else:
                p = PROFILES[st.session_state["profile"]]
                with st.spinner("Varrendo a internet..."):
                    hits = web_similarity_scan(web_text, _get_serpapi_key(), p, 5, 5)
                    st.session_state["internet_last"] = {"hits": hits, "name": "Busca Web"}
        
        web_res = st.session_state.get("internet_last")
        if web_res and web_res.get("hits"):
            for h in web_res["hits"]:
                st.markdown(f"**[{h.title}]({h.link})** - `{h.score*100:.0f}%`")
                st.caption(h.snippet)
                st.divider()
        elif web_res:
            st.success("Nada relevante encontrado.")

# 3. IA
elif selected == "IA":
    st.subheader("🤖 Detector de Padrões de IA")
    st.markdown(f"<div class='disclaimer-box'>{AI_HEURISTIC_NOTE}</div>", unsafe_allow_html=True)
    
    ai_text = st.text_area("Texto para análise:", height=250)
    if st.button("Verificar Padrões", type="primary"):
        res = analyze_ai_indicia(ai_text)
        st.session_state["ai_last"] = {"res": res, "name": "Texto IA"}
    
    ai_data = st.session_state.get("ai_last")
    if ai_data:
        res = ai_data["res"]
        color, label = res["band"]
        st.markdown(f"""
        <div class="result-card" style="border: 2px solid {color}; text-align:center;">
            <h3 style="color:{color}">{label} (Score: {res['score']})</h3>
            <p>Repetição Vocabular: {1.0 - res['metrics']['ttr']:.2f} | Conectores: {res['metrics']['conn']:.1f}</p>
        </div>
        """, unsafe_allow_html=True)

# 4. RELATÓRIOS
elif selected == "Relatórios":
    st.subheader("📊 Central de Downloads")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("**Biblioteca**")
        res_lib = st.session_state.get("last_result")
        if res_lib and generate_pdf_report:
            if st.button("Gerar PDF (Local)"):
                generate_pdf_report("Relatorio_Local.pdf", "Veritas Local", res_lib["name"], res_lib["sim"], res_lib["matches"], {}, DISCL)
                with open("Relatorio_Local.pdf", "rb") as f:
                    st.download_button("📥 Baixar PDF", f, "Relatorio_Local.pdf")
        else:
            st.caption("Sem dados.")

    with c2:
        st.markdown("**Internet**")
        st.caption("Disponível após busca web.")

    with c3:
        st.markdown("**IA**")
        st.caption("Disponível após análise IA.")

# 5. GERENCIAR
elif selected == "Gerenciar":
    st.subheader("📚 Gerenciar Banco de Dados")
    up_lib = st.file_uploader("Adicionar arquivos", type=["docx", "pdf", "txt"], accept_multiple_files=True)
    if up_lib:
        for f in up_lib:
            st.session_state["library"][f.name] = _read_any(f)
        st.success("Arquivos adicionados!")
    
    st.divider()
    if st.session_state["library"]:
        for name in list(st.session_state["library"].keys()):
            c1, c2 = st.columns([4, 1])
            c1.text(f"📄 {name}")
            if c2.button("🗑️", key=f"del_{name}"):
                del st.session_state["library"][name]
                st.rerun()

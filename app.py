import os
import re
import time
import difflib
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urlparse

import streamlit as st

# Tenta importar os módulos auxiliares (necessários para funcionar)
try:
    from veritas_utils import (
        extract_text_from_txt_bytes,
        extract_text_from_docx_bytes,
        extract_text_from_pdf_bytes,
        compute_matches,
        highlight_text,
    )
except ImportError:
    st.error("Erro: Arquivo 'veritas_utils.py' não encontrado.")
    st.stop()

# Tenta importar módulos de relatório (opcionais para não quebrar se faltar)
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
# ESTILO CSS (VISUAL PREMIUM)
# =========================
def _inject_css():
    st.markdown(
        """
        <style>
        /* Fonte e Cores Gerais */
        .main { background-color: #f8fafc; }
        h1 { color: #1e293b; font-weight: 800; letter-spacing: -1px; }
        h2, h3 { color: #334155; }
        
        /* Cards de Resultado */
        .result-card {
            background-color: white;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            margin-bottom: 15px;
        }
        
        /* Pills e Badges */
        .pill {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 99px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-right: 8px;
        }
        .pill-green { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
        .pill-yellow { background: #fef9c3; color: #854d0e; border: 1px solid #fde047; }
        .pill-red { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
        .pill-gray { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }

        /* Métricas */
        div[data-testid="stMetricValue"] { font-size: 2rem !important; color: #0f172a; }
        
        /* Avisos */
        .disclaimer-box {
            font-size: 0.85rem;
            color: #64748b;
            background: #f8fafc;
            padding: 10px;
            border-radius: 8px;
            border-left: 3px solid #cbd5e1;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================
# UTILITÁRIOS DE LEITURA
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
    return extract_text_from_txt_bytes(b) # Fallback

def _safe_words_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))

def _get_band_color(score: float):
    if score < 0.15: return "green", "Baixa Similaridade", "Bom sinal de originalidade."
    if score < 0.40: return "yellow", "Atenção Moderada", "Verifique trechos comuns."
    return "red", "Alta Similaridade", "Revisão obrigatória necessária."

# =========================
# LÓGICA INTERNET (SERPAPI)
# =========================
def _get_serpapi_key() -> Optional[str]:
    # Tenta pegar dos secrets ou env
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
    return list(dict.fromkeys(chunks)) # Remove duplicatas preservando ordem

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
    # Pesquisa exata se for longo, normal se for curto
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
    # Lógica simplificada de busca
    chunks = build_chunks(text, int(profile_params["chunk_words"]), int(profile_params["stride_words"]), num_chunks)
    raw_hits = []
    
    progress_bar = st.progress(0)
    for i, c in enumerate(chunks):
        results = serpapi_search_chunk(c, serpapi_key, num_results)
        for it in results:
            title = it.get("title", "")
            link = it.get("link", "")
            snippet = it.get("snippet", "")
            # Compara similaridade do chunk com o snippet encontrado
            sim = seq_similarity(c, snippet)
            if sim > 0.1: # Filtro mínimo
                raw_hits.append(WebHit(title, link, snippet, sim, c))
        progress_bar.progress((i + 1) / len(chunks))
    progress_bar.empty()

    # Ordena e remove duplicatas de link
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
    # Versão simplificada e robusta das heurísticas
    text = (text or "").strip()
    words = _split_words(text)
    if not words: return {"score": 0, "band": ("gray", "Indefinido"), "details": {}}

    # Métricas
    unique_ratio = len(set(words)) / len(words) # Riqueza vocabular
    
    # Conectores comuns de IA
    ai_connectors = ["além disso", "em suma", "portanto", "todavia", "nesse sentido", "por outro lado", "vale ressaltar"]
    conn_count = sum(text.lower().count(c) for c in ai_connectors)
    conn_density = (conn_count / len(words)) * 1000

    # Vagueza
    vague_words = ["importante", "fundamental", "crucial", "diversos", "vários", "alguns", "significativo"]
    vague_count = sum(text.lower().count(v) for v in vague_words)
    vague_density = (vague_count / len(words)) * 1000

    # Score (Algoritmo "Caseiro")
    score = 0
    if unique_ratio < 0.4: score += 30 # Vocabulário repetitivo
    if conn_density > 8: score += 30 # Muitos conectores
    if vague_density > 10: score += 20 # Muita vagueza
    
    # Normalização
    score = min(100, score)
    
    if score < 30: band = ("green", "Baixo Indício")
    elif score < 60: band = ("yellow", "Indício Moderado")
    else: band = ("red", "Indício Elevado")

    return {
        "score": score, "band": band,
        "metrics": {"ttr": unique_ratio, "conn": conn_density, "vague": vague_density}
    }

# =========================
# INICIALIZAÇÃO DE ESTADO
# =========================
def _init_state():
    defaults = {"library": {}, "library_meta": {}, "last_result": None, "profile": "Padrão (Equilibrado)", "internet_last": None, "ai_last": None}
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

# =========================
# INTERFACE PRINCIPAL
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="⚖️")
_init_state()
_inject_css()

# Cabeçalho
st.markdown(f"<h1>⚖️ {APP_TITLE} <span style='font-size:0.5em; color:#64748b; font-weight:400;'>| {APP_SUBTITLE}</span></h1>", unsafe_allow_html=True)

# Barra Lateral
with st.sidebar:
    st.header("Configurações")
    st.session_state["profile"] = st.selectbox("Perfil de Análise", list(PROFILES.keys()))
    
    params = PROFILES[st.session_state["profile"]]
    st.caption(f"Blocos de {params['chunk_words']} palavras | Precisão: {int(params['threshold']*100)}%")
    
    st.divider()
    st.markdown("### Status da API")
    if _get_serpapi_key():
        st.success("✅ SerpAPI Conectada")
    else:
        st.warning("⚠️ SerpAPI Desconectada (Modo Internet indisponível)")
    
    st.divider()
    st.info("Desenvolvido por **Allminds**")

# Tabs Principais
tabs = st.tabs(["🧪 Biblioteca (Local)", "🌐 Internet (Web)", "🤖 Detector de IA", "📚 Gerenciar Biblioteca"])

# --- TAB 1: BIBLIOTECA (LOCAL) ---
with tabs[0]:
    col_input, col_res = st.columns([1, 1.2], gap="large")
    
    with col_input:
        st.markdown("### 1. Texto para Análise")
        tab_paste, tab_upload = st.tabs(["Colar Texto", "Upload Arquivo"])
        
        with tab_paste:
            text_paste = st.text_area("Cole o conteúdo aqui:", height=250)
        with tab_upload:
            file_upload = st.file_uploader("Word, PDF ou TXT", type=["docx", "pdf", "txt"])
        
        query_text = ""
        query_name = "Texto Inserido"
        
        if text_paste:
            query_text = text_paste
        elif file_upload:
            query_text = _read_any(file_upload)
            query_name = file_upload.name

        btn_analyze = st.button("🔍 Comparar com Biblioteca", type="primary", use_container_width=True, disabled=not query_text)

    with col_res:
        st.markdown("### 2. Resultado")
        
        if btn_analyze:
            corpus = st.session_state["library"]
            if not corpus:
                st.error("Sua biblioteca está vazia! Adicione arquivos na aba 'Gerenciar Biblioteca'.")
            else:
                p = PROFILES[st.session_state["profile"]]
                with st.spinner("Processando similaridade..."):
                    sim, matches = compute_matches(query_text, corpus, p["chunk_words"], p["stride_words"], p["top_k_per_chunk"], p["threshold"])
                    
                    st.session_state["last_result"] = {
                        "sim": sim, "matches": matches, "name": query_name, "text": query_text
                    }

        # Exibir Resultados Armazenados
        res = st.session_state["last_result"]
        if res:
            color, label, desc = _get_band_color(res["sim"])
            
            # Card de Placar
            st.markdown(f"""
            <div class="result-card" style="border-left: 5px solid {color};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <h2 style="margin:0; color:{color};">{res['sim']*100:.1f}%</h2>
                        <span style="font-weight:bold; color:#334155;">Índice Global de Similaridade</span>
                    </div>
                    <div style="text-align:right;">
                        <span class="pill pill-{color}">{label}</span>
                        <p style="font-size:0.8rem; margin-top:5px; color:#64748b;">{desc}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Abas de Detalhe
            sub_t1, sub_t2 = st.tabs(["🔥 Destaques no Texto", "📋 Lista de Fontes"])
            
            with sub_t1:
                html_diff = highlight_text(res["text"], res["matches"])
                st.markdown(f'<div style="background:white; padding:15px; border-radius:8px; border:1px solid #eee; height:400px; overflow-y:scroll;">{html_diff}</div>', unsafe_allow_html=True)
                st.caption("Trechos em vermelho indicam alta similaridade com sua biblioteca.")

            with sub_t2:
                if not res["matches"]:
                    st.info("Nenhuma coincidência relevante encontrada.")
                else:
                    for m in res["matches"][:15]:
                        with st.expander(f"{m.score*100:.0f}% - {m.source_doc}"):
                            st.markdown(f"**Seu texto:** ...{m.query_chunk}...")
                            st.markdown(f"**Fonte:** ...{m.source_chunk}...")

            # Botão PDF
            if generate_pdf_report:
                pdf_path = f"Relatorio_Veritas_{int(time.time())}.pdf"
                generate_pdf_report(pdf_path, "Relatório Veritas (Local)", res["name"], res["sim"], res["matches"], {}, DISCL)
                with open(pdf_path, "rb") as f:
                    st.download_button("📥 Baixar Relatório PDF", f, file_name=os.path.basename(pdf_path))

# --- TAB 2: INTERNET ---
with tabs[1]:
    st.markdown("### Busca na Web (Anti-Plágio Externo)")
    st.markdown(f"<div class='disclaimer-box'>{INTERNET_PRIVACY_NOTE}</div>", unsafe_allow_html=True)
    
    col_web_in, col_web_out = st.columns([1, 1.2], gap="large")
    
    with col_web_in:
        web_text = st.text_area("Cole o texto para busca na web:", height=200, key="web_input")
        mode_web = st.selectbox("Intensidade da Busca", ["Rápida (5 trechos)", "Profunda (15 trechos)"])
        
        can_run = bool(_get_serpapi_key()) and bool(web_text)
        btn_web = st.button("🌐 Buscar na Internet", type="primary", disabled=not can_run)
        
        if not _get_serpapi_key():
            st.error("Chave SerpAPI não configurada.")

    with col_web_out:
        if btn_web:
            n_chunks = 5 if "Rápida" in mode_web else 15
            params = PROFILES[st.session_state["profile"]]
            with st.spinner("Varrendo a internet..."):
                hits = web_similarity_scan(web_text, _get_serpapi_key(), params, n_chunks, 5)
                st.session_state["internet_last"] = hits
        
        # Exibe Resultados
        hits = st.session_state.get("internet_last")
        if hits is not None:
            if not hits:
                st.success("Nenhuma similaridade significativa encontrada na web.")
            else:
                st.markdown("#### Principais Correspondências")
                for h in hits:
                    st.markdown(f"""
                    <div class="result-card" style="padding:10px; margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between;">
                            <a href="{h.link}" target="_blank" style="font-weight:bold; color:#1e40af; text-decoration:none;">{h.title}</a>
                            <span style="font-weight:bold; color:#ef4444;">{h.score*100:.0f}%</span>
                        </div>
                        <div style="font-size:0.85rem; color:#64748b; margin-top:5px;">{h.snippet}</div>
                    </div>
                    """, unsafe_allow_html=True)

# --- TAB 3: IA ---
with tabs[2]:
    st.markdown("### Análise Heurística de IA")
    st.markdown(f"<div class='disclaimer-box'>{AI_HEURISTIC_NOTE}</div>", unsafe_allow_html=True)
    
    ai_text = st.text_area("Cole o texto para análise de IA:", height=200)
    btn_ai = st.button("🤖 Verificar Padrões", type="primary", disabled=not ai_text)
    
    if btn_ai:
        res = analyze_ai_indicia(ai_text)
        color, label = res["band"]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Score Heurístico", f"{res['score']}/100")
        c2.metric("Repetição Vocabular", f"{1.0 - res['metrics']['ttr']:.2f}")
        c3.metric("Conectores de IA", f"{res['metrics']['conn']:.1f}")
        
        st.markdown(f"""
        <div class="result-card" style="text-align:center; border: 2px solid {color}; background-color: {'#f0fdf4' if color=='green' else '#fef2f2'};">
            <h3 style="color:{color}; margin:0;">{label}</h3>
            <p style="font-size:0.9rem; margin-top:5px;">Baseado em padrões de repetição e estrutura.</p>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 4: GERENCIAR BIBLIOTECA ---
with tabs[3]:
    st.markdown("### 📚 Banco de Dados Local")
    st.caption("Estes arquivos são usados para comparação na primeira aba.")
    
    up_lib = st.file_uploader("Adicionar à Biblioteca", type=["docx", "pdf", "txt"], accept_multiple_files=True)
    
    if up_lib:
        for f in up_lib:
            content = _read_any(f)
            if content:
                st.session_state["library"][f.name] = content
        st.success(f"{len(up_lib)} arquivos adicionados!")
    
    st.divider()
    
    # Listagem
    if st.session_state["library"]:
        st.write(f"Total de documentos: **{len(st.session_state['library'])}**")
        for name, text in list(st.session_state["library"].items()):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"📄 **{name}** ({len(text)} caracteres)")
            if c2.button("Remover", key=f"del_{name}"):
                del st.session_state["library"][name]
                st.rerun()
    else:
        st.info("Biblioteca vazia.")

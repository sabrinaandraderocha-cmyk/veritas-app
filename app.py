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
# ESTILO CSS (VISUAL TIPO COPYSPIDER)
# =========================
def _inject_css():
    st.markdown(
        """
        <style>
        /* Fonte e Cores Gerais */
        .main { background-color: #f0f2f5; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        h1 { color: #1e40af; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 0; }
        h2, h3 { color: #334155; }
        
        /* Barra de Ferramentas Superior */
        .toolbar {
            display: flex;
            align-items: center;
            padding: 10px 20px;
            background: #ffffff;
            border-bottom: 2px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .toolbar-title {
            font-size: 1.8rem;
            font-weight: bold;
            color: #1e40af;
            margin-right: 20px;
        }
        .toolbar-btn {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 8px 15px;
            margin-right: 10px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            background: #f8fafc;
            color: #475569;
            cursor: pointer;
            transition: all 0.2s;
        }
        .toolbar-btn:hover {
            background: #eef2ff;
            border-color: #1e40af;
            color: #1e40af;
        }
        .toolbar-btn-icon { font-size: 1.5rem; }
        .toolbar-btn-text { font-size: 0.8rem; font-weight: 600; }

        /* Área de Conteúdo */
        .content-panel {
            background: white;
            padding: 25px;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        /* Botões e Inputs */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
        }
        .stTextInput > div > div > input, .stTextArea > div > div > textarea {
            border-radius: 8px;
            border: 1px solid #cbd5e1;
        }
        
        /* Cards de Resultado */
        .result-card {
            background-color: white;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            margin-bottom: 15px;
            transition: transform 0.2s;
        }
        .result-card:hover { transform: translateY(-2px); }
        
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
        div[data-testid="stMetricValue"] { font-size: 2.2rem !important; color: #0f172a; font-weight: 700; }
        
        /* Avisos */
        .disclaimer-box {
            font-size: 0.85rem;
            color: #64748b;
            background: #f8fafc;
            padding: 12px;
            border-radius: 8px;
            border-left: 4px solid #cbd5e1;
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
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🔍")
_init_state()
_inject_css()

# Barra Superior (Estilo CopySpider)
st.markdown(f"""
<div class="toolbar">
    <div class="toolbar-title">🔍 {APP_TITLE}</div>
    <div style="display:flex;">
        <div class="toolbar-btn"><span class="toolbar-btn-icon">🧪</span><span class="toolbar-btn-text">Biblioteca</span></div>
        <div class="toolbar-btn"><span class="toolbar-btn-icon">🌐</span><span class="toolbar-btn-text">Internet</span></div>
        <div class="toolbar-btn"><span class="toolbar-btn-icon">🤖</span><span class="toolbar-btn-text">IA</span></div>
        <div class="toolbar-btn"><span class="toolbar-btn-icon">📊</span><span class="toolbar-btn-text">Relatórios</span></div>
    </div>
    <div style="flex-grow:1; text-align:right; font-size:0.9rem; color:#64748b;">{APP_SUBTITLE}</div>
</div>
""", unsafe_allow_html=True)


# Barra Lateral
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
        st.warning("⚠️ SerpAPI Desconectada (Modo Internet indisponível)")
    
    st.divider()
    st.info("Desenvolvido por **Allminds**")

# Tabs Principais
main_tabs = st.tabs(["🧪 Biblioteca (Local)", "🌐 Internet (Web)", "🤖 Detector de IA", "📊 Relatórios", "📚 Gerenciar Biblioteca"])

# --- TAB 1: BIBLIOTECA (LOCAL) ---
with main_tabs[0]:
    with st.container(border=False):
        st.markdown('<div class="content-panel">', unsafe_allow_html=True)
        col_input, col_res = st.columns([1, 1.2], gap="large")
        
        with col_input:
            st.markdown("### 1. Texto para Análise")
            tab_paste, tab_upload = st.tabs(["📝 Colar Texto", "📁 Upload Arquivo"])
            
            with tab_paste:
                text_paste = st.text_area("Cole o conteúdo aqui:", height=250, placeholder="Cole o texto do trabalho, artigo ou documento...")
            with tab_upload:
                file_upload = st.file_uploader("Word, PDF ou TXT", type=["docx", "pdf", "txt"], help="Arraste e solte seu arquivo aqui")
            
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
            else:
                st.info("Aguardando análise. Insira um texto e clique em 'Comparar'.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: INTERNET ---
with main_tabs[1]:
    with st.container(border=False):
        st.markdown('<div class="content-panel">', unsafe_allow_html=True)
        st.markdown("### Busca na Web (Anti-Plágio Externo)")
        st.markdown(f"<div class='disclaimer-box'>{INTERNET_PRIVACY_NOTE}</div>", unsafe_allow_html=True)
        
        col_web_in, col_web_out = st.columns([1, 1.2], gap="large")
        
        with col_web_in:
            st.markdown("#### Entrada de Dados")
            tab_paste_web, tab_upload_web = st.tabs(["📝 Colar Texto", "📁 Upload Arquivo"])
            
            web_text = ""
            web_query_name = "Texto Colado"

            with tab_paste_web:
                web_text_paste = st.text_area("Cole o texto para busca na web:", height=200, key="web_input_paste", placeholder="Cole aqui o texto a ser verificado...")
                if web_text_paste: web_text = web_text_paste

            with tab_upload_web:
                web_file_upload = st.file_uploader("Word, PDF ou TXT (Web)", type=["docx", "pdf", "txt"], key="web_input_upload", help="Arraste e solte seu arquivo aqui")
                if web_file_upload:
                    web_text = _read_any(web_file_upload)
                    web_query_name = web_file_upload.name

            mode_web = st.selectbox("Intensidade da Busca", ["Rápida (5 trechos)", "Profunda (15 trechos)"], help="Define quantos fragmentos do texto serão buscados.")
            
            can_run = bool(_get_serpapi_key()) and bool(web_text)
            btn_web = st.button("🌐 Buscar na Internet", type="primary", disabled=not can_run, use_container_width=True)
            
            if not _get_serpapi_key():
                st.error("Chave SerpAPI não configurada. Adicione-a nas configurações.")

        with col_web_out:
            st.markdown("#### Resultados da Web")
            if btn_web:
                n_chunks = 5 if "Rápida" in mode_web else 15
                params = PROFILES[st.session_state["profile"]]
                with st.spinner("Varrendo a internet em busca de similaridades..."):
                    hits = web_similarity_scan(web_text, _get_serpapi_key(), params, n_chunks, 5)
                    st.session_state["internet_last"] = {"hits": hits, "name": web_query_name}
            
            # Exibe Resultados
            web_res = st.session_state.get("internet_last")
            if web_res and web_res.get("hits") is not None:
                hits = web_res["hits"]
                if not hits:
                    st.success("Nenhuma similaridade significativa encontrada na web.")
                else:
                    st.markdown("#### Principais Correspondências")
                    for h in hits:
                        st.markdown(f"""
                        <div class="result-card" style="padding:15px; margin-bottom:10px;">
                            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                                <a href="{h.link}" target="_blank" style="font-weight:600; color:#1e40af; text-decoration:none; font-size:1.1rem;">{h.title}</a>
                                <span style="font-weight:bold; color:#ef4444; font-size:1.1rem;">{h.score*100:.0f}%</span>
                            </div>
                            <div style="font-size:0.9rem; color:#475569; margin-top:8px; line-height:1.4;">{h.snippet}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                 st.info("Aguardando busca. Insira um texto ou arquivo e clique em 'Buscar na Internet'.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: IA ---
with main_tabs[2]:
    with st.container(border=False):
        st.markdown('<div class="content-panel">', unsafe_allow_html=True)
        st.markdown("### Análise Heurística de IA")
        st.markdown(f"<div class='disclaimer-box'>{AI_HEURISTIC_NOTE}</div>", unsafe_allow_html=True)
        
        col_ai_in, col_ai_out = st.columns([1, 1.2], gap="large")
        
        with col_ai_in:
            st.markdown("#### Entrada de Dados")
            tab_paste_ai, tab_upload_ai = st.tabs(["📝 Colar Texto", "📁 Upload Arquivo"])
            
            ai_text = ""
            ai_query_name = "Texto Colado"

            with tab_paste_ai:
                ai_text_paste = st.text_area("Cole o texto para análise de IA:", height=200, key="ai_input_paste", placeholder="Cole aqui o texto para análise heurística...")
                if ai_text_paste: ai_text = ai_text_paste
            
            with tab_upload_ai:
                ai_file_upload = st.file_uploader("Word, PDF ou TXT (IA)", type=["docx", "pdf", "txt"], key="ai_input_upload", help="Arraste e solte seu arquivo aqui")
                if ai_file_upload:
                    ai_text = _read_any(ai_file_upload)
                    ai_query_name = ai_file_upload.name

            btn_ai = st.button("🤖 Verificar Padrões", type="primary", disabled=not ai_text, use_container_width=True)
        
        with col_ai_out:
            st.markdown("#### Resultado Heurístico")
            if btn_ai:
                with st.spinner("Analisando padrões de texto..."):
                    res = analyze_ai_indicia(ai_text)
                    st.session_state["ai_last"] = {"res": res, "name": ai_query_name}
            
            ai_data = st.session_state.get("ai_last")
            if ai_data:
                res = ai_data["res"]
                color, label = res["band"]
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Score Heurístico", f"{res['score']}/100")
                c2.metric("Repetição Vocabular", f"{1.0 - res['metrics']['ttr']:.2f}")
                c3.metric("Conectores de IA", f"{res['metrics']['conn']:.1f}")
                
                st.markdown(f"""
                <div class="result-card" style="text-align:center; border: 2px solid {color}; background-color: {'#f0fdf4' if color=='green' else '#fef2f2'}; padding: 20px;">
                    <h3 style="color:{color}; margin:0; font-size: 1.5rem;">{label}</h3>
                    <p style="font-size:1rem; margin-top:10px; color:#334155;">Baseado em padrões de repetição, vocabulário e estrutura.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Aguardando análise. Insira um texto ou arquivo e clique em 'Verificar Padrões'.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 4: RELATÓRIOS (NOVA) ---
with main_tabs[3]:
    with st.container(border=False):
        st.markdown('<div class="content-panel">', unsafe_allow_html=True)
        st.markdown("### 📊 Central de Relatórios")
        st.markdown("Baixe os resultados das suas últimas análises em formato PDF ou Word.")
        
        col_rep1, col_rep2, col_rep3 = st.columns(3, gap="medium")

        # Relatório Biblioteca
        with col_rep1:
            st.markdown("#### 🧪 Biblioteca (Local)")
            res_lib = st.session_state.get("last_result")
            if res_lib and generate_pdf_report:
                st.success(f"Análise disponível: {res_lib['name']}")
                pdf_path_lib = f"Relatorio_Veritas_Local_{int(time.time())}.pdf"
                generate_pdf_report(pdf_path_lib, "Relatório Veritas (Local)", res_lib["name"], res_lib["sim"], res_lib["matches"], {"Perfil": st.session_state["profile"]}, DISCL)
                with open(pdf_path_lib, "rb") as f:
                    st.download_button("📥 Baixar PDF (Biblioteca)", f, file_name=os.path.basename(pdf_path_lib), use_container_width=True)
            else:
                st.info("Nenhuma análise de biblioteca recente.")

        # Relatório Internet
        with col_rep2:
            st.markdown("#### 🌐 Internet (Web)")
            res_web = st.session_state.get("internet_last")
            if res_web and res_web.get("hits") and generate_web_pdf_report:
                st.success(f"Busca disponível: {res_web.get('name', 'Texto')}")
                pdf_path_web = f"Relatorio_Veritas_Web_{int(time.time())}.pdf"
                # Calcula um score global simples para o relatório
                global_web_score = sum(h.score for h in res_web["hits"][:5]) / max(1, len(res_web["hits"][:5])) if res_web["hits"] else 0
                generate_web_pdf_report(pdf_path_web, "Relatório Veritas (Web)", res_web.get('name', 'Texto'), st.session_state["profile"], global_web_score, res_web["hits"], DISCL)
                with open(pdf_path_web, "rb") as f:
                    st.download_button("📥 Baixar PDF (Internet)", f, file_name=os.path.basename(pdf_path_web), use_container_width=True)
            else:
                st.info("Nenhuma busca na internet recente.")

        # Relatório IA
        with col_rep3:
            st.markdown("#### 🤖 Detector de IA")
            res_ai_data = st.session_state.get("ai_last")
            if res_ai_data and generate_ai_pdf_report:
                st.success(f"Análise disponível: {res_ai_data['name']}")
                
                # PDF
                pdf_path_ai = f"Relatorio_Veritas_IA_{int(time.time())}.pdf"
                generate_ai_pdf_report(pdf_path_ai, "Relatório Veritas (IA)", res_ai_data["name"], res_ai_data["res"], AI_HEURISTIC_NOTE)
                with open(pdf_path_ai, "rb") as f:
                    st.download_button("📥 Baixar PDF (IA)", f, file_name=os.path.basename(pdf_path_ai), use_container_width=True, key="dl_pdf_ai_rep")
                
                # Word
                if generate_ai_docx_report:
                    docx_path_ai = f"Relatorio_Veritas_IA_{int(time.time())}.docx"
                    generate_ai_docx_report(docx_path_ai, "Relatório Veritas (IA)", res_ai_data["name"], res_ai_data["res"], AI_HEURISTIC_NOTE)
                    with open(docx_path_ai, "rb") as f:
                        st.download_button("📥 Baixar Word (IA)", f, file_name=os.path.basename(docx_path_ai), use_container_width=True, key="dl_docx_ai_rep")
            else:
                st.info("Nenhuma análise de IA recente.")

        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 5: GERENCIAR BIBLIOTECA ---
with main_tabs[4]:
    with st.container(border=False):
        st.markdown('<div class="content-panel">', unsafe_allow_html=True)
        st.markdown("### 📚 Gerenciar Banco de Dados Local")
        st.caption("Estes arquivos são usados para comparação na primeira aba 'Biblioteca (Local)'.")
        
        up_lib = st.file_uploader("Adicionar à Biblioteca", type=["docx", "pdf", "txt"], accept_multiple_files=True, help="Selecione múltiplos arquivos para compor sua base de comparação.")
        
        if up_lib:
            for f in up_lib:
                content = _read_any(f)
                if content:
                    st.session_state["library"][f.name] = content
            st.success(f"{len(up_lib)} arquivos adicionados com sucesso!")
        
        st.divider()
        
        # Listagem
        if st.session_state["library"]:
            st.markdown(f"#### Documentos na Biblioteca ({len(st.session_state['library'])})")
            for name, text in list(st.session_state["library"].items()):
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"📄 **{name}** ({len(text)} caracteres)")
                if c2.button("Remover", key=f"del_{name}", use_container_width=True):
                    del st.session_state["library"][name]
                    st.rerun()
            
            if st.button("⚠️ Limpar Biblioteca Inteira", type="primary", key="clear_lib"):
                st.session_state["library"] = {}
                st.rerun()
        else:
            st.info("Sua biblioteca está vazia. Adicione arquivos para começar.")
        st.markdown('</div>', unsafe_allow_html=True)

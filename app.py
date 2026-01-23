import os
import re
import time
import json
import math
import difflib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import streamlit as st

from veritas_utils import (
    extract_text_from_txt_bytes,
    extract_text_from_docx_bytes,
    extract_text_from_pdf_bytes,
    compute_matches,
    highlight_text,
)
from veritas_report import generate_pdf_report

# =========================
# CONFIG GERAL
# =========================
APP_TITLE = "Veritas"

DISCL = (
    "O Veritas realiza análise automatizada de similaridade textual. "
    "O resultado não configura, por si só, juízo definitivo sobre plágio acadêmico, "
    "o qual depende de avaliação contextual e humana (citações, paráfrases, domínio público, etc.)."
)

ETHICAL_NOTE = (
    "⚠️ Similaridade não é, por si só, falta ética. "
    "Trechos conceituais, metodologia, citações e fórmulas recorrentes podem elevar a correspondência. "
    "Use o resultado como apoio de revisão, não como veredito."
)

INTERNET_PRIVACY_NOTE = (
    "🔒 **Privacidade**: ao usar o modo Internet, o Veritas envia **apenas trechos curtos** do seu texto "
    "(e não o texto inteiro), para reduzir exposição. Mesmo assim, evite usar esse modo com textos sensíveis "
    "ou não publicados se isso for um risco para você."
)

# =========================
# PERFIS (substituem sliders)
# =========================
PROFILES = {
    "Rápido (padrão)": {"chunk_words": 60, "stride_words": 25, "threshold": 0.75, "top_k_per_chunk": 1},
    "Rigoroso (cópia literal)": {"chunk_words": 80, "stride_words": 35, "threshold": 0.82, "top_k_per_chunk": 1},
    "Sensível (paráfrase próxima)": {"chunk_words": 50, "stride_words": 20, "threshold": 0.66, "top_k_per_chunk": 1},
}

# =========================
# Leitura de arquivos
# =========================
def _read_any(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    b = uploaded_file.getvalue()
    if name.endswith(".txt"):
        return extract_text_from_txt_bytes(b)
    if name.endswith(".docx"):
        return extract_text_from_docx_bytes(b)
    if name.endswith(".pdf"):
        return extract_text_from_pdf_bytes(b)
    return extract_text_from_txt_bytes(b)

def _safe_words_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))

def _band(global_sim: float):
    if global_sim < 0.15:
        return "🟢 Similaridade esperada (baixa)", "Em geral, indica boa autonomia textual. Revise citações."
    if global_sim < 0.30:
        return "🟡 Atenção editorial (moderada)", "Pode refletir trechos comuns. Revise seções sinalizadas."
    return "🟠 Revisão cuidadosa (elevada)", "Não é acusação. Há sobreposição relevante: revise trechos e citações."

# =========================
# UX: CSS leve
# =========================
def _inject_css():
    st.markdown(
        """
        <style>
          .muted { opacity: 0.75; }
          .card {
            padding: 1rem; border-radius: 14px;
            border: 1px solid rgba(49,51,63,0.18);
            background: rgba(255,255,255,0.02);
          }
          .pill {
            display:inline-block; padding: 0.18rem 0.55rem; border-radius: 999px;
            border: 1px solid rgba(49,51,63,0.18); margin-right: 0.35rem;
          }
          .tight h3 { margin-bottom: 0.2rem; }
          .tight p { margin-top: 0.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================
# INTERNET: SerpAPI
# =========================
def _get_serpapi_key() -> Optional[str]:
    # Preferir secrets do Streamlit Cloud
    key = None
    try:
        key = st.secrets.get("SERPAPI_KEY", None)
    except Exception:
        key = None
    # fallback env
    if not key:
        key = os.getenv("SERPAPI_KEY")
    return key

def _split_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-zÀ-ÿ0-9]+", (text or "").lower())

def build_chunks(text: str, chunk_words: int, stride_words: int, max_chunks: int = 12) -> List[str]:
    """
    Gera chunks curtos para busca web (privacidade).
    Pega no máximo max_chunks trechos para não vazar demais e não estourar custo.
    """
    words = _split_words(text)
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words) and len(chunks) < max_chunks:
        chunk = words[i : i + chunk_words]
        if len(chunk) >= max(12, chunk_words // 2):
            chunks.append(" ".join(chunk))
        i += stride_words
    # Remove duplicados simples
    uniq = []
    seen = set()
    for c in chunks:
        k = c[:120]
        if k not in seen:
            uniq.append(c)
            seen.add(k)
    return uniq

def seq_similarity(a: str, b: str) -> float:
    """
    Similaridade simples (0..1) com difflib. Boa para texto curto/snippet.
    """
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

@dataclass
class WebHit:
    title: str
    link: str
    snippet: str
    score: float
    chunk: str

def serpapi_search_chunk(chunk: str, serpapi_key: str, num_results: int = 5) -> List[Dict]:
    """
    Busca via SerpAPI (Google Search API).
    Requer 'requests' no requirements.txt.
    """
    import requests  # lazy import

    params = {
        "engine": "google",
        "q": chunk,
        "api_key": serpapi_key,
        "num": num_results,
        "hl": "pt",
        "gl": "br",
    }
    r = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    results = data.get("organic_results", []) or []
    cleaned = []
    for it in results[:num_results]:
        cleaned.append(
            {
                "title": it.get("title", "") or "",
                "link": it.get("link", "") or "",
                "snippet": it.get("snippet", "") or "",
            }
        )
    return cleaned

def web_similarity_scan(text: str, serpapi_key: str, profile_params: dict, num_chunks: int = 10, num_results: int = 5) -> List[WebHit]:
    """
    Faz busca por chunks e calcula similaridade chunk vs snippet.
    Retorna hits ordenados por score.
    """
    chunks = build_chunks(
        text,
        chunk_words=int(profile_params["chunk_words"]),
        stride_words=int(profile_params["stride_words"]),
        max_chunks=num_chunks,
    )
    hits: List[WebHit] = []
    for c in chunks:
        try:
            results = serpapi_search_chunk(c, serpapi_key=serpapi_key, num_results=num_results)
        except Exception:
            continue
        for it in results:
            snippet = (it.get("snippet") or "")
            title = (it.get("title") or "")
            link = (it.get("link") or "")
            combined = f"{title}\n{snippet}".strip()
            score = seq_similarity(c, combined)
            hits.append(WebHit(title=title, link=link, snippet=snippet, score=score, chunk=c))
    hits.sort(key=lambda x: x.score, reverse=True)
    return hits

# =========================
# State
# =========================
def _init_state():
    if "library" not in st.session_state:
        st.session_state["library"] = {}  # name -> text
    if "library_meta" not in st.session_state:
        st.session_state["library_meta"] = {}  # name -> dict(tags, category, exclude)
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "profile" not in st.session_state:
        st.session_state["profile"] = "Rápido (padrão)"
    if "internet_last" not in st.session_state:
        st.session_state["internet_last"] = None

# =========================
# APP
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
_init_state()
_inject_css()

st.markdown(
    f"""
    <div class="tight">
      <h1>{APP_TITLE}</h1>
      <p class="muted">Análise de similaridade e integridade acadêmica</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown("**Observação ética**")
    st.caption(DISCL)
    st.caption(ETHICAL_NOTE)

# =========================
# Sidebar (agora útil)
# =========================
with st.sidebar:
    st.subheader("✨ Modo de análise")

    st.session_state["profile"] = st.selectbox(
        "Perfil",
        list(PROFILES.keys()),
        index=list(PROFILES.keys()).index(st.session_state["profile"]),
        help="Perfis substituem configurações técnicas. Use 'Rigoroso' para cópia literal e 'Sensível' para paráfrase próxima.",
    )

    st.caption(
        f"**{st.session_state['profile']}** → "
        f"trecho {PROFILES[st.session_state['profile']]['chunk_words']} palavras | "
        f"passo {PROFILES[st.session_state['profile']]['stride_words']} | "
        f"limiar {PROFILES[st.session_state['profile']]['threshold']}"
    )

    st.divider()
    st.subheader("🔒 Internet (opcional)")
    key = _get_serpapi_key()
    if key:
        st.success("SerpAPI key detectada ✅")
    else:
        st.warning("Sem SERPAPI_KEY. O modo Internet ficará indisponível.")

    st.caption("Recomendado: manter o padrão (Biblioteca). Use Internet só quando fizer sentido.")

tabs = st.tabs(["🧪 Biblioteca (privado)", "🌐 Internet (externo)", "📚 Biblioteca", "⚙️ Sobre"])

# =========================================================
# TAB 1: Biblioteca (privado)
# =========================================================
with tabs[0]:
    col1, col2 = st.columns([1.15, 0.85], gap="large")

    with col1:
        with st.container(border=True):
            st.subheader("Texto para análise")
            mode = st.radio("Como enviar o texto?", ["Colar texto", "Enviar arquivo"], horizontal=True)

            query_name = "Texto colado"
            query_text = ""

            if mode == "Colar texto":
                query_text = st.text_area(
                    "Cole o texto do trabalho/artigo:",
                    height=280,
                    placeholder="Cole seu texto aqui..."
                )
            else:
                up = st.file_uploader("Envie um arquivo (.docx, .pdf, .txt)", type=["docx", "pdf", "txt"])
                if up is not None:
                    query_name = up.name
                    try:
                        query_text = _read_any(up)
                    except Exception as e:
                        st.error(f"Não consegui ler o arquivo. Erro: {e}")

            st.divider()
            if not st.session_state["library"]:
                st.warning("Sua biblioteca está vazia. Vá em **Biblioteca** e adicione fontes para comparar.")

            run = st.button(
                "🔎 Analisar (biblioteca)",
                type="primary",
                use_container_width=True,
                disabled=(not query_text or not st.session_state["library"]),
            )

    with col2:
        with st.container(border=True):
            st.subheader("Resumo")
            wc = _safe_words_count(query_text)
            st.markdown(f"<span class='pill'>📄 {wc} palavras</span>", unsafe_allow_html=True)
            st.write("A comparação é feita contra documentos da sua **Biblioteca Veritas** (modo privado).")
            st.info("Dica: inclua trabalhos anteriores, artigos de referência, capítulos, etc.")

    if run:
        profile_params = PROFILES[st.session_state["profile"]]
        chunk_words = int(profile_params["chunk_words"])
        stride_words = int(profile_params["stride_words"])
        threshold = float(profile_params["threshold"])
        top_k_per_chunk = int(profile_params["top_k_per_chunk"])

        # corpus filtrado (excluir marcado)
        corpus = {}
        for name, text in st.session_state["library"].items():
            meta = st.session_state["library_meta"].get(name, {})
            if meta.get("exclude", False):
                continue
            corpus[name] = text

        if not corpus:
            st.error("Todos os documentos da biblioteca estão marcados como excluídos. Ajuste na aba **Biblioteca**.")
        else:
            with st.spinner("Analisando similaridade na sua biblioteca..."):
                global_sim, matches = compute_matches(
                    query_text=query_text,
                    corpus_docs=corpus,
                    chunk_words=chunk_words,
                    stride_words=stride_words,
                    top_k_per_chunk=top_k_per_chunk,
                    threshold=threshold,
                )

            st.session_state["last_result"] = {
                "query_name": query_name,
                "query_text": query_text,
                "global_sim": float(global_sim),
                "matches": matches,
                "params": {
                    "profile": st.session_state["profile"],
                    "chunk_words": chunk_words,
                    "stride_words": stride_words,
                    "threshold": threshold,
                    "top_k_per_chunk": top_k_per_chunk,
                },
                "corpus_size": len(corpus),
                "ts": int(time.time()),
            }

    res = st.session_state.get("last_result")
    if res:
        st.divider()
        st.subheader("Resultado (Biblioteca)")

        global_sim = float(res.get("global_sim", 0.0))
        band_title, band_msg = _band(global_sim)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Índice global", f"{global_sim*100:.1f}%")
        with m2:
            st.metric("Docs comparados", f"{res.get('corpus_size', 0)}")
        with m3:
            st.metric("Trechos sinalizados", f"{len(res.get('matches') or [])}")

        st.info(f"**{band_title}** — {band_msg}")

        left, right = st.columns([1, 1], gap="large")
        with left:
            with st.container(border=True):
                st.markdown("### Trechos sinalizados")
                matches = res.get("matches") or []
                if not matches:
                    st.success("Nenhuma correspondência acima do limiar foi encontrada.")
                else:
                    for i, m in enumerate(matches[:20], start=1):
                        st.markdown(f"**{i}.** `{m.source_doc}` — **{m.score*100:.1f}%**")
                        st.caption("Trecho analisado")
                        st.write(m.query_chunk)
                        st.caption("Trecho fonte")
                        st.write(m.source_chunk)
                        st.divider()

        with right:
            with st.container(border=True):
                st.markdown("### Texto com destaques")
                highlighted = highlight_text(res["query_text"], res.get("matches") or [])
                st.text_area("Destaques aparecem entre ⟦ ⟧", value=highlighted, height=420)

                st.markdown("### Relatório (PDF)")
                pdf_path = os.path.join(os.getcwd(), f"Relatorio_Veritas_{res.get('ts', int(time.time()))}.pdf")

                band_title, band_msg = _band(res["global_sim"])
                disclaimer_plus = (
                    DISCL
                    + "\n\n"
                    + ETHICAL_NOTE
                    + "\n\n"
                    + f"Perfil usado: {res['params'].get('profile')}\n"
                    + f"Leitura interpretativa (faixa): {band_title} — {band_msg}"
                )

                generate_pdf_report(
                    filepath=pdf_path,
                    title="Relatório de Análise de Similaridade – Veritas",
                    query_name=res["query_name"],
                    global_similarity=res["global_sim"],
                    matches=res.get("matches") or [],
                    params=res.get("params") or {},
                    disclaimer=disclaimer_plus,
                )

                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "⬇️ Baixar relatório em PDF",
                        data=f.read(),
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf",
                        use_container_width=True,
                    )

# =========================================================
# TAB 2: Internet (externo)
# =========================================================
with tabs[1]:
    st.subheader("🌐 Similaridade na Internet (modo externo)")

    st.markdown(INTERNET_PRIVACY_NOTE)

    serp_key = _get_serpapi_key()
    if not serp_key:
        st.error("Modo Internet indisponível: configure `SERPAPI_KEY` nos secrets.")
    else:
        with st.container(border=True):
            st.markdown("**Como funciona:** o Veritas envia *trechos curtos* do seu texto para busca e compara com snippets retornados.")
            consent = st.checkbox("✅ Eu entendo e aceito que trechos do meu texto serão enviados para busca na web.", value=False)

            st.divider()

            mode = st.radio("Como enviar o texto?", ["Colar texto", "Enviar arquivo"], horizontal=True)
            query_name = "Texto colado"
            query_text = ""

            if mode == "Colar texto":
                query_text = st.text_area(
                    "Cole o texto para checar na internet:",
                    height=240,
                    placeholder="Cole seu texto aqui..."
                )
            else:
                up = st.file_uploader("Envie um arquivo (.docx, .pdf, .txt)", type=["docx", "pdf", "txt"], key="internet_uploader")
                if up is not None:
                    query_name = up.name
                    try:
                        query_text = _read_any(up)
                    except Exception as e:
                        st.error(f"Não consegui ler o arquivo. Erro: {e}")

            # Controles úteis (simples)
            colA, colB = st.columns(2)
            with colA:
                num_chunks = st.slider("Quantidade de trechos enviados (menos = mais privado)", 3, 18, 10, 1)
            with colB:
                num_results = st.slider("Resultados por trecho", 3, 10, 5, 1)

            run_web = st.button(
                "🔎 Buscar na internet",
                type="primary",
                use_container_width=True,
                disabled=(not consent or not query_text),
            )

        if run_web:
            profile_params = PROFILES[st.session_state["profile"]]
            with st.spinner("Buscando na web (SerpAPI) e comparando snippets..."):
                hits = web_similarity_scan(
                    text=query_text,
                    serpapi_key=serp_key,
                    profile_params=profile_params,
                    num_chunks=int(num_chunks),
                    num_results=int(num_results),
                )

            st.session_state["internet_last"] = {
                "query_name": query_name,
                "profile": st.session_state["profile"],
                "hits": hits,
                "ts": int(time.time()),
                "num_chunks": int(num_chunks),
                "num_results": int(num_results),
            }

        webres = st.session_state.get("internet_last")
        if webres:
            st.divider()
            st.subheader("Resultado (Internet)")

            hits: List[WebHit] = webres["hits"] or []
            if not hits:
                st.warning("Não encontrei resultados relevantes (ou ocorreu erro de busca). Tente aumentar trechos/resultados.")
            else:
                # score global simples: média dos top hits (não é veredito)
                top = hits[:10]
                global_web = sum(h.score for h in top) / max(1, len(top))
                st.metric("Índice web (heurístico)", f"{global_web*100:.1f}%")
                st.caption("Este índice é apenas um sinal heurístico baseado em snippets, não uma prova.")

                st.markdown("### Principais correspondências encontradas")
                for i, h in enumerate(hits[:20], start=1):
                    st.markdown(f"**{i}. {h.title or '(sem título)'}** — **{h.score*100:.1f}%**")
                    if h.link:
                        st.write(h.link)
                    if h.snippet:
                        st.caption("Snippet da web")
                        st.write(h.snippet)
                    with st.expander("Trecho do seu texto enviado (chunk)", expanded=False):
                        st.write(h.chunk)
                    st.divider()

# =========================================================
# TAB 3: Biblioteca (upload)
# =========================================================
with tabs[2]:
    st.subheader("📚 Biblioteca Veritas")
    st.write("Os documentos aqui são as **fontes de comparação** (modo privado).")

    with st.container(border=True):
        up_lib = st.file_uploader(
            "Adicionar documentos (.docx, .pdf, .txt)",
            type=["docx", "pdf", "txt"],
            accept_multiple_files=True,
        )

        if up_lib:
            added = 0
            for f in up_lib:
                try:
                    st.session_state["library"][f.name] = _read_any(f)
                    st.session_state["library_meta"].setdefault(
                        f.name, {"tags": "", "category": "Referência", "exclude": False}
                    )
                    added += 1
                except Exception as e:
                    st.error(f"Falha ao ler {f.name}: {e}")
            if added:
                st.success(f"{added} documento(s) adicionados.")

    st.divider()
    if st.session_state["library"]:
        st.markdown("### Documentos na biblioteca")
        for name in list(st.session_state["library"].keys()):
            meta = st.session_state["library_meta"].setdefault(
                name, {"tags": "", "category": "Referência", "exclude": False}
            )

            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([0.42, 0.22, 0.24, 0.12], vertical_alignment="center")

                with c1:
                    st.write(f"📄 **{name}**")
                    st.caption(f"{_safe_words_count(st.session_state['library'][name])} palavras")

                with c2:
                    meta["category"] = st.selectbox(
                        "Categoria",
                        ["Referência", "Meu texto (autoria)", "Domínio público", "Modelo/metodologia", "Outros"],
                        index=["Referência", "Meu texto (autoria)", "Domínio público", "Modelo/metodologia", "Outros"].index(
                            meta.get("category", "Referência")
                            if meta.get("category") in ["Referência", "Meu texto (autoria)", "Domínio público", "Modelo/metodologia", "Outros"]
                            else "Referência"
                        ),
                        key=f"cat_{name}",
                    )

                with c3:
                    meta["tags"] = st.text_input(
                        "Tags",
                        value=meta.get("tags", ""),
                        placeholder="ex.: penal; rogers; método",
                        key=f"tags_{name}",
                    )
                    meta["exclude"] = st.checkbox(
                        "Excluir da comparação",
                        value=bool(meta.get("exclude", False)),
                        key=f"exc_{name}",
                        help="Exclui do cálculo e das buscas na biblioteca.",
                    )

                with c4:
                    if st.button("Remover", key=f"rm_{name}", use_container_width=True):
                        st.session_state["library"].pop(name, None)
                        st.session_state["library_meta"].pop(name, None)
                        st.rerun()
    else:
        st.info("Ainda não há documentos na biblioteca.")

# =========================================================
# TAB 4: Sobre
# =========================================================
with tabs[3]:
    st.subheader("⚙️ Sobre o Veritas")
    st.markdown(
        """
- **Modo Biblioteca (privado):** compara apenas com os documentos que você adicionou.
- **Modo Internet (externo):** usa SerpAPI para buscar *trechos curtos* e comparar com snippets da web.
- **Importante:** nenhum modo “prova plágio”; serve como apoio de revisão e integridade acadêmica.
        """
    )
    st.caption(DISCL)
    st.caption(ETHICAL_NOTE)

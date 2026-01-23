import os
import re
import time
import difflib
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urlparse

import streamlit as st

from veritas_utils import (
    extract_text_from_txt_bytes,
    extract_text_from_docx_bytes,
    extract_text_from_pdf_bytes,
    compute_matches,
    highlight_text,
)
from veritas_report import (
    generate_pdf_report,
    generate_web_pdf_report,
    generate_ai_pdf_report,
)

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
    "Similaridade não é, por si só, falta ética. "
    "Trechos conceituais, metodologia, citações e fórmulas recorrentes podem elevar a correspondência. "
    "Use o resultado como apoio de revisão, não como veredito."
)

INTERNET_PRIVACY_NOTE = (
    "🔒 **Privacidade**: ao usar o modo Internet, o Veritas envia **apenas trechos curtos** do seu texto "
    "(e não o texto inteiro), para reduzir exposição. Mesmo assim, evite usar esse modo com textos sensíveis "
    "ou não publicados se isso for um risco para você."
)

AI_HEURISTIC_NOTE = (
    "⚠️ **Ressalva importante**\n\n"
    "Este módulo **não comprova autoria** nem “detecta IA” com certeza. Ele apresenta **indícios heurísticos** "
    "(padrões linguísticos e estatísticos) que **podem ocorrer tanto em textos humanos quanto em textos gerados "
    "ou assistidos por IA**.\n\n"
    "Use o resultado **exclusivamente como apoio à revisão**: fortalecer exemplos, fontes, precisão e marcas autorais."
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
          code { white-space: pre-wrap; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# INTERNET: SerpAPI (com ranking melhorado)
# =========================
def _get_serpapi_key() -> Optional[str]:
    key = None
    try:
        key = st.secrets.get("SERPAPI_KEY", None)
    except Exception:
        key = None
    if not key:
        key = os.getenv("SERPAPI_KEY")
    return key


def _split_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-zÀ-ÿ0-9]+", (text or "").lower())


def build_chunks(text: str, chunk_words: int, stride_words: int, max_chunks: int = 12) -> List[str]:
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

    uniq = []
    seen = set()
    for c in chunks:
        k = c[:120]
        if k not in seen:
            uniq.append(c)
            seen.add(k)
    return uniq


def seq_similarity(a: str, b: str) -> float:
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


TRUST_BOOST_DOMAINS = [
    "scielo", "periodicos.capes", "pubmed", "ncbi.nlm.nih.gov",
    "doi.org", "springer", "wiley", "tandfonline", "elsevier", "sciencedirect",
    "jstor", "cambridge", "oxford", "sagepub", "nature.com", "science.org",
    "ieee.org", "acm.org"
]

PENALIZE_DOMAINS = [
    "brainly", "passeidireto", "scribd", "docsity", "monografias", "trabalhosprontos",
    "resumos", "blogspot", "wordpress", "medium.com", "reddit.com"
]


def _domain_of(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.replace("www.", "")
    except Exception:
        return ""


def _domain_weight(domain: str) -> float:
    d = (domain or "").lower()
    if not d:
        return 1.0
    if d.endswith(".edu") or d.endswith(".gov"):
        return 1.25
    for t in TRUST_BOOST_DOMAINS:
        if t in d:
            return 1.18
    for p in PENALIZE_DOMAINS:
        if p in d:
            return 0.82
    return 1.0


def _snippet_quality(snippet: str) -> float:
    s = (snippet or "").strip()
    n = len(s)
    if n < 60:
        return 0.85
    if n < 120:
        return 0.95
    if n < 220:
        return 1.00
    return 1.06


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def serpapi_search_chunk(chunk: str, serpapi_key: str, num_results: int = 5) -> List[Dict]:
    """
    Busca via SerpAPI (Google Search API).
    Requer 'requests' no requirements.txt.
    """
    import requests  # lazy import

    q = f"\"{chunk}\"" if len(chunk) >= 80 else chunk
    params = {
        "engine": "google",
        "q": q,
        "api_key": serpapi_key,
        "num": num_results,
        "hl": "pt",
        "gl": "br",
    }

    r = requests.get("https://serpapi.com/search.json", params=params, timeout=25)
    r.raise_for_status()
    data = r.json()

    results = data.get("organic_results", []) or []
    cleaned = []
    for it in results[:num_results]:
        cleaned.append(
            {
                "title": (it.get("title") or "").strip(),
                "link": (it.get("link") or "").strip(),
                "snippet": (it.get("snippet") or "").strip(),
            }
        )
    return cleaned


def web_similarity_scan(
    text: str,
    serpapi_key: str,
    profile_params: dict,
    num_chunks: int = 10,
    num_results: int = 5,
    max_final_hits: int = 20,
) -> List[WebHit]:
    chunks = build_chunks(
        text,
        chunk_words=int(profile_params["chunk_words"]),
        stride_words=int(profile_params["stride_words"]),
        max_chunks=num_chunks,
    )

    raw_hits = []
    for c in chunks:
        try:
            results = serpapi_search_chunk(c, serpapi_key=serpapi_key, num_results=num_results)
        except Exception:
            continue

        for it in results:
            title = it.get("title", "") or ""
            link = it.get("link", "") or ""
            snippet = it.get("snippet", "") or ""

            combined = f"{title}\n{snippet}".strip()
            sim = seq_similarity(c, combined)

            domain = _domain_of(link)
            w_dom = _domain_weight(domain)
            w_snip = _snippet_quality(snippet)

            score_final = _clamp(sim * w_dom * w_snip, 0.0, 1.0)

            raw_hits.append(
                {"title": title, "link": link, "snippet": snippet, "domain": domain, "score": score_final, "chunk": c}
            )

    best_by_link = {}
    for h in raw_hits:
        link = h["link"]
        if not link:
            continue
        if link not in best_by_link or h["score"] > best_by_link[link]["score"]:
            best_by_link[link] = h

    deduped = list(best_by_link.values())
    deduped.sort(key=lambda x: x["score"], reverse=True)
    deduped = deduped[:max_final_hits]

    hits: List[WebHit] = []
    for h in deduped:
        title = h["title"]
        domain = h["domain"]
        if domain:
            title = f"{title}  ({domain})"
        hits.append(WebHit(title=title, link=h["link"], snippet=h["snippet"], score=h["score"], chunk=h["chunk"]))
    return hits


# =========================
# INDÍCIOS DE USO DE IA (HEURÍSTICO)
# =========================
AI_CONNECTORS = [
    "além disso", "dessa forma", "nesse sentido", "por fim", "em suma", "portanto",
    "assim", "logo", "contudo", "entretanto", "todavia", "outrossim", "desse modo",
    "vale destacar", "é importante destacar", "cabe ressaltar"
]

AI_VAGUE_WORDS = [
    "importante", "relevante", "significativo", "notável", "essencial", "fundamental",
    "diversos", "vários", "muitos", "alguns", "inúmeros", "de certa forma",
    "em geral", "de modo geral", "de maneira geral"
]

def _sentences(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    parts = re.split(r"(?<=[\.\!\?])\s+|\n+", t)
    return [p.strip() for p in parts if p.strip()]

def _tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-zÀ-ÿ0-9]+", (text or "").lower())

def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    m = sum(values) / len(values)
    v = sum((x - m) ** 2 for x in values) / max(1, len(values))
    return v ** 0.5

def analyze_ai_indicia(text: str) -> Dict:
    t = (text or "").strip()
    toks = _tokens(t)
    sents = _sentences(t)

    word_count = len(toks)
    sent_word_lens = [len(_tokens(s)) for s in sents if len(_tokens(s)) > 0]

    unique = len(set(toks)) if toks else 0
    ttr = (unique / word_count) if word_count else 0.0

    mean_sent = (sum(sent_word_lens) / len(sent_word_lens)) if sent_word_lens else 0.0
    std_sent = _std([float(x) for x in sent_word_lens]) if sent_word_lens else 0.0
    cv_sent = (std_sent / mean_sent) if mean_sent > 0 else 0.0

    low = t.lower()
    conn_hits = sum(len(re.findall(rf"\b{re.escape(c)}\b", low)) for c in AI_CONNECTORS)
    vague_hits = sum(len(re.findall(rf"\b{re.escape(v)}\b", low)) for v in AI_VAGUE_WORDS)

    conn_per_1k = (conn_hits / max(1, word_count)) * 1000.0
    vague_per_1k = (vague_hits / max(1, word_count)) * 1000.0

    rep = 0
    for i in range(2, len(toks)):
        if toks[i] == toks[i-1] or toks[i] == toks[i-2]:
            rep += 1
    rep_per_1k = (rep / max(1, word_count)) * 1000.0

    score = 0.0
    if cv_sent > 0:
        score += _clamp((0.55 - cv_sent) / 0.55, 0.0, 1.0) * 30.0
    else:
        score += 10.0

    score += _clamp(conn_per_1k / 10.0, 0.0, 1.0) * 20.0
    score += _clamp(vague_per_1k / 18.0, 0.0, 1.0) * 20.0
    score += _clamp(rep_per_1k / 12.0, 0.0, 1.0) * 15.0

    if ttr > 0:
        score += _clamp((0.33 - ttr) / 0.33, 0.0, 1.0) * 15.0

    score = _clamp(score, 0.0, 100.0)

    if score < 33:
        band = ("🟢 Baixa", "Poucos indícios de padronização. Ainda assim, revise precisão, fontes e exemplos.")
    elif score < 66:
        band = ("🟡 Moderada", "Há sinais de padronização. Reforce exemplos, especificidade e voz autoral.")
    else:
        band = ("🟠 Elevada", "Sinais mais fortes de padronização. Revise conectores, generalidades e detalhe empírico.")

    flagged_sentences = []
    for s in sents[:400]:
        sl = s.lower()
        c = sum(1 for x in AI_CONNECTORS if x in sl)
        v = sum(1 for x in AI_VAGUE_WORDS if x in sl)
        if (c + v) >= 2 and len(_tokens(s)) >= 10:
            flagged_sentences.append(s)

    return {
        "score": float(score),
        "band": band,
        "word_count": word_count,
        "sent_count": len(sents),
        "ttr": float(ttr),
        "mean_sent": float(mean_sent),
        "cv_sent": float(cv_sent),
        "conn_per_1k": float(conn_per_1k),
        "vague_per_1k": float(vague_per_1k),
        "rep_per_1k": float(rep_per_1k),
        "flagged_sentences": flagged_sentences[:12],
    }


# =========================
# State
# =========================
def _init_state():
    if "library" not in st.session_state:
        st.session_state["library"] = {}
    if "library_meta" not in st.session_state:
        st.session_state["library_meta"] = {}
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "profile" not in st.session_state:
        st.session_state["profile"] = "Rápido (padrão)"
    if "internet_last" not in st.session_state:
        st.session_state["internet_last"] = None
    if "ai_last" not in st.session_state:
        st.session_state["ai_last"] = None


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

with st.sidebar:
    st.subheader("Modo de análise")
    st.session_state["profile"] = st.selectbox(
        "Perfil",
        list(PROFILES.keys()),
        index=list(PROFILES.keys()).index(st.session_state["profile"]),
    )
    st.caption(
        f"{st.session_state['profile']} → "
        f"trecho {PROFILES[st.session_state['profile']]['chunk_words']} | "
        f"passo {PROFILES[st.session_state['profile']]['stride_words']} | "
        f"limiar {PROFILES[st.session_state['profile']]['threshold']}"
    )
    st.divider()
    st.subheader("Internet (opcional)")
    key = _get_serpapi_key()
    if key:
        st.success("SerpAPI key detectada ✅")
    else:
        st.caption("Modo Internet indisponível (SERPAPI_KEY não configurada).")

tabs = st.tabs([
    "🧪 Biblioteca (privado)",
    "🌐 Internet (externo)",
    "🤖 Indícios de Uso de IA (análise heurística)",
    "📚 Biblioteca",
    "⚙️ Sobre",
])

# =========================================================
# TAB 1: Biblioteca (privado)
# =========================================================
with tabs[0]:
    col1, col2 = st.columns([1.15, 0.85], gap="large")

    with col1:
        with st.container(border=True):
            st.subheader("Texto para análise")
            mode = st.radio(
                "Como enviar o texto?",
                ["Colar texto", "Enviar arquivo"],
                horizontal=True,
                key="radio_biblioteca_envio",
            )
            query_name = "Texto colado"
            query_text = ""

            if mode == "Colar texto":
                query_text = st.text_area("Cole o texto do trabalho/artigo:", height=280, placeholder="Cole seu texto aqui...")
            else:
                up = st.file_uploader("Envie um arquivo (.docx, .pdf, .txt)", type=["docx", "pdf", "txt"], key="upl_biblioteca")
                if up is not None:
                    query_name = up.name
                    query_text = _read_any(up)

            st.divider()
            run = st.button("🔎 Analisar (biblioteca)", type="primary", use_container_width=True,
                            disabled=(not query_text or not st.session_state["library"]))

    with col2:
        with st.container(border=True):
            st.subheader("Resumo")
            wc = _safe_words_count(query_text)
            st.markdown(f"<span class='pill'>📄 {wc} palavras</span>", unsafe_allow_html=True)
            st.write("Comparação contra sua **Biblioteca Veritas** (modo privado).")

    if run:
        profile_params = PROFILES[st.session_state["profile"]]
        corpus = {n: t for n, t in st.session_state["library"].items()
                  if not st.session_state["library_meta"].get(n, {}).get("exclude", False)}

        global_sim, matches = compute_matches(
            query_text=query_text,
            corpus_docs=corpus,
            chunk_words=int(profile_params["chunk_words"]),
            stride_words=int(profile_params["stride_words"]),
            top_k_per_chunk=int(profile_params["top_k_per_chunk"]),
            threshold=float(profile_params["threshold"]),
        )

        st.session_state["last_result"] = {
            "query_name": query_name,
            "query_text": query_text,
            "global_sim": float(global_sim),
            "matches": matches,
            "params": {"profile": st.session_state["profile"], **profile_params},
            "corpus_size": len(corpus),
            "ts": int(time.time()),
        }

    res = st.session_state.get("last_result")
    if res:
        st.divider()
        st.subheader("Resultado (Biblioteca)")

        global_sim = float(res.get("global_sim", 0.0))
        band_title, band_msg = _band(global_sim)
        st.info(f"**{band_title}** — {band_msg}")

        left, right = st.columns([1, 1], gap="large")
        with left:
            matches = res.get("matches") or []
            for i, m in enumerate(matches[:20], start=1):
                st.markdown(f"**{i}.** `{m.source_doc}` — **{m.score*100:.1f}%**")
                st.write(m.query_chunk)
                st.write(m.source_chunk)
                st.divider()

        with right:
            highlighted = highlight_text(res["query_text"], res.get("matches") or [])
            st.text_area("Destaques (⟦ ⟧)", value=highlighted, height=360)

            # PDF Biblioteca (já tinha)
            pdf_path = os.path.join(os.getcwd(), f"Relatorio_Veritas_{res.get('ts', int(time.time()))}.pdf")
            generate_pdf_report(
                filepath=pdf_path,
                title="Relatório de Análise de Similaridade – Veritas",
                query_name=res["query_name"],
                global_similarity=res["global_sim"],
                matches=res.get("matches") or [],
                params=res.get("params") or {},
                disclaimer=DISCL + "\n\n" + ETHICAL_NOTE,
            )
            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ Baixar relatório (Biblioteca) em PDF", data=f.read(),
                                   file_name=os.path.basename(pdf_path), mime="application/pdf", use_container_width=True,
                                   key="dl_pdf_bib")

# =========================================================
# TAB 2: Internet (externo)
# =========================================================
with tabs[1]:
    st.subheader("Similaridade na Internet (modo externo)")
    st.markdown(INTERNET_PRIVACY_NOTE)

    serp_key = _get_serpapi_key()
    if not serp_key:
        st.error("Modo Internet indisponível: configure `SERPAPI_KEY` nos secrets.")
    else:
        consent = st.checkbox("✅ Eu entendo e aceito que trechos do meu texto serão enviados para busca na web.", value=False, key="internet_consent")
        mode = st.radio("Como enviar o texto?", ["Colar texto", "Enviar arquivo"], horizontal=True, key="radio_internet_envio")

        query_name = "Texto colado"
        query_text = ""

        if mode == "Colar texto":
            query_text = st.text_area("Cole o texto para checar na internet:", height=220, key="internet_text")
        else:
            up = st.file_uploader("Envie um arquivo (.docx, .pdf, .txt)", type=["docx", "pdf", "txt"], key="internet_uploader")
            if up is not None:
                query_name = up.name
                query_text = _read_any(up)

        colA, colB = st.columns(2)
        with colA:
            num_chunks = st.slider("Trechos enviados (menos = mais privado)", 3, 18, 10, 1, key="internet_chunks")
        with colB:
            num_results = st.slider("Resultados por trecho", 3, 10, 5, 1, key="internet_results")

        run_web = st.button("🔎 Buscar na internet", type="primary", use_container_width=True,
                            disabled=(not consent or not query_text), key="btn_web")

        if run_web:
            profile_params = PROFILES[st.session_state["profile"]]
            hits = web_similarity_scan(
                text=query_text,
                serpapi_key=serp_key,
                profile_params=profile_params,
                num_chunks=int(num_chunks),
                num_results=int(num_results),
                max_final_hits=20,
            )
            st.session_state["internet_last"] = {
                "query_name": query_name,
                "profile": st.session_state["profile"],
                "hits": hits,
                "ts": int(time.time()),
            }

        webres = st.session_state.get("internet_last")
        if webres:
            hits: List[WebHit] = webres["hits"] or []
            top = hits[:10]
            global_web = sum(h.score for h in top) / max(1, len(top))
            st.metric("Índice web (heurístico)", f"{global_web*100:.1f}%")

            for i, h in enumerate(hits[:20], start=1):
                st.markdown(f"**{i}. {h.title}** — **{h.score*100:.1f}%**")
                st.write(h.link)
                st.write(h.snippet)
                with st.expander("Chunk enviado"):
                    st.write(h.chunk)
                st.divider()

            # PDF Internet
            pdf_path_web = os.path.join(os.getcwd(), f"Relatorio_Veritas_Internet_{webres.get('ts', int(time.time()))}.pdf")
            generate_web_pdf_report(
                filepath=pdf_path_web,
                title="Relatório de Similaridade – Internet (Veritas)",
                query_name=webres.get("query_name", "—"),
                profile=webres.get("profile", "—"),
                global_web_score=global_web,
                hits=hits,
                disclaimer=(
                    "Este relatório é baseado em snippets e resultados públicos retornados por busca. "
                    "Ele não comprova autoria ou plágio; serve como apoio de revisão e checagem contextual."
                ),
            )
            with open(pdf_path_web, "rb") as f:
                st.download_button("⬇️ Baixar relatório (Internet) em PDF", data=f.read(),
                                   file_name=os.path.basename(pdf_path_web), mime="application/pdf",
                                   use_container_width=True, key="dl_pdf_web")

# =========================================================
# TAB 3: Indícios de Uso de IA (análise heurística)
# =========================================================
with tabs[2]:
    st.subheader("Indícios de Uso de IA (análise heurística)")
    st.info(AI_HEURISTIC_NOTE)

    mode = st.radio("Como enviar o texto?", ["Colar texto", "Enviar arquivo"], horizontal=True, key="radio_ai_envio")
    query_name = "Texto colado"
    query_text = ""

    if mode == "Colar texto":
        query_text = st.text_area("Cole o texto para análise heurística:", height=240, key="ai_text")
    else:
        up = st.file_uploader("Envie um arquivo (.docx, .pdf, .txt)", type=["docx", "pdf", "txt"], key="ai_uploader")
        if up is not None:
            query_name = up.name
            query_text = _read_any(up)

    run_ai = st.button("🤖 Rodar análise heurística", type="primary", use_container_width=True,
                       disabled=(not query_text), key="btn_ai")

    if run_ai:
        ai = analyze_ai_indicia(query_text)
        st.session_state["ai_last"] = {"query_name": query_name, "ts": int(time.time()), "ai": ai}

    aires = st.session_state.get("ai_last")
    if aires:
        ai = aires["ai"]
        band_title, band_msg = ai["band"]
        st.metric("Índice heurístico", f"{ai['score']:.0f}/100")
        st.info(f"**{band_title}** — {band_msg}")

        st.write(f"- TTR: {ai['ttr']:.2f}")
        st.write(f"- CV (variação de frases): {ai['cv_sent']:.2f}")
        st.write(f"- Conectores/1k: {ai['conn_per_1k']:.1f}")
        st.write(f"- Vagueza/1k: {ai['vague_per_1k']:.1f}")
        st.write(f"- Repetição/1k: {ai['rep_per_1k']:.1f}")

        if ai.get("flagged_sentences"):
            st.markdown("### Trechos para revisão")
            for i, s in enumerate(ai["flagged_sentences"], start=1):
                st.write(f"**{i}.** {s}")

        # PDF IA
        pdf_path_ai = os.path.join(os.getcwd(), f"Relatorio_Veritas_IA_{aires.get('ts', int(time.time()))}.pdf")
        generate_ai_pdf_report(
            filepath=pdf_path_ai,
            title="Relatório – Indícios de Uso de IA (análise heurística) – Veritas",
            query_name=aires.get("query_name", "—"),
            ai_result=ai,
            disclaimer=AI_HEURISTIC_NOTE,
        )
        with open(pdf_path_ai, "rb") as f:
            st.download_button("⬇️ Baixar relatório (IA – heurístico) em PDF", data=f.read(),
                               file_name=os.path.basename(pdf_path_ai), mime="application/pdf",
                               use_container_width=True, key="dl_pdf_ai")

# =========================================================
# TAB 4: Biblioteca (upload)
# =========================================================
with tabs[3]:
    st.subheader("Biblioteca Veritas")
    up_lib = st.file_uploader(
        "Adicionar documentos (.docx, .pdf, .txt)",
        type=["docx", "pdf", "txt"],
        accept_multiple_files=True,
        key="upl_lib",
    )

    if up_lib:
        for f in up_lib:
            st.session_state["library"][f.name] = _read_any(f)
            st.session_state["library_meta"].setdefault(f.name, {"tags": "", "category": "Referência", "exclude": False})
        st.success("Documentos adicionados.")

    st.divider()
    if st.session_state["library"]:
        for name in list(st.session_state["library"].keys()):
            meta = st.session_state["library_meta"].setdefault(name, {"tags": "", "category": "Referência", "exclude": False})
            c1, c2 = st.columns([0.8, 0.2])
            with c1:
                st.write(f"📄 {name}")
                meta["exclude"] = st.checkbox("Excluir", value=bool(meta.get("exclude", False)), key=f"exc_{name}")
            with c2:
                if st.button("Remover", key=f"rm_{name}"):
                    del st.session_state["library"][name]
                    st.session_state["library_meta"].pop(name, None)
                    st.rerun()
    else:
        st.info("Ainda não há documentos na biblioteca.")

# =========================================================
# TAB 5: Sobre
# =========================================================
with tabs[4]:
    st.subheader("Sobre o Veritas")
    st.markdown(
        """
- **Modo Biblioteca (privado):** compara apenas com os documentos que você adicionou.
- **Modo Internet (externo):** usa SerpAPI para buscar *trechos curtos* e comparar com snippets da web.
- **Indícios de Uso de IA (heurístico):** calcula padrões linguísticos locais (não é veredito).
- **Importante:** nenhum modo “prova” plágio ou IA; serve como apoio à revisão e integridade acadêmica.
        """
    )
    st.caption(DISCL)
    st.caption(ETHICAL_NOTE)

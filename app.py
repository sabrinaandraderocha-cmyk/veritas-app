import os
import re
import time
import streamlit as st

from veritas_utils import (
    extract_text_from_txt_bytes,
    extract_text_from_docx_bytes,
    extract_text_from_pdf_bytes,
    compute_matches,
    highlight_text,
)
from veritas_report import generate_pdf_report

# =========================================================
# CONFIG
# =========================================================
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

# =========================================================
# Helpers
# =========================================================
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
        return "🟢 Similaridade esperada (baixa)", (
            "Em geral, indica boa autonomia textual. Ainda assim, revise se as citações estão completas."
        )
    if global_sim < 0.30:
        return "🟡 Atenção editorial (moderada)", (
            "Pode refletir trechos conceituais comuns, metodologia parecida ou paráfrases próximas. "
            "Vale revisar as seções sinalizadas e checar citações/paráfrases."
        )
    return "🟠 Revisão cuidadosa (elevada)", (
        "Não é acusação. Indica bastante sobreposição com sua biblioteca. "
        "Reveja trechos sinalizados, garanta citações corretas e aumente elaboração autoral."
    )


# =========================================================
# NOVO: detecção de seções (heurística)
# =========================================================
_DEFAULT_SECTION_ALIASES = {
    "RESUMO": ["RESUMO", "ABSTRACT"],
    "INTRODUÇÃO": ["INTRODUÇÃO", "INTRODUCAO", "INTRODUCTION"],
    "REFERENCIAL TEÓRICO": ["REFERENCIAL TEÓRICO", "REFERENCIAL TEORICO", "FUNDAMENTAÇÃO", "FUNDAMENTACAO", "MARCO TEÓRICO", "MARCO TEORICO"],
    "METODOLOGIA": ["METODOLOGIA", "MÉTODO", "METODO", "MATERIAIS E MÉTODOS", "MATERIAL E MÉTODOS", "MATERIAIS E METODOS", "MATERIAL E METODOS"],
    "RESULTADOS": ["RESULTADOS", "RESULTS"],
    "DISCUSSÃO": ["DISCUSSÃO", "DISCUSSAO", "DISCUSSION"],
    "CONCLUSÃO": ["CONCLUSÃO", "CONCLUSAO", "CONSIDERAÇÕES FINAIS", "CONSIDERACOES FINAIS", "FINAL CONSIDERATIONS"],
    "REFERÊNCIAS": ["REFERÊNCIAS", "REFERENCIAS", "BIBLIOGRAFIA", "REFERENCES"],
}


def _normalize_title(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"^\s*(\d+(\.\d+)*|[IVXLC]+)\s*[\.\-\)]\s*", "", t, flags=re.IGNORECASE)
    t = t.strip(" :.-\t")
    return t


def _canonical_section(title: str) -> str:
    raw = _normalize_title(title)
    up = raw.upper()
    for canon, aliases in _DEFAULT_SECTION_ALIASES.items():
        if up in aliases:
            return canon
    return raw if raw else "Sem título"


def _is_heading_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if len(s) > 90:
        return False
    if len(s.split()) > 12:
        return False
    if re.match(r"^\s*(\d+(\.\d+)*|[IVXLC]+)\s*[\.\-\)]\s*\S+", s, flags=re.IGNORECASE):
        return True

    letters = re.findall(r"[A-Za-zÀ-ÿ]", s)
    if letters:
        upper_letters = [c for c in letters if c.upper() == c]
        ratio = len(upper_letters) / max(1, len(letters))
        if ratio >= 0.75 and len(s.split()) <= 10:
            return True

    up = _normalize_title(s).upper()
    for _, aliases in _DEFAULT_SECTION_ALIASES.items():
        if up in aliases:
            return True

    return False


def split_into_sections(text: str):
    lines = (text or "").splitlines()
    sections = []
    current_title = "Texto"
    current_canon = "Texto"
    current_buf = []
    current_start = 0

    for i, line in enumerate(lines):
        if _is_heading_line(line):
            prev_text = "\n".join(current_buf).strip()
            if prev_text:
                sections.append({"title": current_title, "canonical": current_canon, "text": prev_text, "start_line": current_start})

            current_title = _normalize_title(line) or "Seção"
            current_canon = _canonical_section(current_title)
            current_buf = []
            current_start = i
        else:
            current_buf.append(line)

    tail = "\n".join(current_buf).strip()
    if tail:
        sections.append({"title": current_title, "canonical": current_canon, "text": tail, "start_line": current_start})

    if not sections:
        t = (text or "").strip()
        return [{"title": "Texto", "canonical": "Texto", "text": t, "start_line": 0}] if t else []

    # junta seções muito pequenas na anterior
    merged = []
    for sec in sections:
        if merged and _safe_words_count(sec["text"]) < 60:
            merged[-1]["text"] = (merged[-1]["text"].rstrip() + "\n\n" + sec["text"].lstrip()).strip()
        else:
            merged.append(sec)
    return merged


def _weighted_average(items):
    num = 0.0
    den = 0.0
    for sim, w in items:
        if w <= 0:
            continue
        num += float(sim) * float(w)
        den += float(w)
    return (num / den) if den > 0 else 0.0


def _doc_summary(matches):
    by_doc = {}
    for m in matches or []:
        by_doc.setdefault(m.source_doc, []).append(m)
    items = []
    for doc, ms in by_doc.items():
        avg = sum(x.score for x in ms) / max(1, len(ms))
        items.append((doc, avg, len(ms)))
    items.sort(key=lambda x: x[1], reverse=True)
    return items


def _init_state():
    if "library" not in st.session_state:
        st.session_state["library"] = {}
    if "library_meta" not in st.session_state:
        st.session_state["library_meta"] = {}
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    if "params" not in st.session_state:
        st.session_state["params"] = {
            "chunk_words": 60,
            "stride_words": 20,
            "threshold": 0.75,
            "top_k_per_chunk": 1,
            "exclude_marked": True,
            "min_section_words": 120,
        }


# =========================================================
# Streamlit Page
# =========================================================
st.set_page_config(page_title=APP_TITLE, layout="wide")
_init_state()

# ---------- CSS leve ----------
st.markdown(
    """
    <style>
      .veritas-hero h1 { margin-bottom: 0.2rem; }
      .veritas-hero p { margin-top: 0.2rem; opacity: 0.8; }
      .small-note { font-size: 0.92rem; opacity: 0.9; }
      .muted { opacity: 0.7; }
      .pill { display: inline-block; padding: 0.18rem 0.55rem; border-radius: 999px; border: 1px solid rgba(49,51,63,0.2); margin-right: 0.35rem; }
      .card { padding: 1rem; border-radius: 14px; border: 1px solid rgba(49,51,63,0.2); background: rgba(255,255,255,0.02); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- HERO ----------
st.markdown(
    f"""
    <div class="veritas-hero">
      <h1>🏛️ {APP_TITLE}</h1>
      <p class="muted">Análise de similaridade e integridade acadêmica</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown("**Observação ética**")
    st.caption(DISCL)
    st.caption(ETHICAL_NOTE)

# =========================================================
# Sidebar: Configurações (mais elegante)
# =========================================================
with st.sidebar:
    st.subheader("⚙️ Configurações")
    p = st.session_state["params"]

    st.markdown("**Detecção**")
    p["chunk_words"] = st.slider("Trecho (palavras)", 30, 140, int(p["chunk_words"]), 5)
    p["stride_words"] = st.slider("Passo (palavras)", 10, 80, int(p["stride_words"]), 5)
    p["threshold"] = st.slider("Limiar (0–1)", 0.50, 0.95, float(p["threshold"]), 0.01)
    p["top_k_per_chunk"] = st.slider("Top-k por trecho", 1, 3, int(p["top_k_per_chunk"]), 1)

    st.markdown("**Por seção**")
    p["min_section_words"] = st.slider("Mín. palavras por seção", 60, 400, int(p["min_section_words"]), 10)

    st.markdown("**Biblioteca**")
    p["exclude_marked"] = st.checkbox("Ignorar docs excluídos", value=bool(p["exclude_marked"]))

    st.divider()
    st.caption("Sugestão: limiar 0,75 é bom para cópia literal. Para paráfrase, teste ~0,65.")

    st.session_state["params"] = p

# =========================================================
# Tabs
# =========================================================
tabs = st.tabs(["🧪 Nova análise", "📚 Biblioteca", "🧾 Relatórios"])

# =========================================================
# TAB: Nova análise
# =========================================================
with tabs[0]:
    col1, col2 = st.columns([1.15, 0.85], gap="large")

    with col1:
        with st.container(border=True):
            st.subheader("Texto para análise")
            mode = st.radio("Como você quer enviar o texto?", ["Colar texto", "Enviar arquivo"], horizontal=True)

            query_name = "Texto colado"
            query_text = ""

            if mode == "Colar texto":
                query_text = st.text_area(
                    "Cole aqui o texto do trabalho/artigo:",
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
                st.warning("Sua biblioteca está vazia. Vá em **Biblioteca** e faça upload de documentos para comparar.")

            run = st.button(
                "🔎 Analisar agora",
                type="primary",
                use_container_width=True,
                disabled=(not query_text or not st.session_state["library"]),
            )

        with st.expander("🧭 Revisão ética (modo formativo)", expanded=True):
            st.markdown(
                "- Os trechos sinalizados têm **citação adequada**?\n"
                "- Há **paráfrase próxima** sem elaboração?\n"
                "- A **metodologia** está específica do seu estudo?\n"
                "- A seção teórica está **dialogando** ou só reproduzindo?\n"
            )

    with col2:
        with st.container(border=True):
            st.subheader("Resumo rápido")
            wc = _safe_words_count(query_text)
            st.markdown(f"<span class='pill'>📄 {wc} palavras</span>", unsafe_allow_html=True)
            st.markdown("<p class='small-note'>A comparação é feita contra documentos da sua <b>Biblioteca Veritas</b>.</p>", unsafe_allow_html=True)

            st.info("Dica: inclua trabalhos anteriores, artigos de referência, TCCs, capítulos, etc.")

    if run:
        params = st.session_state["params"]
        chunk_words = int(params["chunk_words"])
        stride_words = int(params["stride_words"])
        threshold = float(params["threshold"])
        top_k_per_chunk = int(params["top_k_per_chunk"])
        exclude_marked = bool(params["exclude_marked"])
        min_section_words = int(params["min_section_words"])

        corpus = {}
        for name, text in st.session_state["library"].items():
            meta = st.session_state["library_meta"].get(name, {})
            if exclude_marked and meta.get("exclude", False):
                continue
            corpus[name] = text

        if not corpus:
            st.error("Todos os documentos da biblioteca estão marcados como excluídos. Ajuste na aba **Biblioteca**.")
        else:
            with st.spinner("Analisando (global + por seção)..."):
                global_sim, matches = compute_matches(
                    query_text=query_text,
                    corpus_docs=corpus,
                    chunk_words=chunk_words,
                    stride_words=stride_words,
                    top_k_per_chunk=top_k_per_chunk,
                    threshold=threshold,
                )

                sections = split_into_sections(query_text)
                section_rows = []
                section_matches_map = {}
                weighted_items = []

                for idx, sec in enumerate(sections, start=1):
                    sec_text = (sec["text"] or "").strip()
                    sec_words = _safe_words_count(sec_text)
                    if sec_words < min_section_words:
                        continue

                    sec_sim, sec_matches = compute_matches(
                        query_text=sec_text,
                        corpus_docs=corpus,
                        chunk_words=chunk_words,
                        stride_words=stride_words,
                        top_k_per_chunk=top_k_per_chunk,
                        threshold=threshold,
                    )

                    key = f"{idx:02d} — {sec.get('canonical') or sec.get('title')}"
                    section_matches_map[key] = sec_matches
                    section_rows.append(
                        {
                            "Seção": key,
                            "Palavras": sec_words,
                            "Similaridade (%)": round(sec_sim * 100, 1),
                            "Trechos sinalizados": len(sec_matches or []),
                        }
                    )
                    weighted_items.append((sec_sim, sec_words))

                sections_weighted_sim = _weighted_average(weighted_items)

            st.session_state["last_result"] = {
                "query_name": query_name,
                "query_text": query_text,
                "global_sim": float(global_sim),
                "matches": matches,
                "sections_table": section_rows,
                "sections_weighted_sim": float(sections_weighted_sim),
                "section_matches_map": section_matches_map,
                "params": {
                    "chunk_words": chunk_words,
                    "stride_words": stride_words,
                    "threshold": threshold,
                    "top_k_per_chunk": top_k_per_chunk,
                    "exclude_marked": exclude_marked,
                    "min_section_words": min_section_words,
                },
                "corpus_size": len(corpus),
                "ts": int(time.time()),
            }

    res = st.session_state.get("last_result")
    if res:
        st.divider()
        st.subheader("Resultado")

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

        # ---- por seção
        st.markdown("### Similaridade por seção")
        rows = res.get("sections_table") or []
        if not rows:
            st.warning(
                "Não detectei seções suficientes (ou ficaram abaixo do mínimo). "
                "Dica: use títulos como 'INTRODUÇÃO', 'METODOLOGIA', 'RESULTADOS' em linhas separadas."
            )
        else:
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.caption(f"Média por seção (ponderada por palavras): **{res.get('sections_weighted_sim', 0.0)*100:.1f}%**")

            try:
                import pandas as pd
                df = pd.DataFrame(rows).set_index("Seção")[["Similaridade (%)"]]
                st.bar_chart(df, use_container_width=True)
            except Exception:
                pass

            with st.container(border=True):
                st.markdown("**Ver trechos sinalizados por seção**")
                section_keys = [r["Seção"] for r in rows]
                pick = st.selectbox("Escolha uma seção", section_keys)
                sec_matches = res.get("section_matches_map", {}).get(pick, [])
                if not sec_matches:
                    st.success("Nenhum trecho sinalizado nessa seção acima do limiar.")
                else:
                    for i, m in enumerate(sec_matches[:20], start=1):
                        st.markdown(f"**{i}.** `{m.source_doc}` — **{m.score*100:.1f}%**")
                        st.caption("Trecho da seção")
                        st.write(m.query_chunk)
                        st.caption("Trecho fonte")
                        st.write(m.source_chunk)
                        st.divider()

        # ---- destaque e PDF
        left, right = st.columns([1, 1], gap="large")

        with left:
            with st.container(border=True):
                st.markdown("### Trechos sinalizados (global)")
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
                    + f"Leitura interpretativa (faixa): {band_title} — {band_msg}"
                )

                rows = res.get("sections_table") or []
                if rows:
                    disclaimer_plus += "\n\nResumo por seção (similaridade %):\n"
                    for rr in rows[:12]:
                        disclaimer_plus += f"- {rr['Seção']}: {rr['Similaridade (%)']}% (trechos: {rr['Trechos sinalizados']})\n"

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
# TAB: Biblioteca
# =========================================================
with tabs[1]:
    st.subheader("Biblioteca Veritas")
    st.write("Os documentos aqui são as **fontes de comparação**. (MVP: guardado na sessão)")

    with st.container(border=True):
        up_lib = st.file_uploader(
            "Adicionar documentos à biblioteca (.docx, .pdf, .txt)",
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
                st.success(f"{added} documento(s) adicionados à biblioteca.")

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
                    )

                with c4:
                    if st.button("Remover", key=f"rm_{name}", use_container_width=True):
                        st.session_state["library"].pop(name, None)
                        st.session_state["library_meta"].pop(name, None)
                        st.rerun()
    else:
        st.info("Ainda não há documentos na biblioteca.")

# =========================================================
# TAB: Relatórios (placeholder útil)
# =========================================================
with tabs[2]:
    st.subheader("Relatórios")
    st.write("Aqui você pode baixar o relatório após rodar uma análise.")

    res = st.session_state.get("last_result")
    if not res:
        st.info("Rode uma análise na aba **Nova análise** para gerar um relatório.")
    else:
        st.success("Já há uma análise recente disponível. Vá em **Nova análise** e baixe o PDF na coluna da direita.")

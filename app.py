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

APP_TITLE = "Veritas"

DISCL = (
    "O Veritas realiza análise automatizada de similaridade textual. "
    "O resultado não configura, por si só, juízo definitivo sobre plágio acadêmico, "
    "o qual depende de avaliação contextual e humana (citações, paráfrases, domínio público, etc.)."
)

ETHICAL_NOTE = (
    "⚠️ Similaridade não é, por si só, falta ética. "
    "A escrita acadêmica é dialógica: trechos conceituais, metodologia, citações e fórmulas recorrentes "
    "podem elevar a correspondência. Use o relatório como apoio de revisão, não como veredito."
)

# ----------------------------
# Helpers
# ----------------------------
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
    """
    Faixas interpretativas (NÃO punitivas).
    Ajuste livremente depois.
    """
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
        "Não é acusação. Indica que há bastante sobreposição com sua biblioteca. "
        "Reveja trechos sinalizados, garanta citações corretas e aumente elaboração autoral."
    )


def _chunk_type_heuristic(chunk: str) -> str:
    """
    Heurística simples para rotular o tipo de trecho (sem prometer perfeição).
    """
    c = (chunk or "").strip()

    # citação direta provável
    if "“" in c or "”" in c or '"' in c or "''" in c or "‘‘" in c or "’" in c:
        return "📌 Citação direta provável"

    # autor-data provável: (SOBRENOME, 2020) / (Autor, 2020)
    if re.search(r"\([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s\-]+,\s*\d{4}[a-z]?\)", c):
        return "📚 Citação autor-data provável"

    lower = c.lower()

    # metodologia
    metod_terms = [
        "metodologia", "método", "amostra", "participantes", "procedimento",
        "instrumento", "coleta", "análise de dados", "análise estatística",
        "métodos", "material e métodos", "desenho do estudo"
    ]
    if any(t in lower for t in metod_terms):
        return "🧪 Metodologia / Procedimentos (similaridade comum)"

    # teórico/conceitual
    theory_terms = [
        "conceito", "define-se", "definição", "segundo", "de acordo com",
        "compreende-se", "refere-se", "pressupõe", "noção", "teoria"
    ]
    if any(t in lower for t in theory_terms):
        return "📖 Conceitual / Teórico (similaridade comum)"

    return "✍️ Argumentativo / Autoral (revise com atenção)"


def _likely_bibliographic(chunk: str) -> bool:
    lower = (chunk or "").lower()
    bib_markers = ["referências", "bibliografia", "apud", "et al.", "doi:", "http://", "https://"]
    return any(m in lower for m in bib_markers)


def _doc_summary(matches):
    """
    Agrupa correspondências por documento fonte, para facilitar leitura.
    """
    by_doc = {}
    for m in matches or []:
        by_doc.setdefault(m.source_doc, []).append(m)
    # ordena por maior score médio
    items = []
    for doc, ms in by_doc.items():
        avg = sum(x.score for x in ms) / max(1, len(ms))
        items.append((doc, avg, len(ms)))
    items.sort(key=lambda x: x[1], reverse=True)
    return items


def _init_state():
    if "library" not in st.session_state:
        st.session_state["library"] = {}  # name -> text

    # metadados simples (tags/exclusões) por documento
    if "library_meta" not in st.session_state:
        st.session_state["library_meta"] = {}  # name -> dict(tags, category, exclude)

    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    if "params" not in st.session_state:
        st.session_state["params"] = {
            "chunk_words": 60,
            "stride_words": 20,
            "threshold": 0.75,
            "top_k_per_chunk": 1,
            "exclude_marked": True,
        }

    if "ui" not in st.session_state:
        st.session_state["ui"] = {
            "max_matches_show": 20,
            "show_review_mode": True,
            "show_doc_summary": True,
            "show_chunk_labels": True,
        }


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
_init_state()

st.title(f"🏛️ {APP_TITLE}")
st.caption("Análise de similaridade e integridade acadêmica")

tabs = st.tabs(["Nova análise", "Biblioteca", "Configurações"])

# =========================================================
# TAB 1: Nova análise
# =========================================================
with tabs[0]:
    col1, col2 = st.columns([1.15, 0.85], gap="large")

    with col1:
        st.subheader("Texto para análise")
        mode = st.radio(
            "Como você quer enviar o texto?",
            ["Colar texto", "Enviar arquivo"],
            horizontal=True
        )

        query_name = "Texto colado"
        query_text = ""

        if mode == "Colar texto":
            query_text = st.text_area(
                "Cole aqui o texto do trabalho/artigo:",
                height=260,
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

        st.subheader("Rodar análise")

        if not st.session_state["library"]:
            st.warning("Sua biblioteca está vazia. Vá na aba **Biblioteca** e faça upload de documentos para comparar.")

        run = st.button(
            "🔎 Analisar",
            type="primary",
            use_container_width=True,
            disabled=(not query_text or not st.session_state["library"]),
        )

        with st.expander("🧭 Revisão ética (modo formativo)", expanded=st.session_state["ui"]["show_review_mode"]):
            st.write("Use estas perguntas como guia de revisão — o Veritas não substitui avaliação humana.")
            st.markdown(
                "- Os trechos sinalizados têm **citação adequada** (direta ou indireta)?\n"
                "- Onde há paráfrase, você **agregou elaboração autoral** (argumento, contraste, exemplo, crítica)?\n"
                "- Trechos metodológicos estão descritos com **especificidade do seu estudo** (e não apenas modelo genérico)?\n"
                "- Há partes que parecem **bibliografia/recorte técnico** e poderiam ser reorganizadas?\n"
                "- A seção de **introdução/fundamentação** está dialogando com fontes ou apenas reproduzindo?\n"
            )
            st.caption(ETHICAL_NOTE)

    with col2:
        st.subheader("Resumo")
        st.write("A comparação é feita contra documentos da sua **Biblioteca Veritas**.")

        st.info(
            "Dica: inclua trabalhos anteriores, artigos de referência, TCCs, capítulos, etc. "
            "Quanto melhor a biblioteca, melhor a detecção."
        )

        st.markdown("**Observação ética:**")
        st.caption(DISCL)
        st.caption(ETHICAL_NOTE)

        st.divider()
        st.markdown("**Saúde do texto (rápido):**")
        wc = _safe_words_count(query_text)
        st.write(f"• Tamanho do texto: **{wc} palavras**")
        if wc and wc < 150:
            st.warning("Texto bem curto pode gerar resultados instáveis. Se possível, analise seções maiores.")

    if run:
        params = st.session_state["params"]
        chunk_words = int(params["chunk_words"])
        stride_words = int(params["stride_words"])
        threshold = float(params["threshold"])
        top_k_per_chunk = int(params.get("top_k_per_chunk", 1))
        exclude_marked = bool(params.get("exclude_marked", True))

        # Filtra biblioteca se usuário marcou documentos como excluídos
        corpus = {}
        for name, text in st.session_state["library"].items():
            meta = st.session_state["library_meta"].get(name, {})
            if exclude_marked and meta.get("exclude", False):
                continue
            corpus[name] = text

        if not corpus:
            st.error("Todos os documentos da biblioteca estão marcados como excluídos. Ajuste na aba **Biblioteca**.")
        else:
            with st.spinner("Analisando similaridade..."):
                global_sim, matches = compute_matches(
                    query_text=query_text,
                    corpus_docs=corpus,
                    chunk_words=chunk_words,
                    stride_words=stride_words,
                    top_k_per_chunk=top_k_per_chunk,
                    threshold=threshold,
                )

            # Enriquecimento leve (rótulos)
            enriched = []
            for m in (matches or []):
                label = _chunk_type_heuristic(m.query_chunk)
                bib = _likely_bibliographic(m.query_chunk) or _likely_bibliographic(m.source_chunk)
                enriched.append({
                    "source_doc": m.source_doc,
                    "score": m.score,
                    "query_chunk": m.query_chunk,
                    "source_chunk": m.source_chunk,
                    "label": label,
                    "bibliographic_hint": bib,
                })

            st.session_state["last_result"] = {
                "query_name": query_name,
                "global_sim": global_sim,
                "matches": matches,          # original (para highlight_text/report)
                "enriched": enriched,        # para UI
                "params": {
                    "chunk_words": chunk_words,
                    "stride_words": stride_words,
                    "threshold": threshold,
                    "top_k_per_chunk": top_k_per_chunk,
                    "exclude_marked": exclude_marked,
                },
                "query_text": query_text,
                "corpus_size": len(corpus),
            }

    res = st.session_state.get("last_result")
    if res:
        st.divider()
        st.subheader("Resultado")

        global_sim = float(res["global_sim"] or 0.0)
        band_title, band_msg = _band(global_sim)

        c1, c2, c3 = st.columns([0.34, 0.33, 0.33])
        with c1:
            st.metric("Índice global (estimado)", f"{global_sim*100:.1f}%")
        with c2:
            st.metric("Docs comparados", f"{res.get('corpus_size', 0)}")
        with c3:
            st.metric("Trechos sinalizados", f"{len(res.get('enriched', []) or [])}")

        st.info(f"**{band_title}** — {band_msg}")

        if st.session_state["ui"]["show_doc_summary"]:
            with st.expander("📌 Fontes mais presentes no resultado", expanded=True):
                items = _doc_summary(res["matches"])
                if not items:
                    st.write("Nenhuma fonte acima do limiar.")
                else:
                    for doc, avg, n in items[:10]:
                        meta = st.session_state["library_meta"].get(doc, {})
                        tags = meta.get("tags", "")
                        category = meta.get("category", "—")
                        st.write(f"• **{doc}** — média **{avg*100:.1f}%** | trechos: **{n}** | categoria: **{category}** | tags: {tags or '—'}")

        mcol1, mcol2 = st.columns([1, 1], gap="large")

        # ----------------------------
        # Trechos sinalizados (explicáveis)
        # ----------------------------
        with mcol1:
            st.markdown("### Trechos sinalizados (interpretáveis)")

            enriched = res.get("enriched", []) or []
            if not enriched:
                st.success("Nenhuma correspondência acima do limiar foi encontrada.")
            else:
                max_show = int(st.session_state["ui"]["max_matches_show"])
                for i, m in enumerate(enriched[:max_show], start=1):
                    header = f"**{i}.** Fonte: `{m['source_doc']}` — **{m['score']*100:.1f}%**"
                    st.markdown(header)

                    if st.session_state["ui"]["show_chunk_labels"]:
                        st.caption(f"Tipo de trecho (heurística): {m['label']}")
                        if m["bibliographic_hint"]:
                            st.caption("Possível trecho bibliográfico/técnico (heurística). Revise com cuidado, mas sem alarme.")

                    st.caption("Trecho analisado")
                    st.write(m["query_chunk"])
                    st.caption("Trecho fonte")
                    st.write(m["source_chunk"])

                    with st.expander("Perguntas rápidas de revisão", expanded=False):
                        st.markdown(
                            "- Este trecho precisa de **citação direta/indireta**?\n"
                            "- A **paráfrase** está distante o suficiente e com elaboração?\n"
                            "- Dá para inserir **comentário autoral** (contraste, justificativa, exemplo)?\n"
                            "- Esse trecho é **metodologia/definição padrão** (onde a similaridade é comum)?\n"
                        )
                    st.divider()

        # ----------------------------
        # Destaques + Relatório
        # ----------------------------
        with mcol2:
            st.markdown("### Texto com destaques (melhor esforço)")
            highlighted = highlight_text(res["query_text"], res["matches"])
            st.text_area("Destaques aparecem entre ⟦ ⟧", value=highlighted, height=420)

            st.markdown("### Relatório (PDF)")
            pdf_path = os.path.join(os.getcwd(), f"Relatorio_Veritas_{int(time.time())}.pdf")

            # inclui uma “camada” interpretativa no disclaimer (sem quebrar o gerador)
            band_title, band_msg = _band(res["global_sim"])
            disclaimer_plus = (
                DISCL
                + "\n\n"
                + ETHICAL_NOTE
                + "\n\n"
                + f"Leitura interpretativa (faixa): {band_title} — {band_msg}"
            )

            generate_pdf_report(
                filepath=pdf_path,
                title="Relatório de Análise de Similaridade – Veritas",
                query_name=res["query_name"],
                global_similarity=res["global_sim"],
                matches=res["matches"],
                params=res["params"],
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
# TAB 2: Biblioteca
# =========================================================
with tabs[1]:
    st.subheader("Biblioteca Veritas")
    st.write("Os documentos aqui são as **fontes de comparação**. Eles ficam salvos na sessão (MVP local).")

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

                # cria meta padrão se não existir
                st.session_state["library_meta"].setdefault(
                    f.name,
                    {"tags": "", "category": "Referência", "exclude": False}
                )
                added += 1
            except Exception as e:
                st.error(f"Falha ao ler {f.name}: {e}")

        if added:
            st.success(f"{added} documento(s) adicionados à biblioteca.")

    st.divider()
    if st.session_state["library"]:
        st.markdown("### Documentos na biblioteca (com tags/categorias)")
        st.caption("Você pode marcar documentos para **excluir da comparação** (ex.: rascunhos, versões repetidas).")

        for name in list(st.session_state["library"].keys()):
            meta = st.session_state["library_meta"].setdefault(
                name, {"tags": "", "category": "Referência", "exclude": False}
            )

            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([0.44, 0.24, 0.20, 0.12], vertical_alignment="center")

                with c1:
                    st.write(f"📄 **{name}**")
                    st.caption(f"{_safe_words_count(st.session_state['library'][name])} palavras")

                with c2:
                    meta["category"] = st.selectbox(
                        "Categoria",
                        ["Referência", "Meu texto (autoria)", "Domínio público", "Modelo/metodologia", "Outros"],
                        index=["Referência", "Meu texto (autoria)", "Domínio público", "Modelo/metodologia", "Outros"].index(
                            meta.get("category", "Referência") if meta.get("category") in
                            ["Referência", "Meu texto (autoria)", "Domínio público", "Modelo/metodologia", "Outros"]
                            else "Referência"
                        ),
                        key=f"cat_{name}",
                    )

                with c3:
                    meta["tags"] = st.text_input(
                        "Tags (opcional)",
                        value=meta.get("tags", ""),
                        placeholder="ex.: rogers; fenomenologia; penal; metodologia",
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

        st.session_state["library_meta"] = st.session_state["library_meta"]
    else:
        st.info("Ainda não há documentos na biblioteca.")


# =========================================================
# TAB 3: Configurações
# =========================================================
with tabs[2]:
    st.subheader("Configurações da análise (MVP)")
    st.write("Ajuste o tamanho dos trechos, limiar e opções de visualização. Em geral, os padrões funcionam bem.")

    params = st.session_state["params"]
    ui = st.session_state["ui"]

    st.markdown("### Parâmetros de detecção")
    params["chunk_words"] = st.slider("Tamanho do trecho (palavras)", 30, 140, int(params["chunk_words"]), 5)
    params["stride_words"] = st.slider("Passo entre trechos (palavras)", 10, 80, int(params["stride_words"]), 5)
    params["threshold"] = st.slider("Limiar de sinalização (0–1)", 0.50, 0.95, float(params["threshold"]), 0.01)
    params["top_k_per_chunk"] = st.slider("Melhor fonte por trecho (top-k)", 1, 3, int(params.get("top_k_per_chunk", 1)), 1)
    params["exclude_marked"] = st.checkbox("Ignorar docs marcados como excluídos na biblioteca", value=bool(params.get("exclude_marked", True)))

    st.caption("Sugestão: limiar 0,75 é bom para cópia literal. Para textos com muita paráfrase, reduza para ~0,65.")

    st.divider()
    st.markdown("### Visualização e modo formativo")
    ui["max_matches_show"] = st.slider("Máximo de trechos exibidos", 5, 60, int(ui["max_matches_show"]), 5)
    ui["show_doc_summary"] = st.checkbox("Mostrar resumo por documento (fontes mais presentes)", value=bool(ui["show_doc_summary"]))
    ui["show_chunk_labels"] = st.checkbox("Mostrar rótulos de tipo de trecho (heurística)", value=bool(ui["show_chunk_labels"]))
    ui["show_review_mode"] = st.checkbox("Manter expander de Revisão ética aberto por padrão", value=bool(ui["show_review_mode"]))

    st.session_state["params"] = params
    st.session_state["ui"] = ui

    st.divider()
    st.markdown("### Nota importante")
    st.caption(
        "Os rótulos (citação/metodologia/conceitual) são heurísticos e servem apenas para orientar leitura. "
        "O Veritas não faz julgamento definitivo sobre plágio."
    )

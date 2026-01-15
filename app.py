import os
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

def _init_state():
    if "library" not in st.session_state:
        st.session_state["library"] = {}  # name -> text
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "params" not in st.session_state:
        st.session_state["params"] = {"chunk_words": 60, "stride_words": 20, "threshold": 0.75}

st.set_page_config(page_title=APP_TITLE, layout="wide")
_init_state()

st.title(f"🏛️ {APP_TITLE}")
st.caption("Análise de similaridade e integridade acadêmica")

tabs = st.tabs(["Nova análise", "Biblioteca", "Configurações"])

with tabs[0]:
    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        st.subheader("Texto para análise")
        mode = st.radio("Como você quer enviar o texto?", ["Colar texto", "Enviar arquivo"], horizontal=True)

        query_name = "Texto colado"
        query_text = ""
        if mode == "Colar texto":
            query_text = st.text_area(
                "Cole aqui o texto do trabalho/artigo:",
                height=260,
                placeholder="Cole seu texto aqui..."
            )
        else:
            up = st.file_uploader("Envie um arquivo (.docx, .pdf, .txt)", type=["docx","pdf","txt"])
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

    with col2:
        st.subheader("Resumo")
        st.write("A comparação é feita contra documentos da sua **Biblioteca Veritas**.")
        st.info("Dica: inclua trabalhos anteriores, artigos de referência, TCCs, capítulos, etc. Quanto melhor a biblioteca, melhor a detecção.")
        st.markdown("**Observação ética:**")
        st.caption(DISCL)

    if run:
        params = st.session_state["params"]
        chunk_words = int(params["chunk_words"])
        stride_words = int(params["stride_words"])
        threshold = float(params["threshold"])

        with st.spinner("Analisando similaridade..."):
            global_sim, matches = compute_matches(
                query_text=query_text,
                corpus_docs=st.session_state["library"],
                chunk_words=chunk_words,
                stride_words=stride_words,
                top_k_per_chunk=1,
                threshold=threshold,
            )

        st.session_state["last_result"] = {
            "query_name": query_name,
            "global_sim": global_sim,
            "matches": matches,
            "params": {"chunk_words": chunk_words, "stride_words": stride_words, "threshold": threshold},
            "query_text": query_text,
        }

    res = st.session_state.get("last_result")
    if res:
        st.divider()
        st.subheader("Resultado")

        global_sim = res["global_sim"]
        st.metric("Índice global de similaridade (estimado)", f"{global_sim*100:.1f}%")

        mcol1, mcol2 = st.columns([1,1], gap="large")
        with mcol1:
            st.markdown("### Trechos sinalizados")
            matches = res["matches"]
            if not matches:
                st.success("Nenhuma correspondência acima do limiar foi encontrada.")
            else:
                for i, m in enumerate(matches[:20], start=1):
                    st.markdown(f"**{i}.** Fonte: `{m.source_doc}` — **{m.score*100:.1f}%**")
                    st.caption("Trecho analisado")
                    st.write(m.query_chunk)
                    st.caption("Trecho fonte")
                    st.write(m.source_chunk)
                    st.divider()

        with mcol2:
            st.markdown("### Texto com destaques (melhor esforço)")
            highlighted = highlight_text(res["query_text"], res["matches"])
            st.text_area("Destaques aparecem entre ⟦ ⟧", value=highlighted, height=520)

            st.markdown("### Relatório (PDF)")
            pdf_path = os.path.join(os.getcwd(), f"Relatorio_Veritas_{int(time.time())}.pdf")
            generate_pdf_report(
                filepath=pdf_path,
                title="Relatório de Análise de Similaridade – Veritas",
                query_name=res["query_name"],
                global_similarity=res["global_sim"],
                matches=res["matches"],
                params=res["params"],
                disclaimer=DISCL,
            )
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "⬇️ Baixar relatório em PDF",
                    data=f.read(),
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf",
                    use_container_width=True,
                )

with tabs[1]:
    st.subheader("Biblioteca Veritas")
    st.write("Os documentos aqui são as **fontes de comparação**. Eles ficam salvos na sessão (MVP local).")

    up_lib = st.file_uploader(
        "Adicionar documentos à biblioteca (.docx, .pdf, .txt)",
        type=["docx","pdf","txt"],
        accept_multiple_files=True,
    )
    if up_lib:
        added = 0
        for f in up_lib:
            try:
                st.session_state["library"][f.name] = _read_any(f)
                added += 1
            except Exception as e:
                st.error(f"Falha ao ler {f.name}: {e}")
        if added:
            st.success(f"{added} documento(s) adicionados à biblioteca.")

    st.divider()
    if st.session_state["library"]:
        st.markdown("### Documentos na biblioteca")
        for name in list(st.session_state["library"].keys()):
            c1, c2 = st.columns([0.8, 0.2])
            with c1:
                st.write(f"📄 {name}")
            with c2:
                if st.button("Remover", key=f"rm_{name}"):
                    del st.session_state["library"][name]
                    st.rerun()
    else:
        st.info("Ainda não há documentos na biblioteca.")

with tabs[2]:
    st.subheader("Configurações da análise (MVP)")
    st.write("Ajuste o tamanho dos trechos e o limiar. Em geral, os padrões funcionam bem.")
    params = st.session_state["params"]

    params["chunk_words"] = st.slider("Tamanho do trecho (palavras)", 30, 120, int(params["chunk_words"]), 5)
    params["stride_words"] = st.slider("Passo entre trechos (palavras)", 10, 60, int(params["stride_words"]), 5)
    params["threshold"] = st.slider("Limiar de sinalização (0–1)", 0.50, 0.95, float(params["threshold"]), 0.01)

    st.session_state["params"] = params
    st.caption("Sugestão: limiar 0,75 é bom para cópia literal. Para textos com muita paráfrase, reduza para ~0,65.")

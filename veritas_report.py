from __future__ import annotations

from typing import List, Dict, Any
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.lib import colors

from veritas_utils import Match


def _wrap_text(text: str, font_name: str, font_size: int, max_width: float):
    words = (text or "").split()
    lines = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        if stringWidth(test, font_name, font_size) <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _new_page(c: canvas.Canvas, width: float, height: float, margin: float) -> float:
    c.showPage()
    return height - margin


def _draw_header(c: canvas.Canvas, title: str, margin: float, y: float) -> float:
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, y, title)
    return y - 0.8 * cm


def _draw_kv(c: canvas.Canvas, margin: float, y: float, label: str, value: str) -> float:
    c.setFont("Helvetica", 11)
    c.setFillColor(colors.grey)
    c.drawString(margin, y, f"{label} {value}")
    c.setFillColor(colors.black)
    return y - 0.5 * cm


def _draw_section_title(c: canvas.Canvas, margin: float, y: float, txt: str) -> float:
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.black)
    c.drawString(margin, y, txt)
    return y - 0.55 * cm


# =========================================================
# PDF 1) Biblioteca
# =========================================================
def generate_pdf_report(
    filepath: str,
    title: str,
    query_name: str,
    global_similarity: float,
    matches: List[Match],
    params: Dict,
    disclaimer: str,
):
    c = canvas.Canvas(filepath, pagesize=A4)
    width, height = A4
    margin = 2.0 * cm
    y = height - margin

    y = _draw_header(c, title, margin, y)
    y = _draw_kv(c, margin, y, "Documento analisado:", query_name)
    y = _draw_kv(c, margin, y, "Índice global de similaridade (estimado):", f"{global_similarity*100:.1f}%")
    y -= 0.3 * cm

    y = _draw_section_title(c, margin, y, "Parâmetros")
    c.setFont("Helvetica", 10)

    for k, v in params.items():
        c.drawString(margin, y, f"- {k}: {v}")
        y -= 0.42 * cm
        if y < margin + 5 * cm:
            y = _new_page(c, width, height, margin)
            c.setFont("Helvetica", 10)

    y -= 0.2 * cm
    y = _draw_section_title(c, margin, y, "Correspondências relevantes")
    c.setFont("Helvetica", 10)
    max_w = width - 2 * margin

    if not matches:
        c.drawString(margin, y, "Nenhuma correspondência acima do limiar foi encontrada.")
        y -= 0.5 * cm
    else:
        for idx, m in enumerate(matches[:50], start=1):
            c.setFont("Helvetica-Bold", 10)
            c.drawString(margin, y, f"{idx}. Fonte: {m.source_doc} | Similaridade: {m.score*100:.1f}%")
            y -= 0.45 * cm

            c.setFont("Helvetica", 9)
            q_lines = _wrap_text("Trecho analisado: " + (m.query_chunk or ""), "Helvetica", 9, max_w)
            s_lines = _wrap_text("Trecho fonte: " + (m.source_chunk or ""), "Helvetica", 9, max_w)

            for line in q_lines:
                c.drawString(margin, y, line)
                y -= 0.38 * cm
                if y < margin + 4 * cm:
                    y = _new_page(c, width, height, margin)
                    c.setFont("Helvetica", 9)

            for line in s_lines:
                c.drawString(margin, y, line)
                y -= 0.38 * cm
                if y < margin + 4 * cm:
                    y = _new_page(c, width, height, margin)
                    c.setFont("Helvetica", 9)

            y -= 0.25 * cm
            if y < margin + 4 * cm:
                y = _new_page(c, width, height, margin)

    y -= 0.2 * cm
    y = _draw_section_title(c, margin, y, "Observação ética")
    c.setFont("Helvetica", 9)

    for line in _wrap_text(disclaimer, "Helvetica", 9, max_w):
        c.drawString(margin, y, line)
        y -= 0.38 * cm
        if y < margin + 2 * cm:
            y = _new_page(c, width, height, margin)
            c.setFont("Helvetica", 9)

    c.save()


# =========================================================
# PDF 2) Internet (externo)
# =========================================================
def generate_web_pdf_report(
    filepath: str,
    title: str,
    query_name: str,
    profile: str,
    global_web_score: float,
    hits: List[Any],
    disclaimer: str,
):
    """
    hits: lista com atributos: title, link, snippet, score, chunk
    """
    c = canvas.Canvas(filepath, pagesize=A4)
    width, height = A4
    margin = 2.0 * cm
    y = height - margin
    max_w = width - 2 * margin

    y = _draw_header(c, title, margin, y)
    y = _draw_kv(c, margin, y, "Documento analisado:", query_name)
    y = _draw_kv(c, margin, y, "Perfil:", profile)
    y = _draw_kv(c, margin, y, "Índice web (heurístico):", f"{global_web_score*100:.1f}%")
    y -= 0.3 * cm

    y = _draw_section_title(c, margin, y, "Ressalva")
    c.setFont("Helvetica", 9)
    for line in _wrap_text(disclaimer, "Helvetica", 9, max_w):
        c.drawString(margin, y, line)
        y -= 0.38 * cm
        if y < margin + 3 * cm:
            y = _new_page(c, width, height, margin)
            c.setFont("Helvetica", 9)

    y -= 0.2 * cm
    y = _draw_section_title(c, margin, y, "Principais correspondências (Internet)")
    c.setFont("Helvetica", 9)

    if not hits:
        c.drawString(margin, y, "Nenhuma correspondência foi registrada.")
        y -= 0.5 * cm
    else:
        for idx, h in enumerate(hits[:25], start=1):
            title_txt = getattr(h, "title", "") or ""
            link_txt = getattr(h, "link", "") or ""
            snip_txt = getattr(h, "snippet", "") or ""
            chunk_txt = getattr(h, "chunk", "") or ""
            score = float(getattr(h, "score", 0.0) or 0.0)

            c.setFont("Helvetica-Bold", 10)
            c.drawString(margin, y, f"{idx}. Score: {score*100:.1f}%")
            y -= 0.45 * cm

            c.setFont("Helvetica", 9)
            for line in _wrap_text("Título: " + title_txt, "Helvetica", 9, max_w):
                c.drawString(margin, y, line)
                y -= 0.38 * cm
                if y < margin + 4 * cm:
                    y = _new_page(c, width, height, margin)
                    c.setFont("Helvetica", 9)

            for line in _wrap_text("Link: " + link_txt, "Helvetica", 9, max_w):
                c.drawString(margin, y, line)
                y -= 0.38 * cm
                if y < margin + 4 * cm:
                    y = _new_page(c, width, height, margin)
                    c.setFont("Helvetica", 9)

            if snip_txt:
                for line in _wrap_text("Snippet: " + snip_txt, "Helvetica", 9, max_w):
                    c.drawString(margin, y, line)
                    y -= 0.38 * cm
                    if y < margin + 4 * cm:
                        y = _new_page(c, width, height, margin)
                        c.setFont("Helvetica", 9)

            if chunk_txt:
                for line in _wrap_text("Trecho enviado (chunk): " + chunk_txt, "Helvetica", 9, max_w):
                    c.drawString(margin, y, line)
                    y -= 0.38 * cm
                    if y < margin + 4 * cm:
                        y = _new_page(c, width, height, margin)
                        c.setFont("Helvetica", 9)

            y -= 0.25 * cm
            if y < margin + 4 * cm:
                y = _new_page(c, width, height, margin)

    c.save()


# =========================================================
# PDF 3) Indícios de Uso de IA (heurístico)
# =========================================================
def generate_ai_pdf_report(
    filepath: str,
    title: str,
    query_name: str,
    ai_result: Dict[str, Any],
    disclaimer: str,
):
    """
    ai_result: dict retornado por analyze_ai_indicia()
    """
    c = canvas.Canvas(filepath, pagesize=A4)
    width, height = A4
    margin = 2.0 * cm
    y = height - margin
    max_w = width - 2 * margin

    score = float(ai_result.get("score", 0.0) or 0.0)
    band = ai_result.get("band", ("—", "")) or ("—", "")
    band_title = band[0]
    band_msg = band[1] if len(band) > 1 else ""

    y = _draw_header(c, title, margin, y)
    y = _draw_kv(c, margin, y, "Documento analisado:", query_name)
    y = _draw_kv(c, margin, y, "Índice heurístico:", f"{score:.0f}/100")
    y = _draw_kv(c, margin, y, "Faixa:", str(band_title))
    y -= 0.2 * cm

    if band_msg:
        c.setFont("Helvetica", 9)
        for line in _wrap_text("Interpretação: " + band_msg, "Helvetica", 9, max_w):
            c.drawString(margin, y, line)
            y -= 0.38 * cm
            if y < margin + 4 * cm:
                y = _new_page(c, width, height, margin)
                c.setFont("Helvetica", 9)

    y -= 0.2 * cm
    y = _draw_section_title(c, margin, y, "Ressalva")
    c.setFont("Helvetica", 9)
    for line in _wrap_text(disclaimer, "Helvetica", 9, max_w):
        c.drawString(margin, y, line)
        y -= 0.38 * cm
        if y < margin + 3 * cm:
            y = _new_page(c, width, height, margin)
            c.setFont("Helvetica", 9)

    y -= 0.2 * cm
    y = _draw_section_title(c, margin, y, "Indicadores")
    c.setFont("Helvetica", 10)

    metrics = [
        ("Palavras", ai_result.get("word_count", 0)),
        ("Frases", ai_result.get("sent_count", 0)),
        ("Diversidade lexical (TTR)", f"{ai_result.get('ttr', 0):.2f}"),
        ("Média palavras/frase", f"{ai_result.get('mean_sent', 0):.1f}"),
        ("Variação entre frases (CV)", f"{ai_result.get('cv_sent', 0):.2f}"),
        ("Conectores por 1.000 palavras", f"{ai_result.get('conn_per_1k', 0):.1f}"),
        ("Generalidades por 1.000 palavras", f"{ai_result.get('vague_per_1k', 0):.1f}"),
        ("Repetição local por 1.000 palavras", f"{ai_result.get('rep_per_1k', 0):.1f}"),
    ]

    for k, v in metrics:
        c.drawString(margin, y, f"- {k}: {v}")
        y -= 0.42 * cm
        if y < margin + 5 * cm:
            y = _new_page(c, width, height, margin)
            c.setFont("Helvetica", 10)

    y -= 0.2 * cm
    y = _draw_section_title(c, margin, y, "Trechos para revisão (heurístico)")
    c.setFont("Helvetica", 9)

    flagged = ai_result.get("flagged_sentences", []) or []
    if not flagged:
        c.drawString(margin, y, "Nenhum trecho foi sinalizado nesta heurística.")
        y -= 0.5 * cm
    else:
        for idx, s in enumerate(flagged[:12], start=1):
            for line in _wrap_text(f"{idx}. {s}", "Helvetica", 9, max_w):
                c.drawString(margin, y, line)
                y -= 0.38 * cm
                if y < margin + 3 * cm:
                    y = _new_page(c, width, height, margin)
                    c.setFont("Helvetica", 9)
            y -= 0.2 * cm
            if y < margin + 3 * cm:
                y = _new_page(c, width, height, margin)
                c.setFont("Helvetica", 9)

    c.save()

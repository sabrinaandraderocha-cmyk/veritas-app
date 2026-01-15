from __future__ import annotations
from typing import List, Dict
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.lib import colors

from veritas_utils import Match

def _wrap_text(text: str, font_name: str, font_size: int, max_width: float):
    words = text.split()
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

    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, y, title)
    y -= 0.8*cm
    c.setFont("Helvetica", 11)
    c.setFillColor(colors.grey)
    c.drawString(margin, y, f"Documento analisado: {query_name}")
    y -= 0.5*cm
    c.drawString(margin, y, f"Índice global de similaridade (estimado): {global_similarity*100:.1f}%")
    y -= 0.8*cm
    c.setFillColor(colors.black)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Parâmetros")
    y -= 0.5*cm
    c.setFont("Helvetica", 10)
    for k, v in params.items():
        c.drawString(margin, y, f"- {k}: {v}")
        y -= 0.42*cm
        if y < margin + 5*cm:
            c.showPage()
            y = height - margin

    y -= 0.2*cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Correspondências relevantes")
    y -= 0.6*cm

    c.setFont("Helvetica", 10)
    max_w = width - 2*margin

    if not matches:
        c.drawString(margin, y, "Nenhuma correspondência acima do limiar foi encontrada.")
        y -= 0.5*cm
    else:
        for idx, m in enumerate(matches[:50], start=1):
            c.setFont("Helvetica-Bold", 10)
            c.drawString(margin, y, f"{idx}. Fonte: {m.source_doc} | Similaridade: {m.score*100:.1f}%")
            y -= 0.45*cm

            c.setFont("Helvetica", 9)
            q_lines = _wrap_text("Trecho analisado: " + m.query_chunk, "Helvetica", 9, max_w)
            s_lines = _wrap_text("Trecho fonte: " + m.source_chunk, "Helvetica", 9, max_w)

            for line in q_lines:
                c.drawString(margin, y, line)
                y -= 0.38*cm
                if y < margin + 4*cm:
                    c.showPage()
                    y = height - margin
                    c.setFont("Helvetica", 9)
            for line in s_lines:
                c.drawString(margin, y, line)
                y -= 0.38*cm
                if y < margin + 4*cm:
                    c.showPage()
                    y = height - margin
                    c.setFont("Helvetica", 9)

            y -= 0.25*cm
            if y < margin + 4*cm:
                c.showPage()
                y = height - margin

    y -= 0.2*cm
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Observação ética")
    y -= 0.5*cm
    c.setFont("Helvetica", 9)
    for line in _wrap_text(disclaimer, "Helvetica", 9, max_w):
        c.drawString(margin, y, line)
        y -= 0.38*cm
        if y < margin + 2*cm:
            c.showPage()
            y = height - margin

    c.save()

"""Étape 6b — Export du compte rendu : PDF (ReportLab) et Word (python-docx)."""
from __future__ import annotations

import io
from datetime import date
from xml.sax.saxutils import escape

ACCENT = (0.16, 0.33, 0.62)  # bleu MeetMind


def _fmt_date(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        return date.fromisoformat(iso).strftime("%d/%m/%Y")
    except ValueError:
        return iso


def _deadline(a: dict) -> str:
    if a.get("deadline_date"):
        return _fmt_date(a["deadline_date"])
    return a.get("deadline") or "—"


# ======================================================================= PDF
def to_pdf(m: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title=f"Compte rendu — {m['title']}", author="MeetMind")
    ss = getSampleStyleSheet()
    accent = colors.Color(*ACCENT)
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=accent, alignment=TA_LEFT, fontSize=20, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=accent, spaceBefore=14, spaceAfter=6)
    h3 = ParagraphStyle("h3", parent=ss["Heading4"], spaceBefore=6, spaceAfter=2)
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=10.5, leading=15)
    meta = ParagraphStyle("meta", parent=body, textColor=colors.grey, fontSize=9.5)
    cell = ParagraphStyle("cell", parent=body, fontSize=9.5, leading=12)

    story = [
        Paragraph(escape(m["title"]), h1),
        Paragraph(f"Compte rendu de réunion · {_fmt_date(m.get('meeting_date'))}", meta),
    ]
    if m.get("participants"):
        story.append(Paragraph("<b>Participants :</b> " + escape(", ".join(m["participants"])), body))
    story.append(Spacer(1, 6))

    story += [Paragraph("Résumé", h2), Paragraph(escape(m.get("summary") or "—"), body)]

    topics = [t for t in m.get("topics", []) if t.get("summary")]
    if len(topics) > 1:
        story.append(Paragraph("Sujets abordés", h2))
        for i, t in enumerate(topics, 1):
            story += [Paragraph(f"{i}. {escape(t['label'])}", h3), Paragraph(escape(t["summary"]), body)]

    story.append(Paragraph("Décisions prises", h2))
    if m.get("decisions"):
        story.append(ListFlowable(
            [ListItem(Paragraph(escape(d["text"]), body), leftIndent=12) for d in m["decisions"]],
            bulletType="bullet", start="•",
        ))
    else:
        story.append(Paragraph("Aucune décision détectée.", meta))

    story.append(Paragraph("Actions à réaliser", h2))
    if m.get("actions"):
        rows = [["Tâche", "Responsable", "Échéance", "Statut"]] + [
            [Paragraph(escape(a["task"]), cell), Paragraph(escape(a.get("owner") or "À définir"), cell),
             Paragraph(escape(_deadline(a)), cell), Paragraph(escape(a.get("status") or "à faire"), cell)]
            for a in m["actions"]
        ]
        table = Table(rows, colWidths=[8.2 * cm, 3.4 * cm, 2.8 * cm, 2.6 * cm], repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), accent),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.96, 0.99)]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.Color(0.8, 0.82, 0.88)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(table)
    else:
        story.append(Paragraph("Aucune action détectée.", meta))

    story += [Spacer(1, 18), Paragraph("Généré automatiquement par MeetMind — à relire avant diffusion.", meta)]
    doc.build(story)
    return buf.getvalue()


# ====================================================================== Word
def to_docx(m: dict) -> bytes:
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt, RGBColor

    accent = RGBColor(int(ACCENT[0] * 255), int(ACCENT[1] * 255), int(ACCENT[2] * 255))
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    def heading(text, level):
        h = doc.add_heading(text, level=level)
        for r in h.runs:
            r.font.color.rgb = accent
        return h

    heading(m["title"], 0)
    p = doc.add_paragraph()
    run = p.add_run(f"Compte rendu de réunion · {_fmt_date(m.get('meeting_date'))}")
    run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)
    if m.get("participants"):
        p = doc.add_paragraph()
        p.add_run("Participants : ").bold = True
        p.add_run(", ".join(m["participants"]))

    heading("Résumé", 1)
    doc.add_paragraph(m.get("summary") or "—")

    topics = [t for t in m.get("topics", []) if t.get("summary")]
    if len(topics) > 1:
        heading("Sujets abordés", 1)
        for i, t in enumerate(topics, 1):
            doc.add_heading(f"{i}. {t['label']}", level=3)
            doc.add_paragraph(t["summary"])

    heading("Décisions prises", 1)
    for d in m.get("decisions") or []:
        doc.add_paragraph(d["text"], style="List Bullet")
    if not m.get("decisions"):
        doc.add_paragraph("Aucune décision détectée.")

    heading("Actions à réaliser", 1)
    if m.get("actions"):
        table = doc.add_table(rows=1, cols=4)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for cell, label in zip(table.rows[0].cells, ["Tâche", "Responsable", "Échéance", "Statut"]):
            cell.text = ""
            r = cell.paragraphs[0].add_run(label)
            r.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear")
            shd.set(qn("w:fill"), "%02X%02X%02X" % tuple(accent))
            cell._tc.get_or_add_tcPr().append(shd)
        for a in m["actions"]:
            cells = table.add_row().cells
            cells[0].text = a["task"]
            cells[1].text = a.get("owner") or "À définir"
            cells[2].text = _deadline(a)
            cells[3].text = a.get("status") or "à faire"
    else:
        doc.add_paragraph("Aucune action détectée.")

    p = doc.add_paragraph()
    p.add_run("\nGénéré automatiquement par MeetMind — à relire avant diffusion.").italic = True

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ============================================================ texte / e-mail
def to_text(m: dict) -> str:
    lines = [f"Compte rendu — {m['title']} ({_fmt_date(m.get('meeting_date'))})", ""]
    if m.get("participants"):
        lines += [f"Participants : {', '.join(m['participants'])}", ""]
    lines += ["RÉSUMÉ", m.get("summary") or "—", "", "DÉCISIONS"]
    lines += [f"  • {d['text']}" for d in m.get("decisions") or []] or ["  (aucune)"]
    lines += ["", "ACTIONS"]
    lines += [
        f"  • {a['task']} — {a.get('owner') or 'À définir'} — échéance : {_deadline(a)}"
        for a in m.get("actions") or []
    ] or ["  (aucune)"]
    lines += ["", "— Généré par MeetMind"]
    return "\n".join(lines)

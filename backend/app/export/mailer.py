"""Envoi du compte rendu par e-mail aux participants (SMTP)."""
from __future__ import annotations

import re
import smtplib
from email.message import EmailMessage

from ..config import settings
from ..nlp.textutils import strip_accents
from .documents import to_pdf, to_text

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class MailNotConfigured(RuntimeError):
    pass


def send_report(meeting: dict, recipients: list[str], attach_pdf: bool = True) -> None:
    if not (settings.smtp_host and settings.smtp_user and settings.smtp_password):
        raise MailNotConfigured("SMTP non configuré : renseignez SMTP_HOST, SMTP_USER, SMTP_PASSWORD dans .env")
    bad = [r for r in recipients if not EMAIL_RE.match(r)]
    if bad or not recipients:
        raise ValueError(f"Adresses invalides : {', '.join(bad) or '(aucune)'}")

    msg = EmailMessage()
    msg["Subject"] = f"Compte rendu — {meeting['title']}"
    msg["From"] = settings.smtp_from or settings.smtp_user
    msg["To"] = ", ".join(recipients)
    msg.set_content(to_text(meeting))
    if attach_pdf:
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", strip_accents(meeting["title"]))[:60].strip("_") or "reunion"
        msg.add_attachment(to_pdf(meeting), maintype="application", subtype="pdf",
                           filename=f"compte_rendu_{safe}.pdf")

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as s:
        s.starttls()
        s.login(settings.smtp_user, settings.smtp_password)
        s.send_message(msg)

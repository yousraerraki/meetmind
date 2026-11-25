"""Interface MeetMind (prototype Streamlit).

Lancer :  streamlit run frontend/streamlit_app.py
(l'API doit tourner : uvicorn backend.app.main:app --reload)
"""
from __future__ import annotations

import os
from datetime import date

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
STATUSES = ["à faire", "en cours", "fait"]

st.set_page_config(page_title="MeetMind", page_icon="📝", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      .mm-card {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:1rem 1.2rem; margin-bottom:.8rem;}
      .mm-muted {color: grey; font-size: .9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ API
def api(method: str, path: str, **kwargs):
    try:
        r = requests.request(method, f"{API}{path}", timeout=600, **kwargs)
    except requests.ConnectionError:
        st.error(f"Impossible de joindre l'API ({API}). Lancez : `uvicorn backend.app.main:app --reload`")
        st.stop()
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = r.text
        st.error(f"Erreur {r.status_code} : {detail}")
        return None
    return r


# --------------------------------------------------------- affichage CR
def show_report(m: dict, key: str) -> None:
    st.subheader(m["title"])
    st.caption(f"📅 {m['meeting_date']}  ·  👥 {', '.join(m['participants']) or '—'}  ·  🌐 {m.get('language', '')}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Sujets", len(m.get("topics", [])))
    c2.metric("Décisions", len(m["decisions"]))
    c3.metric("Actions", len(m["actions"]))

    tab_cr, tab_edit, tab_export, tab_tr = st.tabs(["📄 Compte rendu", "✏️ Éditer", "📤 Exporter", "🗒️ Transcription"])

    with tab_cr:
        st.markdown("#### Résumé")
        st.write(m["summary"] or "—")
        topics = [t for t in m.get("topics", []) if t.get("summary")]
        if len(topics) > 1:
            st.markdown("#### Sujets abordés")
            for i, t in enumerate(topics, 1):
                with st.expander(f"{i}. {t['label']}"):
                    st.write(t["summary"])
        st.markdown("#### ✅ Décisions prises")
        if m["decisions"]:
            for d in m["decisions"]:
                st.markdown(f"- {d['text']}" + (f"  <span class='mm-muted'>— {d['speaker']}</span>" if d.get("speaker") else ""),
                            unsafe_allow_html=True)
        else:
            st.info("Aucune décision détectée.")
        st.markdown("#### 📌 Actions à réaliser")
        if m["actions"]:
            st.dataframe(
                [{"Tâche": a["task"], "Responsable": a["owner"] or "À définir",
                  "Échéance": a["deadline_date"] or a["deadline"] or "—", "Statut": a["status"]} for a in m["actions"]],
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("Aucune action détectée.")

    with tab_edit:
        st.caption("Corrigez le compte rendu généré, puis enregistrez.")
        title = st.text_input("Titre", m["title"], key=f"{key}-title")
        summary = st.text_area("Résumé", m["summary"], height=160, key=f"{key}-sum")
        participants = st.text_input("Participants (séparés par des virgules)", ", ".join(m["participants"]),
                                     key=f"{key}-parts")
        st.markdown("**Décisions**")
        decisions = st.data_editor(
            [{"text": d["text"], "speaker": d.get("speaker") or ""} for d in m["decisions"]] or [{"text": "", "speaker": ""}],
            num_rows="dynamic", use_container_width=True, key=f"{key}-dec",
            column_config={"text": st.column_config.TextColumn("Décision", width="large"),
                           "speaker": st.column_config.TextColumn("Proposée par")},
        )
        st.markdown("**Actions**")
        actions = st.data_editor(
            [{"task": a["task"], "owner": a["owner"] or "", "deadline_date": a["deadline_date"] or "",
              "deadline": a["deadline"] or "", "status": a["status"]} for a in m["actions"]]
            or [{"task": "", "owner": "", "deadline_date": "", "deadline": "", "status": "à faire"}],
            num_rows="dynamic", use_container_width=True, key=f"{key}-act",
            column_config={
                "task": st.column_config.TextColumn("Tâche", width="large"),
                "owner": st.column_config.TextColumn("Responsable"),
                "deadline_date": st.column_config.TextColumn("Échéance (AAAA-MM-JJ)"),
                "deadline": st.column_config.TextColumn("Échéance (texte)"),
                "status": st.column_config.SelectboxColumn("Statut", options=STATUSES),
            },
        )
        if st.button("💾 Enregistrer les corrections", key=f"{key}-save", type="primary"):
            payload = {
                "title": title, "summary": summary,
                "participants": [p.strip() for p in participants.split(",") if p.strip()],
                "decisions": [d for d in decisions if (d.get("text") or "").strip()],
                "actions": [{k: (v or None) if k != "task" else v for k, v in a.items()}
                            for a in actions if (a.get("task") or "").strip()],
            }
            if api("PUT", f"/meetings/{m['id']}", json=payload):
                st.success("Compte rendu mis à jour.")
                st.session_state.pop("current", None)
                st.rerun()

    with tab_export:
        c1, c2, c3 = st.columns(3)
        for col, fmt, label in [(c1, "pdf", "📕 PDF"), (c2, "docx", "📘 Word"), (c3, "txt", "📄 Texte")]:
            r = api("GET", f"/meetings/{m['id']}/export/{fmt}")
            if r:
                col.download_button(label, r.content, file_name=f"compte_rendu_{m['id']}.{fmt}",
                                    use_container_width=True, key=f"{key}-dl-{fmt}")
        st.markdown("**Envoyer par e-mail aux participants**")
        with st.form(f"{key}-mail"):
            to = st.text_input("Destinataires (séparés par des virgules)")
            attach = st.checkbox("Joindre le PDF", value=True)
            if st.form_submit_button("📧 Envoyer"):
                rec = [x.strip() for x in to.split(",") if x.strip()]
                if api("POST", f"/meetings/{m['id']}/email", json={"recipients": rec, "attach_pdf": attach}):
                    st.success(f"Envoyé à {len(rec)} destinataire(s).")

    with tab_tr:
        st.text_area("Transcription nettoyée", m.get("clean_transcript", ""), height=350, disabled=True,
                     key=f"{key}-tr")
        with st.expander("Modèles utilisés et temps de traitement"):
            st.json({"modèles": m.get("models_used", {}), "temps (s)": m.get("timings", {})})


# ================================================================ pages
def page_new():
    st.title("📝 Nouvelle réunion")
    st.caption("Importez la transcription ou l'audio : MeetMind génère le résumé, les décisions et les actions.")
    c1, c2 = st.columns([3, 1])
    title = c1.text_input("Titre de la réunion", "Réunion d'équipe")
    mdate = c2.date_input("Date", date.today(), format="DD/MM/YYYY")

    tab_text, tab_file, tab_rec = st.tabs(["Coller un texte", "Charger un fichier", "🎙️ Enregistrer"])
    with tab_text:
        text = st.text_area(
            "Transcription", height=260,
            placeholder="Karim : Bonjour à tous…\nYousra : J'ai testé BARThez…\n\n(format conseillé : « Nom : phrase », une intervention par ligne)",
        )
    with tab_file:
        upl = st.file_uploader("Fichier .txt, .docx, .pdf, .vtt/.srt (Teams, Zoom) ou audio .mp3/.wav",
                               type=["txt", "docx", "pdf", "vtt", "srt", "mp3", "wav", "m4a"])
        lang = st.selectbox("Langue de l'audio", ["fr", "en", "ar", "auto"], index=0,
                            help="« ar » / « auto » pour une réunion en darija")
        c_a, c_b = st.columns([1, 3])
        n_spk = c_a.number_input("Nb de locuteurs", 0, 20, 0, help="0 = détection automatique")
        spk_names = c_b.text_input("Noms des locuteurs (optionnel)", placeholder="SPEAKER_00=Yousra, SPEAKER_01=Rania")
        vocab = st.text_input("Noms des participants et mots importants", "Yousra, Rania",
                              help="Aide Whisper à bien écrire les prénoms et les termes techniques")

    with tab_rec:
        recorded = None
        if hasattr(st, "audio_input"):  # Streamlit ≥ 1.40
            recorded = st.audio_input("Cliquez sur le micro, parlez, puis arrêtez l'enregistrement")
        else:
            st.info("Mettez Streamlit à jour (pip install -U streamlit) pour enregistrer au micro.")

    if st.button("🚀 Générer le compte rendu", type="primary", use_container_width=True):
        data = {"title": title, "meeting_date": mdate.isoformat(),
                "audio_language": "" if lang == "auto" else lang, "speaker_names": spk_names,
                "vocabulary": vocab}
        if n_spk:
            data["num_speakers"] = int(n_spk)
        files = None
        if recorded is not None:
            files = {"file": ("enregistrement.wav", recorded.getvalue())}
        elif upl is not None:
            files = {"file": (upl.name, upl.getvalue())}
        elif text.strip():
            data["text"] = text
        else:
            st.warning("Collez un texte ou chargez un fichier.")
            return
        with st.spinner("Analyse en cours : transcription → nettoyage → résumé · décisions · actions…"):
            r = api("POST", "/meetings/analyze", data=data, files=files)
        if r:
            st.session_state["current"] = r.json()["id"]

    if st.session_state.get("current"):
        r = api("GET", f"/meetings/{st.session_state['current']}")
        if r:
            st.divider()
            m = r.json()
            st.success(f"Compte rendu généré en {m.get('timings', {}).get('total', '?')} s")
            show_report(m, key=f"new-{m['id']}")


def page_history():
    st.title("🗂️ Historique des réunions")
    q = st.text_input("🔎 Rechercher (mot-clé, personne, décision…)")
    r = api("GET", "/meetings", params={"q": q} if q else None)
    meetings = r.json() if r else []
    if not meetings:
        st.info("Aucune réunion trouvée.")
        return
    options = {f"{m['meeting_date']} — {m['title']}  ({m['n_decisions']} décisions, {m['n_actions']} actions)": m["id"]
               for m in meetings}
    choice = st.selectbox(f"{len(meetings)} réunion(s)", list(options))
    mid = options[choice]
    r = api("GET", f"/meetings/{mid}")
    if r:
        show_report(r.json(), key=f"hist-{mid}")
        with st.expander("🗑️ Supprimer cette réunion"):
            if st.button("Confirmer la suppression", key=f"del-{mid}"):
                api("DELETE", f"/meetings/{mid}")
                st.rerun()


def page_actions():
    st.title("📌 Suivi des actions")
    c1, c2 = st.columns(2)
    owner = c1.text_input("Responsable")
    status = c2.selectbox("Statut", ["Tous"] + STATUSES)
    params = {}
    if owner:
        params["owner"] = owner
    if status != "Tous":
        params["status"] = status
    r = api("GET", "/actions", params=params)
    rows = r.json() if r else []
    if not rows:
        st.info("Aucune action.")
        return
    today = date.today().isoformat()
    late = sum(1 for a in rows if a["deadline_date"] and a["deadline_date"] < today and a["status"] != "fait")
    c1, c2, c3 = st.columns(3)
    c1.metric("Actions", len(rows))
    c2.metric("Terminées", sum(a["status"] == "fait" for a in rows))
    c3.metric("En retard", late)

    edited = st.data_editor(
        [{"id": a["id"], "Tâche": a["task"], "Responsable": a["owner"] or "À définir",
          "Échéance": a["deadline_date"] or a["deadline"] or "—", "Réunion": a["meeting_title"],
          "Statut": a["status"]} for a in rows],
        hide_index=True, use_container_width=True, disabled=["id", "Tâche", "Responsable", "Échéance", "Réunion"],
        column_config={"id": None, "Statut": st.column_config.SelectboxColumn("Statut", options=STATUSES)},
        key="actions-editor",
    )
    if st.button("💾 Enregistrer les statuts", type="primary"):
        before = {a["id"]: a["status"] for a in rows}
        changed = [e for e in edited if before.get(e["id"]) != e["Statut"]]
        for e in changed:
            api("PATCH", f"/actions/{e['id']}", json={"status": e["Statut"]})
        st.success(f"{len(changed)} action(s) mise(s) à jour.")
        st.rerun()


# ================================================================= main
with st.sidebar:
    st.markdown("## 📝 MeetMind")
    st.caption("Assistant intelligent de comptes rendus de réunion")
    page = st.radio("Navigation", ["Nouvelle réunion", "Historique", "Suivi des actions"], label_visibility="collapsed")
    st.divider()
    try:
        h = requests.get(f"{API}/health", timeout=5).json()
        st.caption("**Modèles actifs**")
        for k, v in h["models"].items():
            st.caption(f"• {k} : {v}")
    except Exception:
        st.caption("⚠️ API non joignable")
    st.caption("Yousra Erraki — 2026-2027")

{"Nouvelle réunion": page_new, "Historique": page_history, "Suivi des actions": page_actions}[page]()

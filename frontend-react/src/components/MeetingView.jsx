import { useEffect, useState } from "react";
import { api, downloadBlob, formatDate, STATUSES } from "../api.js";

const TABS = [
  ["cr", "Compte rendu"],
  ["edit", "Éditer"],
  ["export", "Exporter"],
  ["transcript", "Transcription"],
];

export default function MeetingView({ meeting, onChange, onDelete }) {
  const [tab, setTab] = useState("cr");
  useEffect(() => setTab("cr"), [meeting.id]);
  const topics = (meeting.topics || []).filter((t) => t.summary);

  return (
    <section className="card meeting">
      <div className="meeting-head">
        <div>
          <h2>{meeting.title}</h2>
          <p className="muted">
            {formatDate(meeting.meeting_date)} · {meeting.participants.join(", ") || "participants non détectés"}
          </p>
        </div>
        <div className="stats">
          <Stat n={topics.length || 1} label="sujets" />
          <Stat n={meeting.decisions.length} label="décisions" />
          <Stat n={meeting.actions.length} label="actions" />
        </div>
      </div>

      <div className="tabs">
        {TABS.map(([id, label]) => (
          <button key={id} className={tab === id ? "on" : ""} onClick={() => setTab(id)}>{label}</button>
        ))}
      </div>

      {tab === "cr" && <Report meeting={meeting} topics={topics} />}
      {tab === "edit" && <Editor meeting={meeting} onSaved={(m) => { onChange?.(m); setTab("cr"); }} />}
      {tab === "export" && <Exporter meeting={meeting} />}
      {tab === "transcript" && <Transcript meeting={meeting} />}

      {onDelete && (
        <div className="danger-zone">
          <button className="ghost danger" onClick={() => confirm("Supprimer définitivement cette réunion ?") && onDelete()}>
            Supprimer cette réunion
          </button>
        </div>
      )}
    </section>
  );
}

function Stat({ n, label }) {
  return <div className="stat"><b>{n}</b><span>{label}</span></div>;
}

function Report({ meeting, topics }) {
  return (
    <div className="report">
      <h3>Résumé</h3>
      <p className="summary">{meeting.summary || "—"}</p>

      {topics.length > 1 && (
        <>
          <h3>Sujets abordés</h3>
          {topics.map((t, i) => (
            <details key={t.id} className="topic">
              <summary><span className="badge">{i + 1}</span>{t.label}</summary>
              <p>{t.summary}</p>
            </details>
          ))}
        </>
      )}

      <h3>Décisions prises</h3>
      {meeting.decisions.length ? (
        <ul className="decisions">
          {meeting.decisions.map((d) => (
            <li key={d.id}>{d.text}{d.speaker && <span className="muted"> — {d.speaker}</span>}</li>
          ))}
        </ul>
      ) : <p className="muted">Aucune décision détectée.</p>}

      <h3>Actions à réaliser</h3>
      {meeting.actions.length ? (
        <table className="table">
          <thead><tr><th>Tâche</th><th>Responsable</th><th>Échéance</th><th>Statut</th></tr></thead>
          <tbody>
            {meeting.actions.map((a) => (
              <tr key={a.id}>
                <td title={a.source}>{a.task}</td>
                <td>{a.owner || <span className="muted">À définir</span>}</td>
                <td>{a.deadline_date ? formatDate(a.deadline_date) : a.deadline || "—"}</td>
                <td><span className={`pill s-${a.status.replace(" ", "-")}`}>{a.status}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : <p className="muted">Aucune action détectée.</p>}
    </div>
  );
}

function Editor({ meeting, onSaved }) {
  const [title, setTitle] = useState(meeting.title);
  const [summary, setSummary] = useState(meeting.summary);
  const [participants, setParticipants] = useState(meeting.participants.join(", "));
  const [decisions, setDecisions] = useState(meeting.decisions.map(({ text, speaker }) => ({ text, speaker: speaker || "" })));
  const [actions, setActions] = useState(
    meeting.actions.map((a) => ({ task: a.task, owner: a.owner || "", deadline_date: a.deadline_date || "",
                                  deadline: a.deadline || "", status: a.status, source: a.source || "" }))
  );
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const setDec = (i, k, v) => setDecisions(decisions.map((d, j) => (j === i ? { ...d, [k]: v } : d)));
  const setAct = (i, k, v) => setActions(actions.map((a, j) => (j === i ? { ...a, [k]: v } : a)));

  async function save() {
    setSaving(true);
    setError("");
    try {
      const m = await api.update(meeting.id, {
        title, summary,
        participants: participants.split(",").map((p) => p.trim()).filter(Boolean),
        decisions: decisions.filter((d) => d.text.trim()).map((d) => ({ text: d.text, speaker: d.speaker || null })),
        actions: actions.filter((a) => a.task.trim()).map((a) => ({
          ...a, owner: a.owner || null, deadline: a.deadline || null, deadline_date: a.deadline_date || null,
        })),
      });
      onSaved(m);
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="editor">
      <label className="field"><span>Titre</span><input value={title} onChange={(e) => setTitle(e.target.value)} /></label>
      <label className="field"><span>Résumé</span><textarea rows={5} value={summary} onChange={(e) => setSummary(e.target.value)} /></label>
      <label className="field"><span>Participants (séparés par des virgules)</span>
        <input value={participants} onChange={(e) => setParticipants(e.target.value)} /></label>

      <h4>Décisions</h4>
      {decisions.map((d, i) => (
        <div className="edit-row" key={i}>
          <input className="grow" value={d.text} onChange={(e) => setDec(i, "text", e.target.value)} placeholder="Décision" />
          <input value={d.speaker} onChange={(e) => setDec(i, "speaker", e.target.value)} placeholder="Proposée par" />
          <button className="icon" title="Supprimer" onClick={() => setDecisions(decisions.filter((_, j) => j !== i))}>✕</button>
        </div>
      ))}
      <button className="ghost" onClick={() => setDecisions([...decisions, { text: "", speaker: "" }])}>+ Ajouter une décision</button>

      <h4>Actions</h4>
      {actions.map((a, i) => (
        <div className="edit-row" key={i}>
          <input className="grow" value={a.task} onChange={(e) => setAct(i, "task", e.target.value)} placeholder="Tâche" />
          <input value={a.owner} onChange={(e) => setAct(i, "owner", e.target.value)} placeholder="Responsable" />
          <input type="date" value={a.deadline_date} onChange={(e) => setAct(i, "deadline_date", e.target.value)} />
          <select value={a.status} onChange={(e) => setAct(i, "status", e.target.value)}>
            {STATUSES.map((s) => <option key={s}>{s}</option>)}
          </select>
          <button className="icon" title="Supprimer" onClick={() => setActions(actions.filter((_, j) => j !== i))}>✕</button>
        </div>
      ))}
      <button className="ghost" onClick={() => setActions([...actions, { task: "", owner: "", deadline_date: "", deadline: "", status: "à faire", source: "" }])}>
        + Ajouter une action
      </button>

      {error && <div className="alert error">{error}</div>}
      <div className="actions-bar">
        <button className="primary" onClick={save} disabled={saving}>{saving ? "Enregistrement…" : "Enregistrer les corrections"}</button>
      </div>
    </div>
  );
}

function Exporter({ meeting }) {
  const [busy, setBusy] = useState("");
  const [to, setTo] = useState("");
  const [attach, setAttach] = useState(true);
  const [msg, setMsg] = useState(null);

  async function dl(fmt) {
    setBusy(fmt);
    try {
      downloadBlob(await api.exportFile(meeting.id, fmt), `compte_rendu_${meeting.id}.${fmt}`);
    } catch (e) {
      setMsg({ type: "error", text: e.message });
    } finally {
      setBusy("");
    }
  }

  async function send(e) {
    e.preventDefault();
    const recipients = to.split(/[,;\s]+/).filter(Boolean);
    setBusy("mail");
    setMsg(null);
    try {
      await api.email(meeting.id, recipients, attach);
      setMsg({ type: "success", text: `Envoyé à ${recipients.length} destinataire(s).` });
    } catch (err) {
      setMsg({ type: "error", text: err.message });
    } finally {
      setBusy("");
    }
  }

  return (
    <div className="exporter">
      <div className="export-grid">
        {[["pdf", "PDF", "Pour diffuser ou archiver"], ["docx", "Word", "Pour retravailler le texte"], ["txt", "Texte", "Pour coller dans un message"]].map(
          ([fmt, label, hint]) => (
            <button key={fmt} className="export-btn" onClick={() => dl(fmt)} disabled={!!busy}>
              <b>{busy === fmt ? "…" : label}</b><small>{hint}</small>
            </button>
          )
        )}
      </div>
      <form className="mail" onSubmit={send}>
        <h4>Envoyer par e-mail aux participants</h4>
        <input value={to} onChange={(e) => setTo(e.target.value)} placeholder="adresse1@exemple.com, adresse2@exemple.com" />
        <label className="check"><input type="checkbox" checked={attach} onChange={(e) => setAttach(e.target.checked)} /> Joindre le PDF</label>
        <button className="primary" disabled={!to.trim() || busy === "mail"}>{busy === "mail" ? "Envoi…" : "Envoyer"}</button>
      </form>
      {msg && <div className={`alert ${msg.type}`}>{msg.text}</div>}
    </div>
  );
}

function Transcript({ meeting }) {
  return (
    <div>
      <pre className="transcript">{meeting.clean_transcript}</pre>
      <details className="tech">
        <summary>Modèles utilisés et temps de traitement</summary>
        <table className="table compact">
          <tbody>
            {Object.entries(meeting.models_used || {}).map(([k, v]) => <tr key={k}><td>{k}</td><td>{v}</td></tr>)}
            {Object.entries(meeting.timings || {}).map(([k, v]) => <tr key={"t" + k}><td>temps · {k}</td><td>{v} s</td></tr>)}
          </tbody>
        </table>
      </details>
    </div>
  );
}

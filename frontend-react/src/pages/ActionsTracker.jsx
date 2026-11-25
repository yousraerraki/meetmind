import { useEffect, useState } from "react";
import { api, formatDate, STATUSES } from "../api.js";

export default function ActionsTracker({ onOpenMeeting }) {
  const [owner, setOwner] = useState("");
  const [status, setStatus] = useState("");
  const [rows, setRows] = useState(null);
  const [error, setError] = useState("");

  const load = () => api.actions({ owner, status }).then(setRows).catch((e) => setError(e.message));

  useEffect(() => {
    const t = setTimeout(load, 250);
    return () => clearTimeout(t);
  }, [owner, status]); // eslint-disable-line react-hooks/exhaustive-deps

  async function change(a, s) {
    setRows(rows.map((r) => (r.id === a.id ? { ...r, status: s } : r)));
    try {
      await api.setStatus(a.id, s);
    } catch (e) {
      setError(e.message);
      load();
    }
  }

  const today = new Date().toISOString().slice(0, 10);
  const isLate = (a) => a.deadline_date && a.deadline_date < today && a.status !== "fait";
  const done = rows?.filter((a) => a.status === "fait").length || 0;
  const late = rows?.filter(isLate).length || 0;

  return (
    <>
      <header className="page-head">
        <h1>Suivi des actions</h1>
        <p>Toutes les actions décidées en réunion, triées par échéance.</p>
      </header>
      {error && <div className="alert error">{error}</div>}

      <div className="kpis">
        <div className="card kpi"><b>{rows?.length ?? "–"}</b><span>actions</span></div>
        <div className="card kpi"><b>{done}</b><span>terminées</span></div>
        <div className={`card kpi ${late ? "late" : ""}`}><b>{late}</b><span>en retard</span></div>
      </div>

      <section className="card">
        <div className="row">
          <label className="field grow"><span>Responsable</span>
            <input value={owner} onChange={(e) => setOwner(e.target.value)} placeholder="ex. Yousra" /></label>
          <label className="field"><span>Statut</span>
            <select value={status} onChange={(e) => setStatus(e.target.value)}>
              <option value="">Tous</option>
              {STATUSES.map((s) => <option key={s}>{s}</option>)}
            </select></label>
        </div>
        {rows?.length === 0 && <p className="muted">Aucune action.</p>}
        {rows?.length > 0 && (
          <table className="table">
            <thead><tr><th>Tâche</th><th>Responsable</th><th>Échéance</th><th>Réunion</th><th>Statut</th></tr></thead>
            <tbody>
              {rows.map((a) => (
                <tr key={a.id} className={isLate(a) ? "late" : ""}>
                  <td>{a.task}</td>
                  <td>{a.owner || <span className="muted">À définir</span>}</td>
                  <td>{a.deadline_date ? formatDate(a.deadline_date) : a.deadline || "—"}{isLate(a) && <span className="pill s-late">en retard</span>}</td>
                  <td><button className="link" onClick={() => onOpenMeeting(a.meeting_id)}>{a.meeting_title}</button></td>
                  <td>
                    <select className={`status s-${a.status.replace(" ", "-")}`} value={a.status} onChange={(e) => change(a, e.target.value)}>
                      {STATUSES.map((s) => <option key={s}>{s}</option>)}
                    </select>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </>
  );
}

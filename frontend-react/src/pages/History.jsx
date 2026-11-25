import { useEffect, useState } from "react";
import { api, formatDate } from "../api.js";
import MeetingView from "../components/MeetingView.jsx";

export default function History({ initialId }) {
  const [q, setQ] = useState("");
  const [items, setItems] = useState(null);
  const [selected, setSelected] = useState(initialId || null);
  const [meeting, setMeeting] = useState(null);
  const [error, setError] = useState("");

  const load = (query) =>
    api.list(query).then((list) => {
      setItems(list);
      if (!selected && list.length) setSelected(list[0].id);
    }).catch((e) => setError(e.message));

  useEffect(() => {
    const t = setTimeout(() => load(q), 250); // recherche à la frappe
    return () => clearTimeout(t);
  }, [q]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (selected) api.get(selected).then(setMeeting).catch((e) => setError(e.message));
    else setMeeting(null);
  }, [selected]);

  async function remove() {
    await api.remove(meeting.id);
    setSelected(null);
    setMeeting(null);
    load(q);
  }

  return (
    <>
      <header className="page-head">
        <h1>Historique des réunions</h1>
        <p>Retrouvez une réunion par mot-clé, personne, décision ou action.</p>
      </header>
      {error && <div className="alert error">{error}</div>}
      <div className="history">
        <aside className="card list">
          <input className="search" placeholder="Rechercher…" value={q} onChange={(e) => setQ(e.target.value)} />
          {items === null && <p className="muted">Chargement…</p>}
          {items?.length === 0 && <p className="muted">Aucune réunion trouvée.</p>}
          {items?.map((m) => (
            <button key={m.id} className={`list-item ${selected === m.id ? "on" : ""}`} onClick={() => setSelected(m.id)}>
              <b>{m.title}</b>
              <small>{formatDate(m.meeting_date)} · {m.n_decisions} déc. · {m.n_actions} actions
                {m.n_actions_open ? ` (${m.n_actions_open} ouvertes)` : ""}</small>
            </button>
          ))}
        </aside>
        <div className="detail">
          {meeting ? (
            <MeetingView meeting={meeting} onChange={(m) => { setMeeting(m); load(q); }} onDelete={remove} />
          ) : (
            items?.length > 0 && <p className="muted">Sélectionnez une réunion.</p>
          )}
        </div>
      </div>
    </>
  );
}

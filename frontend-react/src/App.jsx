import { useEffect, useState } from "react";
import { api } from "./api.js";
import NewMeeting from "./pages/NewMeeting.jsx";
import History from "./pages/History.jsx";
import ActionsTracker from "./pages/ActionsTracker.jsx";

const PAGES = [
  { id: "new", label: "Nouvelle réunion", icon: "＋" },
  { id: "history", label: "Historique", icon: "🗂" },
  { id: "actions", label: "Suivi des actions", icon: "✓" },
];

export default function App() {
  const [page, setPage] = useState(() => window.location.hash.slice(1) || "new");
  const [health, setHealth] = useState(null);
  const [openId, setOpenId] = useState(null); // réunion à ouvrir dans l'historique

  useEffect(() => {
    window.location.hash = page;
  }, [page]);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ error: true }));
  }, []);

  const openMeeting = (id) => {
    setOpenId(id);
    setPage("history");
  };

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <span className="logo">M</span>
          <div>
            <strong>MeetMind</strong>
            <small>Comptes rendus intelligents</small>
          </div>
        </div>
        <nav>
          {PAGES.map((p) => (
            <button key={p.id} className={`nav-item ${page === p.id ? "active" : ""}`} onClick={() => setPage(p.id)}>
              <span className="nav-icon">{p.icon}</span>
              {p.label}
            </button>
          ))}
        </nav>
        <div className="sidebar-foot">
          <div className="models">
            <span className="label">Modèles actifs</span>
            {health?.error && <span className="warn">⚠ API non joignable</span>}
            {health?.models &&
              Object.entries(health.models).map(([k, v]) => (
                <div key={k} className="model-row">
                  <b>{k}</b>
                  <span>{v}</span>
                </div>
              ))}
          </div>
          <small className="credits">Yousra Erraki<br />2026 – 2027</small>
        </div>
      </aside>

      <main className="content">
        {page === "new" && <NewMeeting onOpen={openMeeting} />}
        {page === "history" && <History initialId={openId} />}
        {page === "actions" && <ActionsTracker onOpenMeeting={openMeeting} />}
      </main>
    </div>
  );
}

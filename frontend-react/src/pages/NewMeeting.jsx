import { useRef, useState } from "react";
import { api } from "../api.js";
import MeetingView from "../components/MeetingView.jsx";
import Recorder from "../components/Recorder.jsx";

const ACCEPT = ".txt,.docx,.pdf,.vtt,.srt,.mp3,.wav,.m4a";
const AUDIO = /\.(mp3|wav|m4a|ogg|flac|webm)$/i;
const today = () => new Date().toISOString().slice(0, 10);

export default function NewMeeting() {
  const [title, setTitle] = useState("Réunion d'équipe");
  const [date, setDate] = useState(today());
  const [mode, setMode] = useState("text");
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [recording, setRecording] = useState(null); // fichier enregistré au micro
  const [lang, setLang] = useState("fr");
  const [vocabulary, setVocabulary] = useState("Yousra, Rania");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [meeting, setMeeting] = useState(null);
  const [drag, setDrag] = useState(false);
  const inputRef = useRef();

  const isAudio = (mode === "file" && file && AUDIO.test(file.name)) || mode === "record";

  async function submit() {
    setError("");
    const form = new FormData();
    form.append("title", title || "Réunion");
    form.append("meeting_date", date);
    form.append("audio_language", lang === "auto" ? "" : lang);
    if (vocabulary.trim()) form.append("vocabulary", vocabulary);
    if (mode === "file") {
      if (!file) return setError("Choisissez un fichier.");
      form.append("file", file);
    } else if (mode === "record") {
      if (!recording) return setError("Enregistrez d'abord la réunion (bouton 🎙️), puis arrêtez l'enregistrement.");
      form.append("file", recording);
    } else {
      if (!text.trim()) return setError("Collez la transcription de la réunion.");
      form.append("text", text);
    }
    setLoading(true);
    setMeeting(null);
    try {
      setMeeting(await api.analyze(form));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function loadSample() {
    const res = await fetch("/sample.txt");
    setText(await res.text());
    setTitle("Réunion projet MeetMind");
    setMode("text");
  }

  return (
    <>
      <header className="page-head">
        <h1>Nouvelle réunion</h1>
        <p>Importez la transcription ou l'audio : MeetMind produit le résumé, les décisions et les actions.</p>
      </header>

      <section className="card">
        <div className="row">
          <label className="field grow">
            <span>Titre de la réunion</span>
            <input value={title} onChange={(e) => setTitle(e.target.value)} />
          </label>
          <label className="field">
            <span>Date</span>
            <input type="date" value={date} onChange={(e) => setDate(e.target.value)} />
          </label>
        </div>

        <div className="segmented">
          <button className={mode === "text" ? "on" : ""} onClick={() => setMode("text")}>Coller un texte</button>
          <button className={mode === "file" ? "on" : ""} onClick={() => setMode("file")}>Charger un fichier</button>
          <button className={mode === "record" ? "on" : ""} onClick={() => setMode("record")}>🎙️ Enregistrer</button>
        </div>

        {mode === "text" && (
          <label className="field">
            <span className="between">
              Transcription
              <button className="link" type="button" onClick={loadSample}>Charger un exemple</button>
            </span>
            <textarea
              rows={12}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder={"Karim : Bonjour à tous…\nYousra : J'ai testé BARThez…\n\nFormat conseillé : « Nom : phrase », une intervention par ligne."}
            />
          </label>
        )}
        {mode === "file" && (
            <div
              className={`dropzone ${drag ? "drag" : ""}`}
              onClick={() => inputRef.current.click()}
              onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => { e.preventDefault(); setDrag(false); setFile(e.dataTransfer.files[0]); }}
            >
              <input ref={inputRef} type="file" accept={ACCEPT} hidden onChange={(e) => setFile(e.target.files[0])} />
              {file ? (
                <><strong>{file.name}</strong><small>{(file.size / 1024 / 1024).toFixed(2)} Mo · cliquer pour changer</small></>
              ) : (
                <><strong>Déposez un fichier ici</strong><small>.txt · .docx · .pdf · .vtt/.srt (Teams, Zoom) · audio .mp3 / .wav</small></>
              )}
            </div>
        )}
        {mode === "record" && <Recorder onRecorded={setRecording} />}
        {isAudio && (
              <div className="row">
                <label className="field">
                  <span>Langue de l'audio</span>
                  <select value={lang} onChange={(e) => setLang(e.target.value)}>
                    <option value="fr">Français</option>
                    <option value="en">Anglais</option>
                    <option value="ar">Arabe / darija</option>
                    <option value="auto">Détection auto</option>
                  </select>
                </label>
                <label className="field grow">
                  <span>Noms des participants et mots importants</span>
                  <input placeholder="Yousra, Rania, BARThez, encadrant…" value={vocabulary}
                         onChange={(e) => setVocabulary(e.target.value)} />
                  <small className="muted">Aide Whisper à bien écrire les prénoms et les termes techniques.</small>
                </label>
              </div>
            )}
        {error && <div className="alert error">{error}</div>}
        <button className="primary big" onClick={submit} disabled={loading}>
          {loading ? <><span className="spinner" /> Analyse en cours…</> : "Générer le compte rendu"}
        </button>
        {loading && (
          <p className="muted center">
            {isAudio ? "Transcription → " : ""}nettoyage → résumé · décisions · actions (en parallèle)
          </p>
        )}
      </section>

      {meeting && (
        <>
          <div className="alert success">
            Compte rendu généré en {meeting.timings?.total ?? "?"} s — enregistré dans l'historique.
          </div>
          <MeetingView meeting={meeting} onChange={setMeeting} />
        </>
      )}
    </>
  );
}

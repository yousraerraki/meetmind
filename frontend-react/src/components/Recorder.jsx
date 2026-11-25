import { useEffect, useRef, useState } from "react";

// Enregistrement direct depuis le micro du navigateur (MediaRecorder).
// Fonctionne sur http://localhost (contexte sécurisé) et en HTTPS.
const pickMime = () => {
  const types = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
  return types.find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t)) || "";
};
const extFor = (mime) => (mime.includes("mp4") ? "m4a" : mime.includes("ogg") ? "ogg" : "webm");
const fmt = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

export default function Recorder({ onRecorded }) {
  const [state, setState] = useState("idle"); // idle | recording | done
  const [seconds, setSeconds] = useState(0);
  const [level, setLevel] = useState(0);
  const [url, setUrl] = useState("");
  const [error, setError] = useState("");
  const rec = useRef(null);
  const chunks = useRef([]);
  const timer = useRef(null);
  const audioCtx = useRef(null);
  const raf = useRef(null);
  const stream = useRef(null);

  const cleanup = () => {
    clearInterval(timer.current);
    cancelAnimationFrame(raf.current);
    stream.current?.getTracks().forEach((t) => t.stop());
    audioCtx.current?.close().catch(() => {});
  };
  useEffect(() => () => { cleanup(); if (url) URL.revokeObjectURL(url); }, []); // eslint-disable-line

  async function start() {
    setError("");
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      return setError("Votre navigateur ne permet pas l'enregistrement. Utilisez Chrome ou Edge.");
    }
    try {
      stream.current = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    } catch {
      return setError("Accès au micro refusé. Autorisez le micro (icône 🔒 à gauche de l'adresse), puis réessayez.");
    }
    // Vu-mètre : montre que le micro capte bien la voix
    audioCtx.current = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = audioCtx.current.createAnalyser();
    analyser.fftSize = 512;
    audioCtx.current.createMediaStreamSource(stream.current).connect(analyser);
    const data = new Uint8Array(analyser.fftSize);
    const tick = () => {
      analyser.getByteTimeDomainData(data);
      let peak = 0;
      for (const v of data) peak = Math.max(peak, Math.abs(v - 128));
      setLevel(Math.min(1, peak / 64));
      raf.current = requestAnimationFrame(tick);
    };
    tick();

    const mime = pickMime();
    chunks.current = [];
    rec.current = new MediaRecorder(stream.current, mime ? { mimeType: mime } : undefined);
    rec.current.ondataavailable = (e) => e.data.size && chunks.current.push(e.data);
    rec.current.onstop = () => {
      const type = rec.current.mimeType || mime || "audio/webm";
      const blob = new Blob(chunks.current, { type });
      const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-");
      const file = new File([blob], `enregistrement-${stamp}.${extFor(type)}`, { type });
      if (url) URL.revokeObjectURL(url);
      setUrl(URL.createObjectURL(blob));
      setState("done");
      onRecorded(file);
    };
    rec.current.start(1000);
    setSeconds(0);
    timer.current = setInterval(() => setSeconds((s) => s + 1), 1000);
    setState("recording");
  }

  function stop() {
    rec.current?.state === "recording" && rec.current.stop();
    cleanup();
    setLevel(0);
  }

  function reset() {
    if (url) URL.revokeObjectURL(url);
    setUrl("");
    setSeconds(0);
    setState("idle");
    onRecorded(null);
  }

  return (
    <div className={`recorder ${state}`}>
      {state === "idle" && (
        <>
          <button type="button" className="mic-btn" onClick={start} aria-label="Démarrer l'enregistrement">🎙️</button>
          <strong>Enregistrer la réunion au micro</strong>
          <small>Cliquez, parlez, puis arrêtez : le compte rendu sera généré à partir de l'enregistrement.</small>
        </>
      )}
      {state === "recording" && (
        <>
          <button type="button" className="mic-btn stop" onClick={stop} aria-label="Arrêter l'enregistrement">■</button>
          <strong><span className="rec-dot" /> Enregistrement… {fmt(seconds)}</strong>
          <div className="meter"><span style={{ width: `${Math.round(level * 100)}%` }} /></div>
          <small>La barre bouge quand le micro capte votre voix. Cliquez sur ■ pour arrêter.</small>
        </>
      )}
      {state === "done" && (
        <>
          <strong>✓ Enregistrement prêt ({fmt(seconds)})</strong>
          <audio controls src={url} />
          <button type="button" className="link" onClick={reset}>Recommencer l'enregistrement</button>
        </>
      )}
      {error && <div className="alert error">{error}</div>}
    </div>
  );
}

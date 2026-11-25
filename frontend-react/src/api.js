// Client de l'API FastAPI. En dev, Vite proxifie /api → http://localhost:8000.
const BASE = import.meta.env.VITE_API_URL || "/api";

async function request(method, path, { json, form, params } = {}) {
  const url = new URL(BASE + path, window.location.origin);
  Object.entries(params || {}).forEach(([k, v]) => v !== undefined && v !== "" && url.searchParams.set(k, v));
  const opts = { method, headers: {} };
  if (json) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(json);
  } else if (form) {
    opts.body = form;
  }
  let res;
  try {
    res = await fetch(url, opts);
  } catch {
    throw new Error("API injoignable. Lancez : uvicorn backend.app.main:app --reload");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* corps non JSON */
    }
    throw new Error(`Erreur ${res.status} : ${detail}`);
  }
  if (res.status === 204) return null;
  const type = res.headers.get("content-type") || "";
  return type.includes("application/json") ? res.json() : res.blob();
}

export const api = {
  health: () => request("GET", "/health"),
  analyze: (form) => request("POST", "/meetings/analyze", { form }),
  list: (q) => request("GET", "/meetings", { params: { q } }),
  get: (id) => request("GET", `/meetings/${id}`),
  update: (id, json) => request("PUT", `/meetings/${id}`, { json }),
  remove: (id) => request("DELETE", `/meetings/${id}`),
  exportFile: (id, fmt) => request("GET", `/meetings/${id}/export/${fmt}`),
  email: (id, recipients, attach_pdf) => request("POST", `/meetings/${id}/email`, { json: { recipients, attach_pdf } }),
  actions: (params) => request("GET", "/actions", { params }),
  setStatus: (id, status) => request("PATCH", `/actions/${id}`, { json: { status } }),
};

export const STATUSES = ["à faire", "en cours", "fait"];

export function formatDate(iso) {
  if (!iso) return "";
  const [y, m, d] = iso.split("-");
  return d && m && y ? `${d}/${m}/${y}` : iso;
}

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

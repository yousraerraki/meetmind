import os
from pathlib import Path

os.environ["NLP_MODE"] = "light"
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_meetmind.db")

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.main import app  # noqa: E402

client = TestClient(app)
SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "reunion_projet.txt"


def teardown_module():
    Path("test_meetmind.db").unlink(missing_ok=True)


def test_full_flow():
    r = client.post("/meetings/analyze", data={"title": "Réunion test", "meeting_date": "2026-11-11"},
                    files={"file": ("reunion.txt", SAMPLE.read_bytes())})
    assert r.status_code == 200, r.text
    m = r.json()
    assert m["n_actions"] >= 5 and m["n_decisions"] >= 3

    # historique + recherche
    assert any(x["id"] == m["id"] for x in client.get("/meetings", params={"q": "BARThez"}).json())
    assert client.get("/meetings", params={"q": "introuvable-xyz"}).json() == []

    # édition manuelle
    actions = m["actions"]
    actions[0]["owner"] = "Rania Bennani"
    r = client.put(f"/meetings/{m['id']}", json={"summary": "Résumé corrigé.", "actions": actions})
    assert r.json()["summary"] == "Résumé corrigé."
    assert r.json()["actions"][0]["owner"] == "Rania Bennani"

    # suivi des actions
    aid = r.json()["actions"][0]["id"]
    assert client.patch(f"/actions/{aid}", json={"status": "fait"}).json()["status"] == "fait"
    assert any(a["id"] == aid for a in client.get("/actions", params={"status": "fait"}).json())

    # exports
    pdf = client.get(f"/meetings/{m['id']}/export/pdf")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    docx = client.get(f"/meetings/{m['id']}/export/docx")
    assert docx.status_code == 200 and docx.content[:2] == b"PK"

    # e-mail sans SMTP configuré → 503 explicite
    assert client.post(f"/meetings/{m['id']}/email", json={"recipients": ["a@b.com"]}).status_code == 503

    assert client.delete(f"/meetings/{m['id']}").status_code == 204


def test_text_input_and_errors():
    r = client.post("/meetings/analyze", data={"text": "Karim : On a décidé de valider le budget."})
    assert r.status_code == 200 and r.json()["n_decisions"] == 1
    assert client.post("/meetings/analyze", data={}).status_code == 422
    r = client.post("/meetings/analyze", files={"file": ("x.exe", b"MZ")})
    assert r.status_code == 415


def test_pdf_and_docx_input():
    from backend.app.export.documents import to_docx
    import docx, io
    d = docx.Document()
    for line in SAMPLE.read_text(encoding="utf-8").splitlines():
        d.add_paragraph(line)
    buf = io.BytesIO(); d.save(buf)
    r = client.post("/meetings/analyze", files={"file": ("reunion.docx", buf.getvalue())})
    assert r.status_code == 200 and r.json()["n_actions"] >= 5
    pdf = (Path(__file__).resolve().parents[1] / "data" / "samples" / "reunion_marketing.pdf").read_bytes()
    r = client.post("/meetings/analyze", data={"meeting_date": "2026-11-17"}, files={"file": ("m.pdf", pdf)})
    assert r.status_code == 200, r.text
    assert r.json()["n_actions"] >= 4 and "Leila Benjelloun" in r.json()["participants"]

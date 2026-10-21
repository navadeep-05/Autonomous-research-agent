import os
from pypdf import PdfReader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")


def load_documents(data_dir=DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)
    docs = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        ext = os.path.splitext(name)[1].lower()
        text = ""
        try:
            if ext in (".txt", ".md"):
                with open(path, encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            elif ext == ".pdf":
                text = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
        except Exception:
            continue
        if text.strip():
            docs.append({"name": name, "text": text})
    return docs
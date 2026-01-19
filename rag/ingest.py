from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
import logging

logging.getLogger("pypdf").setLevel(logging.ERROR)

DOCS_DIR = Path("data/docs")

def ingest():
    load_dotenv()
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDF files found in data/docs")

    docs = []
    for pdf in pdfs:
        try:
            loader = PyPDFLoader(str(pdf))
            docs.extend(loader.load()) 
        except Exception as e:
            print(f"[SKIP] {pdf.name}: {e}")


    print(f"Loaded {len(docs)} Pages from PDFs.")
    return docs

if __name__ == "__main__":
    ingest()
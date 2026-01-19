from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.ingest import ingest
from collections import defaultdict
from pathlib import Path

def split_docs(chunk_size=1000, chunk_overlap=150):
    load_dotenv()
    docs = ingest()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    counters = defaultdict(int)
    for d in chunks:
        src = Path(d.metadata.get("source", "unknown")).name
        d.metadata["chunk_id"] = counters[src]
        counters[src] += 1

    print(f"Created {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    split_docs()

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.ingest import ingest


def split_docs(chunk_size=1000, chunk_overlap=150):
    load_dotenv()
    docs = ingest()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    split_docs()

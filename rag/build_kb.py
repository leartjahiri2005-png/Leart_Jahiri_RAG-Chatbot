from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from rag.split import split_docs 

KB_DIR = Path("data/kb_lc")

def main():
    load_dotenv()
    KB_DIR.mkdir(parents=True, exist_ok=True)

    chunks = split_docs(chunk_size=1000, chunk_overlap=150)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(KB_DIR))

    print(f"KB saved to  {KB_DIR}")
    print(f"Total Chunks: {len(chunks)}")

if __name__ == "__main__":
    main()

from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

KB_DIR = Path("data/kb_lc")
DISTANCE_THRESHOLD = 1.2  

def load_vectorstore() -> FAISS:
    """Load FAISS index from data/kb_lc."""
    load_dotenv()
    return FAISS.load_local(
        str(KB_DIR),
        OpenAIEmbeddings(model="text-embedding-3-small"),
        allow_dangerous_deserialization=True,
    )


def retrieve(question: str, k: int = 5, use_mmr: bool = True, source_filter=None):
    """
    Returns: (docs, sources, context)
    - docs: list of LangChain Document
    - sources: list strings "file.pdf - page X - chunk#Y"
    - context: a single string ready to be passed to the LLM
    """
    vs = load_vectorstore()

    if source_filter:
        use_mmr = False

    fetch_k = max(10, k * 3)

    docs_scores = vs.similarity_search_with_score(question, k=fetch_k)
    if not docs_scores:
        return [], [], ""

    if source_filter:
        allowed = set(source_filter)
        docs_scores = [
            (d, s)
            for (d, s) in docs_scores
            if Path(d.metadata.get("source", "unknown")).name in allowed
        ]
        if not docs_scores:
            return [], [], ""

    best_distance = min(s for _, s in docs_scores)
    if best_distance > DISTANCE_THRESHOLD:
        return [], [], ""

    if use_mmr:
        retriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": max(50, fetch_k),
                "lambda_mult": 0.5
            },
        )
        docs = (
            retriever.get_relevant_documents(question)
            if hasattr(retriever, "get_relevant_documents")
            else retriever.invoke(question)
        )
    else:
        docs = [d for (d, _) in docs_scores][:k]

    if not docs:
        return [], [], ""

    context = "\n\n".join(
        f"[{Path(d.metadata.get('source','unknown')).name} "
        f"p.{(d.metadata.get('page',0)+1) if isinstance(d.metadata.get('page',None), int) else '?'}]\n"
        f"{d.page_content}"
        for d in docs
    )
    sources = []
    for d in docs:
        src = Path(d.metadata.get("source", "unknown")).name
        page = d.metadata.get("page", None)
        page_str = (page + 1) if isinstance(page, int) else "?"
        chunk_id = d.metadata.get("chunk_id", "?")
        sources.append(f"{src} - page {page_str} - chunk#{chunk_id}")

    sources = list(dict.fromkeys(sources))
    return docs, sources, context

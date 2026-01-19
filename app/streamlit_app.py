import sys
from pathlib import Path
import glob

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st
from rag.qa import run

NOT_FOUND = "I cannot find this in the provided documents."

st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")

with st.sidebar:
    st.header("⚙️ Settings")

    k = st.slider("Top-K chunks", min_value=1, max_value=10, value=5, step=1)
    show_sources = st.toggle("Show sources", value=True)
    sources_in_expander = st.toggle("Sources in expander", value=True)

    pdf_names = sorted([Path(p).name for p in glob.glob("data/docs/*.pdf")])
    selected_pdfs = st.multiselect("📁 Filter by PDF (optional)", pdf_names, default=[])

    st.divider()
    st.subheader("🧪 Examples")
    examples = [
        "What is credit risk in banking?",
        "Explain AML and KYC.",
        "What is operational risk?",
        "Summarize key points related to fraud, financial crime, or money laundering in the banking documents.",
        "Tell me about Japan (out of scope test).",
    ]
    for q in examples:
        if st.button(q, use_container_width=True):
            st.session_state["prefill"] = q
            st.rerun()

    st.divider()
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Tip: Use the same wording that appears in your PDFs for best retrieval.")

st.title("📄 RAG Chatbot")
st.caption("Grounded answers from your PDFs with citations when available.")
st.caption(f"Top-K: {k}   •   Citations: {'On' if show_sources else 'Off'}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

        if (
            m["role"] == "assistant"
            and show_sources
            and m.get("sources")
            and m["content"].strip() != NOT_FOUND
        ):
            if sources_in_expander:
                with st.expander("Sources"):
                    for s in m["sources"]:
                        st.markdown(f"- {s}")
            else:
                st.markdown("**Sources:**")
                for s in m["sources"]:
                    st.markdown(f"- {s}")

prefill = st.session_state.pop("prefill", None) if "prefill" in st.session_state else None

prompt = st.chat_input("Type your question...")
if prefill and not prompt:
    prompt = prefill

if prompt:
    prev_msgs = st.session_state.messages[-6:] if st.session_state.messages else []
    history_lines = []
    for mm in prev_msgs:
        role = "User" if mm["role"] == "user" else "Assistant"
        history_lines.append(f"{role}: {mm['content']}")
    history_text = "\n".join(history_lines) if history_lines else "(none)"

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            answer, sources = run(
                prompt,
                k=k,
                source_filter=selected_pdfs,
                history_text=history_text,   
            )

        answer = (answer or "").strip()

        if answer == NOT_FOUND:
            st.info(NOT_FOUND)
            if selected_pdfs:
                st.warning("No relevant text found in the selected PDF(s). Try removing the filter or selecting more PDFs.")
        else:
            st.markdown(answer)

        if show_sources and sources and answer != NOT_FOUND:
            if sources_in_expander:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")
            else:
                st.markdown("**Sources:**")
                for s in sources:
                    st.markdown(f"- {s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer if answer else NOT_FOUND, "sources": sources}
    )

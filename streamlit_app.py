"""Local document library and chat interface. No separate API server."""
import logging
import os
from pathlib import Path

import streamlit as st
from rag_runtime import load_engine

st.set_page_config(page_title="Document Explorer", page_icon="\U0001F4DA", layout="wide")


@st.cache_resource
def get_engine(project_root):
    return load_engine(project_root)


def friendly_error(exc):
    logging.getLogger(__name__).error("Operation failed: %s", type(exc).__name__)
    if isinstance(exc, ValueError):
        st.error(str(exc))
    else:
        st.error("Processing could not finish. Check your AI connection and available API quota, then try again. Completed work is saved.")


def render_result(result):
    if result.get("status") != "answered":
        st.info(result.get("answer", "No answer is available yet."))
        return
    st.markdown(result["answer"])
    sources = result.get("sources", [])
    if sources:
        st.caption("Sources from your PDFs")
    displayed_images = set()
    for source in sources:
        metadata = source["metadata"]
        label = f"[{source['label']}] {metadata['source']} | PDF page {metadata['page_number']}"
        with st.expander(label):
            st.text(source["content"])
            image_path = metadata.get("image_path") or metadata.get("page_image_path")
            path = engine.asset(image_path)
            if source.get("image_sent") and path and path not in displayed_images:
                displayed_images.add(path)
                st.image(str(path), caption=f"PDF page {metadata['page_number']}", width="stretch")
    if result.get("missing_images"):
        st.caption("Some saved images are missing. Reprocess their PDF to restore them.")


project_root = os.getenv("MULTIMODALRAG_PROJECT_ROOT", str(Path(__file__).resolve().parent))
try:
    engine = get_engine(project_root)
    ready = engine.api_ready()
    documents = engine.status()
except Exception as exc:
    friendly_error(exc)
    st.stop()

with st.sidebar:
    st.title("Document Explorer")
    st.caption("Text, figures, and answers in one place.")
    st.divider()
    st.subheader("AI connection")
    if ready:
        st.success("Connected")
    else:
        st.info("Prepare documents now. Add your API key to .env when you're ready to search and ask questions.")
    if st.button("Refresh library", width="stretch"):
        st.rerun()
    if st.button("Clear conversation", width="stretch"):
        st.session_state["messages"] = []
        st.rerun()
    st.divider()
    st.caption("Your files and processed content are stored locally.")

st.title("Explore your documents")
st.write("Upload PDFs, discover their text and images, and ask questions with sources.")
indexed = [row for row in documents if row["indexing"] in {"indexed", "partial"}]
columns = st.columns(3)
columns[0].metric("Documents", len(documents))
columns[1].metric("Pages", sum(row["pages"] for row in documents))
columns[2].metric("Searchable documents", len(indexed))

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))
ask_tab, documents_tab = st.tabs(["Ask questions", "Documents"])

with documents_tab:
    st.subheader("Add PDFs")
    with st.form("upload_form", clear_on_submit=True):
        uploads = st.file_uploader("Choose PDFs", type=["pdf"], accept_multiple_files=True)
        use_models = st.checkbox("Make these documents searchable", value=ready, disabled=not ready)
        if ready:
            st.caption("Indexing generates image descriptions and embeddings using your configured API.")
        submitted = st.form_submit_button("Process PDFs", type="primary")
    if submitted:
        if not uploads:
            st.error("Choose at least one PDF first.")
        else:
            results = []
            with st.status("Processing PDFs...", expanded=True) as progress:
                for upload in uploads:
                    st.write(upload.name)
                    try:
                        results.append(engine.ingest_upload(upload.name, upload.getvalue(), use_models=use_models))
                    except Exception as exc:
                        friendly_error(exc)
                progress.update(label="Processing finished", state="complete")
            if results:
                st.session_state["flash"] = f"Processed {len(results)} PDF(s). " + (
                    "They are ready to search." if all(r["status"]=="indexed" for r in results)
                    else "Local preparation is saved. Index them after adding your API key.")
                st.rerun()
    st.divider()
    st.subheader("Saved PDFs")
    saved = engine.saved_pdfs()
    if saved:
        st.caption(f"{len(saved)} saved PDF file(s). Previously completed work is reused.")
        actions = st.columns(2)
        prepare_saved = actions[0].button("Prepare saved PDFs", width="stretch")
        index_saved = actions[1].button("Index saved PDFs", disabled=not ready, width="stretch")
        if prepare_saved or index_saved:
            try:
                with st.spinner("Processing saved PDFs..."):
                    results = engine.ingest_saved(use_models=index_saved)
                st.session_state["flash"] = f"Processed {len(results)} saved document(s)."
                st.rerun()
            except Exception as exc:
                friendly_error(exc)
    if documents:
        labels = {"awaiting_api_key": "Prepared | waiting for key", "indexed": "Searchable",
                  "partial": "Partially searchable", "failed": "Needs retry",
                  "not_started": "Not indexed", "indexing": "Indexing"}
        st.dataframe([{"Document": row["source"], "Pages": row["pages"],
                       "Status": labels.get(row["indexing"], row["indexing"]),
                       "Text chunks": row["text_chunks"], "Image descriptions": row["image_descriptions"]}
                      for row in documents], hide_index=True, width="stretch")
    elif not saved:
        st.info("Your library is empty. Upload your first PDF above.")

with ask_tab:
    options = ["all"] + [row["document_id"] for row in indexed]
    names = {row["document_id"]: f"{row['source']} | {row['pages']} pages" for row in indexed}
    selected = st.selectbox("Search in", options, format_func=lambda value: "All documents" if value=="all" else names[value])
    if not ready:
        st.info("Your API key can be added as the final setup step. Preparing PDFs is available in Documents.")
    elif not indexed:
        st.info("Index your PDFs in Documents to start asking questions.")
    messages = st.session_state.setdefault("messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            if message["role"]=="user":
                st.markdown(message["content"])
            else:
                render_result(message["result"])
    question = st.chat_input("Ask about a passage, diagram, table, or photograph", disabled=not ready or not indexed)
    if question:
        messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Finding sources and reading images..."):
                    result = engine.answer(question, document_id=None if selected=="all" else selected)
                messages.append({"role":"assistant", "result":result})
                render_result(result)
            except Exception as exc:
                friendly_error(exc)

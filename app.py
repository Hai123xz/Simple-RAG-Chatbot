import hashlib
import os
import tempfile
from typing import List

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

st.set_page_config(page_title="Simple PDF RAG", page_icon="📄")
st.title("Simple PDF RAG")

# Session state for the in-memory vectorstore and chunks
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []

st.sidebar.header("Upload & Ingest PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF(s)", type="pdf", accept_multiple_files=True
)
ingest = st.sidebar.button("Ingest")
clear = st.sidebar.button("Clear ingested data")

if clear:
    st.session_state.vectorstore = None
    st.session_state.chunks = []
    st.sidebar.success("Cleared ingested data")


def _safe_doc(doc: Document) -> Document:
    """Ensure a Document has non-empty page_content and metadata dict."""
    content = getattr(doc, "page_content", "") or ""
    if not content.strip():
        return None
    md = getattr(doc, "metadata", {}) or {}
    if not isinstance(md, dict):
        md = {"source": str(md)}
    return Document(page_content=content, metadata=md)


@st.cache_resource
def get_embeddings_model():
    # cached so the model isn't reloaded on every run
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


if ingest and uploaded_files:
    with st.spinner("Loading PDFs and creating vector store..."):
        docs: List[Document] = []
        for uploaded in uploaded_files:
            # write to a temporary file because PyPDFLoader expects a filename
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp.flush()
                tmp_path = tmp.name
            try:
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
            except Exception as e:
                st.sidebar.error(f"Failed to load {uploaded.name}: {e}")
                pages = []
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            for i, p in enumerate(pages):
                safe = _safe_doc(p)
                if not safe:
                    continue
                md = safe.metadata or {}
                md["source"] = uploaded.name
                md["id"] = f"{uploaded.name}-{i}"
                docs.append(Document(page_content=safe.page_content, metadata=md))

        if not docs:
            st.sidebar.warning("No text could be extracted from the uploaded PDFs.")
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)
            # ensure chunk metadata
            for idx, c in enumerate(chunks):
                md = c.metadata or {}
                md["chunk_id"] = f"chunk-{idx}"
                # keep source/id if present
                chunks[idx] = Document(page_content=c.page_content, metadata=md)

            embeddings = get_embeddings_model()
            try:
                # prefer named args (some versions use `embedding` or `embedding_function`)
                vectordb = Chroma.from_documents(
                    documents=chunks, embedding=embeddings, persist_directory=None
                )
            except TypeError:
                # fallback to positional args
                vectordb = Chroma.from_documents(
                    chunks, embeddings, persist_directory=None
                )

            st.session_state.vectorstore = vectordb
            st.session_state.chunks = chunks
            st.sidebar.success(
                f"Ingested {len(chunks)} chunks from {len(uploaded_files)} file(s)"
            )


st.header("Ask questions about the uploaded PDF(s)")
query = st.text_input("Ask about information in the pdf file")


def _get_doc_source(h: Document) -> str:
    md = getattr(h, "metadata", {}) or {}
    return md.get("source", "unknown")


def _get_doc_id(h: Document) -> str:
    md = getattr(h, "metadata", {}) or {}
    if md.get("id"):
        return str(md.get("id"))
    if md.get("chunk_id"):
        return str(md.get("chunk_id"))
    content = getattr(h, "page_content", "") or ""
    return hashlib.md5(content.encode("utf-8")).hexdigest()


if query:
    if st.session_state.vectorstore is None:
        st.warning("Please upload and ingest a PDF first in the sidebar.")
    else:
        with st.spinner("Retrieving relevant passages..."):
            try:
                hits = st.session_state.vectorstore.similarity_search(query, k=4)
            except Exception as e:
                st.error(f"Vector search failed: {e}")
                hits = []

        if not hits:
            st.info("No relevant passages found.")
        else:
            context = "\n\n".join(
                [
                    f"[{_get_doc_source(h)}] {h.page_content}"
                    for h in hits
                    if getattr(h, "page_content", None)
                ]
            )

            system = (
                "You are an assistant that answers questions using ONLY the provided context."
                " Be concise and cite sources in brackets after each factual statement."
            )
            prompt = f"{system}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

            llm = None
            try:
                if os.getenv("USE_OLLAMA") == "1":
                    from langchain_groq import ChatGroq

                    llm = ChatGroq(
                        model="llama-3.1-8b-instant",
                        api_key=st.secrets.get("GROQ_API_KEY", ""),
                        temperature=0,
                    )
                else:
                    from langchain_openai import ChatOpenAI

                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            except Exception as e:
                st.warning(
                    f"LLM not available; returning source passages instead. ({e})"
                )

            if llm is None:
                answer_text = context
            else:
                try:
                    res = llm.invoke(prompt)
                    answer_text = res.content if hasattr(res, "content") else str(res)
                except Exception as e:
                    st.error(f"LLM call failed: {e}")
                    answer_text = context

            st.markdown("**Answer:**")
            st.write(answer_text)

            with st.expander("Sources used"):
                for h in hits:
                    source = _get_doc_source(h)
                    sid = _get_doc_id(h)
                    st.caption(f"{source} – {sid}")
                    st.code(h.page_content or "")

import hashlib
import json
import os
import re
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

load_dotenv()

st.set_page_config(page_title="Banking RAG MVP", page_icon="🏦")
st.title("Banking RAG Chatbot")

# --- Load data ---
with open("data/banking_faqs.json", encoding="utf-8") as f:
    faqs = json.load(f)

docs = [
    Document(
        page_content=f"Q: {x['question']} A: {x['answer']}",
        metadata={"id": x["id"], "source": x["source"], "country": x["country"]},
    )
    for x in faqs
]


# --- Embeddings (local, no API) ---
@st.cache_resource
def get_vectorstore():
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    persist_dir = "./chroma_db"
    try:
        # If a persisted DB already exists, load it so we keep ingested docs
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            vs = Chroma(persist_directory=persist_dir, embedding_function=emb)
        else:
            vs = Chroma.from_documents(docs, emb, persist_directory=persist_dir)
            vs.persist()
    except Exception:
        # Fallback: create from faqs
        vs = Chroma.from_documents(docs, emb, persist_directory=persist_dir)
        vs.persist()
    return vs


vectorstore = get_vectorstore()

# --- Update knowledge base (PDF ingest) ---
st.sidebar.header("Update Knowledge Base")
uploaded = st.sidebar.file_uploader(
    "Upload PDF", type="pdf", accept_multiple_files=True
)

if uploaded and st.sidebar.button("Ingest"):
    with st.spinner("Chunking and embedding..."):
        pdf_docs = []
        for file in uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp.flush()
                loader = PyPDFLoader(tmp.name)
                pdf_docs.extend(loader.load())
                tmp_name = tmp.name
            try:
                os.unlink(tmp_name)
            except Exception:
                pass

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(pdf_docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        persist_dir = "./chroma_db"
        try:
            if os.path.exists(persist_dir) and os.listdir(persist_dir):
                existing = Chroma(
                    persist_directory=persist_dir, embedding_function=embeddings
                )
                # try add_documents if available
                try:
                    existing.add_documents(chunks)
                except Exception:
                    # fallback: recreate DB from chunks
                    vectordb = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory=persist_dir,
                    )
                    vectordb.persist()
                    existing = vectordb
                existing.persist()
                vectordb = existing
            else:
                vectordb = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=persist_dir,
                )
                vectordb.persist()
        except Exception:
            vectordb = Chroma.from_documents(
                documents=chunks, embedding=embeddings, persist_directory=persist_dir
            )
            vectordb.persist()

        # clear cached vectorstore and reload
        try:
            get_vectorstore.clear()
        except Exception:
            pass
        # ensure in-memory variable points to the newly created/updated DB
        try:
            vectorstore = vectordb
        except Exception:
            # fallback to calling get_vectorstore()
            vectorstore = get_vectorstore()
        st.sidebar.success(f"Added {len(chunks)} chunks!")

# --- BM25 for keyword ---
tokenized = [d.page_content.lower().split() for d in docs]
bm25 = BM25Okapi(tokenized)

# --- Simple guardrails ---
PII_PATTERNS = [
    r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b",
    r"\b\d{9,12}\b",
]  # card, account
BLOCKED_TOPICS = ["transfer money", "send money", "wire funds", "make payment"]


def redact_pii(text):
    for pat in PII_PATTERNS:
        text = re.sub(pat, "[REDACTED]", text)
    return text


def is_blocked(q):
    return any(t in q.lower() for t in BLOCKED_TOPICS)


def get_doc_id(doc):
    md = getattr(doc, "metadata", {}) or {}
    if isinstance(md, dict):
        if md.get("id"):
            return str(md.get("id"))
        if md.get("source"):
            return str(md.get("source"))
    content = getattr(doc, "page_content", "") or ""
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def get_doc_source(doc):
    md = getattr(doc, "metadata", {}) or {}
    return md.get("source", "unknown")


def hybrid_retrieve(query, k=3):
    query = redact_pii(query)
    # vector
    vec_hits = vectorstore.similarity_search(query, k=k)
    # bm25
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_idx = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:k]
    bm25_hits = [docs[i] for i in bm25_idx]
    # merge dedupe
    seen = set()
    merged = []
    for h in vec_hits + bm25_hits:
        doc_id = get_doc_id(h)
        if doc_id not in seen:
            merged.append(h)
            seen.add(doc_id)
    return merged[:k]


# --- LLM ---
USE_OLLAMA = os.getenv("USE_OLLAMA") == "1"
if USE_OLLAMA:
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="llama-3.1-8b-instant",  # same family as your local model
        api_key=st.secrets["GROQ_API_KEY"],
        temperature=0,
    )
else:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

SYSTEM = """You are a banking FAQ assistant. Rules:
- Answer ONLY using provided sources. Cite source name in brackets.
- Never execute transactions. If asked, say: 'I can share info, but please use the app to transact after login.'
- If question contains PII, ignore it.
- Be concise, factual. For Vietnam questions, answer in Vietnamese.
"""


def answer(query):
    if is_blocked(query):
        return (
            "⚠️ MVP is FAQ-only. I can't move money here. For transfers, please log in to the banking app.",
            [],
        )
    hits = hybrid_retrieve(query)
    context = "\n\n".join([f"[{get_doc_source(h)}] {h.page_content}" for h in hits])
    prompt = f"{SYSTEM}\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    resp = (
        llm.invoke(prompt).content
        if hasattr(llm.invoke(prompt), "content")
        else llm.invoke(prompt)
    )
    return resp, hits


# --- UI ---
q = st.text_input("Ask about fees, policies, cards...")
if q:
    with st.spinner("Retrieving..."):
        ans, hits = answer(q)
    st.markdown("**Answer:**")
    st.write(ans)
    with st.expander("Sources used"):
        for h in hits:
            st.caption(f"{get_doc_source(h)} – {get_doc_id(h)}")
            st.code(h.page_content)

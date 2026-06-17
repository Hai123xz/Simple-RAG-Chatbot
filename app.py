
import os, json, re
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

load_dotenv()

st.set_page_config(page_title="Banking RAG MVP", page_icon="🏦")
st.title("Banking RAG Chatbot")

# --- Load data ---
with open("data/banking_faqs.json", encoding="utf-8") as f:
    faqs = json.load(f)

docs = [Document(page_content=f"Q: {x['question']} A: {x['answer']}", 
                  metadata={"id":x["id"],"source":x["source"],"country":x["country"]}) for x in faqs]

# --- Embeddings (local, no API) ---
@st.cache_resource
def get_vectorstore():
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = Chroma.from_documents(docs, emb, persist_directory="./chroma_db")
    return vs

vectorstore = get_vectorstore()

# --- BM25 for keyword ---
tokenized = [d.page_content.lower().split() for d in docs]
bm25 = BM25Okapi(tokenized)

# --- Simple guardrails ---
PII_PATTERNS = [r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b", r"\b\d{9,12}\b"] # card, account
BLOCKED_TOPICS = ["transfer money","send money","wire funds","make payment"]

def redact_pii(text):
    for pat in PII_PATTERNS:
        text = re.sub(pat, "[REDACTED]", text)
    return text

def is_blocked(q):
    return any(t in q.lower() for t in BLOCKED_TOPICS)

def hybrid_retrieve(query, k=3):
    query = redact_pii(query)
    # vector
    vec_hits = vectorstore.similarity_search(query, k=k)
    # bm25
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
    bm25_hits = [docs[i] for i in bm25_idx]
    # merge dedupe
    seen=set(); merged=[]
    for h in vec_hits+bm25_hits:
        if h.metadata["id"] not in seen:
            merged.append(h); seen.add(h.metadata["id"])
    return merged[:k]

# --- LLM ---
USE_OLLAMA = os.getenv("USE_OLLAMA")=="1"
if USE_OLLAMA:
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model="llama-3.1-8b-instant",   # same family as your local model
        api_key=st.secrets["GROQ_API_KEY"],
        temperature=0
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
        return "⚠️ MVP is FAQ-only. I can't move money here. For transfers, please log in to the banking app.", []
    hits = hybrid_retrieve(query)
    context = "\n\n".join([f"[{h.metadata['source']}] {h.page_content}" for h in hits])
    prompt = f"{SYSTEM}\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    resp = llm.invoke(prompt).content if hasattr(llm.invoke(prompt), 'content') else llm.invoke(prompt)
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
            st.caption(f"{h.metadata['source']} – {h.metadata['id']}")
            st.code(h.page_content)

import csv
import difflib
import hashlib
import io
import math
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


def _cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors (iterables). Returns -1..1.

    Falls back to 0.0 on error.
    """
    try:
        a_vec = [float(x) for x in a]
        b_vec = [float(x) for x in b]
    except Exception:
        return 0.0
    if not a_vec or not b_vec:
        return 0.0
    # if dims differ, truncate to shortest
    min_len = min(len(a_vec), len(b_vec))
    if len(a_vec) != len(b_vec):
        a_vec = a_vec[:min_len]
        b_vec = b_vec[:min_len]
    dot = sum(x * y for x, y in zip(a_vec, b_vec))
    na = math.sqrt(sum(x * x for x in a_vec))
    nb = math.sqrt(sum(y * y for y in b_vec))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# Top-level tabs: Ask (existing QA) and Test (new evaluation tab)
qa_tab, test_tab = st.tabs(["Ask", "Test"])

with qa_tab:
    st.header("Ask questions about the uploaded PDF(s)")
    query = st.text_input("Ask about information in the pdf file")

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
                prompt = (
                    f"{system}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
                )

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
                        answer_text = (
                            res.content if hasattr(res, "content") else str(res)
                        )
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

with test_tab:
    st.header("Test and evaluate retrieval/QA from a PDF")
    st.write(
        "Upload a single PDF (the source to retrieve from) and a CSV containing test questions and expected answers."
    )
    st.write(
        "CSV columns should include at least a question and expected answer. Optionally include a relevant_doc_id column to evaluate retrieval hits."
    )

    test_pdf = st.file_uploader("PDF for test (single file)", type="pdf")
    test_csv = st.file_uploader("CSV with test questions", type=["csv"])

    k = st.number_input("Top-K retrieval (k)", min_value=1, max_value=20, value=4)
    sim_threshold = st.slider(
        "Similarity threshold for passing (0-1)",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
    )
    use_llm = st.checkbox(
        "Use LLM to generate answers (if unchecked, returns retrieved context)",
        value=True,
    )

    run_tests = st.button("Run tests")

    def _parse_test_csv(uploaded_csv) -> List[dict]:
        # return list of dicts: {question, expected, relevant}
        if uploaded_csv is None:
            return []
        # make sure we read from the start (Streamlit UploadedFile may be at EOF if previously read)
        try:
            uploaded_csv.seek(0)
        except Exception:
            pass

        text = None
        try:
            raw = uploaded_csv.read()
            if isinstance(raw, bytes):
                # try utf-8 with BOM then fall back to latin-1 with replacement
                try:
                    text = raw.decode("utf-8-sig")
                except Exception:
                    text = raw.decode("latin-1", errors="replace")
            else:
                text = str(raw)
        except Exception:
            return []

        if not text or not text.strip():
            return []

        f = io.StringIO(text)
        # try to sniff dialect to handle different delimiters/quoting
        try:
            sample = text[:4096]
            dialect = csv.Sniffer().sniff(sample)
            f.seek(0)
            reader = csv.DictReader(f, dialect=dialect)
        except Exception:
            f.seek(0)
            reader = csv.DictReader(f)

        rows = []
        for r in reader:
            if r is None:
                continue
            # normalize keys (some CSVs may have weird capitalization/spaces)
            lower = {str(k).lower().strip(): (v or "") for k, v in r.items()}
            question = ""
            expected = ""
            relevant = ""
            # heuristics for column names
            for k, v in lower.items():
                if k in ("question", "question_text", "q"):
                    question = v
                if k in ("expected_answer", "expected", "answer", "expectedanswer"):
                    expected = v
                if k in ("relevant_doc_id", "doc_id", "relevant_doc", "relevant_id"):
                    relevant = v
            # fallback: first non-empty column as question
            if not question:
                for k, v in lower.items():
                    if v and not question:
                        question = v
                        break
            rows.append(
                {
                    "question": (question or "").strip(),
                    "expected": (expected or "").strip(),
                    "relevant": (relevant or "").strip(),
                }
            )
        return rows

    def _create_vectordb_from_pdf(uploaded_pdf):
        # returns (vectordb, chunks) or (None, None) on failure
        if uploaded_pdf is None:
            return None, None
        # ensure we read the PDF from the start
        try:
            uploaded_pdf.seek(0)
        except Exception:
            pass

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            tmp.flush()
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
        except Exception as e:
            st.error(f"Failed to load PDF for test: {e}")
            pages = []
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        docs: List[Document] = []
        for i, p in enumerate(pages):
            safe = _safe_doc(p)
            if not safe:
                continue
            md = safe.metadata or {}
            md["source"] = getattr(uploaded_pdf, "name", "test_pdf")
            md["id"] = f"{getattr(uploaded_pdf, 'name', 'test_pdf')}-{i}"
            docs.append(Document(page_content=safe.page_content, metadata=md))

        if not docs:
            st.error("No text could be extracted from the test PDF.")
            return None, None

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        for idx, c in enumerate(chunks):
            md = c.metadata or {}
            md["chunk_id"] = f"chunk-{idx}"
            chunks[idx] = Document(page_content=c.page_content, metadata=md)

        embeddings = get_embeddings_model()
        try:
            vectordb = Chroma.from_documents(
                documents=chunks, embedding=embeddings, persist_directory=None
            )
        except TypeError:
            vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=None)

        return vectordb, chunks

    if run_tests:
        if test_pdf is None or test_csv is None:
            st.error("Please upload both a PDF and a CSV file to run tests.")
        else:
            with st.spinner("Preparing test vectorstore and running evaluations..."):
                rows = _parse_test_csv(test_csv)
                if not rows:
                    st.error("No rows parsed from CSV or CSV had no header/rows.")
                else:
                    vectordb, chunks = _create_vectordb_from_pdf(test_pdf)
                    if vectordb is None:
                        st.error("Failed to create vectorstore from test PDF.")
                    else:
                        # prepare LLM if requested
                        llm = None
                        if use_llm:
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
                                    f"LLM not available; will use retrieved context as answer. ({e})"
                                )
                                llm = None

                        results = []
                        for i, r in enumerate(rows):
                            q = r.get("question", "") or ""
                            expected = r.get("expected", "") or ""
                            relevant = r.get("relevant", "") or ""
                            try:
                                hits = vectordb.similarity_search(q, k=int(k))
                            except Exception as e:
                                st.error(
                                    f"Vector search failed for question {i + 1}: {e}"
                                )
                                hits = []

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
                            prompt = f"{system}\n\nContext:\n{context}\n\nQuestion: {q}\nAnswer:"

                            answer_text = context
                            if llm is not None:
                                try:
                                    res = llm.invoke(prompt)
                                    answer_text = (
                                        res.content
                                        if hasattr(res, "content")
                                        else str(res)
                                    )
                                except Exception as e:
                                    st.warning(
                                        f"LLM call failed for question {i + 1}: {e}"
                                    )
                                    answer_text = context

                            # semantic comparison using embeddings with sentence-level checks
                            sim_full = 0.0
                            sim_sent_max = 0.0
                            sim = 0.0
                            rel_sim_max = 0.0
                            rel_found_top1 = False
                            rel_found_topk = False
                            try:
                                emb_model = get_embeddings_model()
                                expected_emb = None
                                answer_emb = None
                                if expected.strip():
                                    try:
                                        expected_emb = emb_model.embed_query(expected)
                                    except Exception:
                                        try:
                                            expected_emb = emb_model.embed_documents(
                                                [expected]
                                            )[0]
                                        except Exception:
                                            expected_emb = None
                                if (answer_text or "").strip():
                                    try:
                                        answer_emb = emb_model.embed_query(answer_text)
                                    except Exception:
                                        try:
                                            answer_emb = emb_model.embed_documents(
                                                [answer_text]
                                            )[0]
                                        except Exception:
                                            answer_emb = None
                                # full-answer similarity
                                if expected_emb is not None and answer_emb is not None:
                                    cos = _cosine_similarity(expected_emb, answer_emb)
                                    sim_full = (cos + 1.0) / 2.0
                                else:
                                    norm = lambda s: (s or "").strip().lower()
                                    if expected.strip():
                                        sim_full = difflib.SequenceMatcher(
                                            None, norm(expected), norm(answer_text)
                                        ).ratio()
                                    else:
                                        sim_full = 0.0

                                # sentence-level similarity: split answer and compare each sentence
                                import re

                                def _split_sentences(text: str):
                                    if not text or not text.strip():
                                        return []
                                    parts = re.split(r"(?<=[.!?])\s+", text.strip())
                                    return [p.strip() for p in parts if p.strip()]

                                sentences = _split_sentences(answer_text)
                                if sentences and expected_emb is not None:
                                    try:
                                        sent_embs = emb_model.embed_documents(sentences)
                                    except Exception:
                                        sent_embs = []
                                        for s in sentences:
                                            try:
                                                sent_embs.append(
                                                    emb_model.embed_query(s)
                                                )
                                            except Exception:
                                                try:
                                                    sent_embs.append(
                                                        emb_model.embed_documents([s])[
                                                            0
                                                        ]
                                                    )
                                                except Exception:
                                                    sent_embs.append(None)
                                    for se in sent_embs:
                                        if se is None:
                                            continue
                                        cos_s = _cosine_similarity(expected_emb, se)
                                        ssim = (cos_s + 1.0) / 2.0
                                        if ssim > sim_sent_max:
                                            sim_sent_max = ssim

                                # final answer similarity is the max of full and sentence-level
                                sim = max(sim_full, sim_sent_max)

                                # relevant_doc (CSV) similarity to retrieved chunks
                                if relevant.strip():
                                    try:
                                        try:
                                            rel_emb = emb_model.embed_query(relevant)
                                        except Exception:
                                            rel_emb = emb_model.embed_documents(
                                                [relevant]
                                            )[0]
                                        hit_texts = [h.page_content or "" for h in hits]
                                        if hit_texts:
                                            try:
                                                hits_embs = emb_model.embed_documents(
                                                    hit_texts
                                                )
                                            except Exception:
                                                hits_embs = []
                                                for t in hit_texts:
                                                    try:
                                                        hits_embs.append(
                                                            emb_model.embed_query(t)
                                                        )
                                                    except Exception:
                                                        try:
                                                            hits_embs.append(
                                                                emb_model.embed_documents(
                                                                    [t]
                                                                )[0]
                                                            )
                                                        except Exception:
                                                            hits_embs.append(None)
                                            sims = []
                                            for he in hits_embs:
                                                if he is None:
                                                    sims.append(0.0)
                                                    continue
                                                cosh = _cosine_similarity(rel_emb, he)
                                                sims.append((cosh + 1.0) / 2.0)
                                            if sims:
                                                rel_sim_max = max(sims)
                                                rel_found_top1 = sims[0] >= float(
                                                    sim_threshold
                                                )
                                                rel_found_topk = rel_sim_max >= float(
                                                    sim_threshold
                                                )
                                    except Exception:
                                        found_any = any(
                                            (
                                                relevant.lower()
                                                in (h.page_content or "").lower()
                                            )
                                            for h in hits
                                        )
                                        rel_found_topk = found_any
                                        rel_found_top1 = len(hits) >= 1 and (
                                            relevant.lower()
                                            in (hits[0].page_content or "").lower()
                                        )

                                else:
                                    if expected.strip():
                                        exp_emb = expected_emb
                                        if exp_emb is None:
                                            try:
                                                exp_emb = emb_model.embed_query(
                                                    expected
                                                )
                                            except Exception:
                                                try:
                                                    exp_emb = emb_model.embed_documents(
                                                        [expected]
                                                    )[0]
                                                except Exception:
                                                    exp_emb = None
                                        hit_texts = [h.page_content or "" for h in hits]
                                        if hit_texts and exp_emb is not None:
                                            try:
                                                hits_embs = emb_model.embed_documents(
                                                    hit_texts
                                                )
                                            except Exception:
                                                hits_embs = []
                                                for t in hit_texts:
                                                    try:
                                                        hits_embs.append(
                                                            emb_model.embed_query(t)
                                                        )
                                                    except Exception:
                                                        try:
                                                            hits_embs.append(
                                                                emb_model.embed_documents(
                                                                    [t]
                                                                )[0]
                                                            )
                                                        except Exception:
                                                            hits_embs.append(None)
                                            sims = []
                                            for he in hits_embs:
                                                if he is None:
                                                    sims.append(0.0)
                                                    continue
                                                cosh = _cosine_similarity(exp_emb, he)
                                                sims.append((cosh + 1.0) / 2.0)
                                            if sims:
                                                rel_sim_max = max(sims)
                                                rel_found_top1 = sims[0] >= float(
                                                    sim_threshold
                                                )
                                                rel_found_topk = rel_sim_max >= float(
                                                    sim_threshold
                                                )
                            except Exception:
                                # embedding step failed; fallback to substring checks
                                sim = sim_full
                                rel_sim_max = 0.0
                                if relevant.strip():
                                    found_any = any(
                                        (
                                            relevant.lower()
                                            in (h.page_content or "").lower()
                                        )
                                        for h in hits
                                    )
                                    rel_found_topk = found_any
                                    rel_found_top1 = len(hits) >= 1 and (
                                        relevant.lower()
                                        in (hits[0].page_content or "").lower()
                                    )
                                elif expected.strip():
                                    found_any = any(
                                        (
                                            expected.lower()
                                            in (h.page_content or "").lower()
                                        )
                                        for h in hits
                                    )
                                    rel_found_topk = found_any
                                    rel_found_top1 = len(hits) >= 1 and (
                                        expected.lower()
                                        in (hits[0].page_content or "").lower()
                                    )

                            passed = (
                                sim >= float(sim_threshold)
                                if expected.strip()
                                else False
                            )

                            top_ids = [_get_doc_id(h) for h in hits]

                            results.append(
                                {
                                    "question": q,
                                    "expected": expected,
                                    "answer": answer_text,
                                    "similarity": sim,
                                    "sim_full": sim_full,
                                    "sim_sent_max": sim_sent_max,
                                    "passed": passed,
                                    "top_ids": top_ids,
                                    "rel_found_top1": rel_found_top1,
                                    "rel_found_topk": rel_found_topk,
                                    "rel_sim_max": rel_sim_max,
                                    "hits": hits,
                                }
                            )

                        # summary
                        total = len(results)
                        avg_sim = (
                            sum(r["similarity"] for r in results) / total
                            if total
                            else 0.0
                        )
                        pass_rate = (
                            sum(1 for r in results if r["passed"]) / total
                            if total
                            else 0.0
                        )
                        recall1 = (
                            sum(1 for r in results if r["rel_found_top1"]) / total
                            if total
                            else 0.0
                        )
                        recallk = (
                            sum(1 for r in results if r["rel_found_topk"]) / total
                            if total
                            else 0.0
                        )

                        st.subheader("Summary")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Questions", str(total))
                        c2.metric("Avg similarity", f"{avg_sim:.3f}")
                        c3.metric("Pass rate", f"{pass_rate * 100:.1f}%")
                        c4.metric(f"Recall@{k}", f"{recallk * 100:.1f}%")

                        st.markdown("---")
                        st.subheader("Per-question details")
                        for idx, res in enumerate(results, start=1):
                            with st.expander(f"Q{idx}: {res['question']}"):
                                st.write("**Expected:**", res["expected"])
                                st.write("**Answer:**")
                                st.write(res["answer"])
                                st.write(
                                    f"**Full answer similarity:** {res.get('sim_full', 0.0):.3f}"
                                )
                                st.write(
                                    f"**Max sentence similarity:** {res.get('sim_sent_max', 0.0):.3f}"
                                )
                                st.write(
                                    f"**Similarity used:** {res['similarity']:.3f}"
                                )
                                st.write(f"**Passed threshold:** {res['passed']}")
                                st.write(
                                    f"**Relevant found in top-1:** {res['rel_found_top1']}"
                                )
                                st.write(
                                    f"**Relevant found in top-{k}:** {res['rel_found_topk']}"
                                )
                                st.write(
                                    f"**Relevant similarity (max across top-{k}):** {res.get('rel_sim_max', 0.0):.3f}"
                                )
                                st.write("**Top retrieved IDs:**")
                                for hid in res["top_ids"]:
                                    st.caption(hid)
                                if res["hits"]:
                                    with st.expander("Show retrieved chunks"):
                                        for h in res["hits"]:
                                            st.caption(
                                                f"[{_get_doc_source(h)}] {_get_doc_id(h)}"
                                            )
                                            st.code(h.page_content or "")

                        st.success("Testing complete")
